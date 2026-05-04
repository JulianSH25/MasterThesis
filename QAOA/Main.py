import time
import csv
import os
import uuid
import fcntl
import platform
import socket
import resource
import subprocess
from datetime import datetime
import numpy as np

import pandas as pan
from numpy.f2py.auxfuncs import throw_error

from Circuit import QAOACircuit
from ParamOptimisation import BayesianOptimiser, optimise_cobyla, grid_search, optimise_adam
import sys
from pathlib import Path
from utils import classify_graph, get_benchmark_params
from WarmStart import get_warm_start_state, extract_correlations

from instance_generator import instance_generator

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

#edges = [(0, 1), (1, 2)]  # , (2, 3), (3, 4), (4, 5), (5, 6)]  # ring
#weights = [1.0] * len(edges)
#set_of_nodes = {i for k in edges for i in k}
#n = len(set_of_nodes)
#print(set_of_nodes, n)
#p = 20

optimiser_bayesian = False
optimiser_cobyla = True
gridsearch = False

precision = None

benchmark_params: dict = get_benchmark_params()
parameters = benchmark_params["parameter_vector"]

optimiser = benchmark_params["optimiser"].lower()
_initial_energy_prodStates = 0

# TODO move to utils
def get_processor_name():
    try:
        chip_name = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        for line in chip_name.splitlines():
            if "Chip:" in line or "Processor Name:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()

# TODO move
def get_total_ram_gb():
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return None

# TODO move
def get_physical_cores():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count()

# TODO move
def get_logical_cores():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count()

# TODO move
def get_peak_ram_mb():
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 2)
    except Exception:
        return None

def main(
    p: int,
    N_bayes: int | float,
    m = None,
    init_initial_state=False,
    self_init_linegraph=False,
    edges=None,
    weights=None,
    __initial_state__=None,
    fixed_initial_point=None,
    return_initial_point=False,
):
    """
    This method builds and optimises a QAOA instance on a line graph.

    :param m: number of edges for the generated line graph
    :param p: number of QAOA layers
    :param N_bayes: number of optimisation iterations
    :param init_initial_state: whether to inject an SDP-derived warm-start state
    :param self_init_linegraph: whether to use line-graph singlet state preparation
    :return: None
    """
    assert m is not None or edges is not None
    edges = edges if edges is not None else [(i, i + 1) for i in range(m)] # Optionally replace by desired edge list, if a linegraph is not desired
    weights = weights if weights is not None else [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    assert not (init_initial_state and __initial_state__), "Cannot provide both init_initial_state=True and a custom __initial_state__. Please choose one of the two options for a valid benchmark configuration."
    (initial_state, product_states, classical_cut), moment_matrix = get_warm_start_state((edges, weights), n) if init_initial_state and not __initial_state__ else (None, None)
    if __initial_state__ is not None:
        initial_state = __initial_state__
        # BUG if a custom initial state is provided, the classical cut from the SDP warm start is not available for HAMQAOA
    

    warm_start_correlations = extract_correlations(moment_matrix, edges) if moment_matrix is not None else None

    print(f"warm_start_correlations: {warm_start_correlations}")

    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n} nodes, p: {p} layers, N_bayes: {N_bayes} iterations")
    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights)

    initial_energy_prodStates = None

    """if product_states is not None:
        product_states_matrix: dict[tuple[int, int], np.ndarray] = {}
        idx = 0
        for (i, j) in edges:
            product_states_matrix[(i, j)] = product_states[idx]
            idx += 1
        initial_energy_prodStates = QAOA.qaoa_compute_energy(product_states_matrix, edges, weights, (1, 1, 1))  # compute initial energy of the warm-start product states; this is used for the HAMQAOA initial state energy and for the warm-start fallback mechanism in the optimiser
        print(f"Initial energy of warm-start product states: {initial_energy_prodStates}")"""

    benchmark_params: dict = get_benchmark_params()
    QAOA.params = benchmark_params["parameter_vector"]

    QAOA.initial_state = initial_state
    QAOA.classical_WS_cut = classical_cut
    if get_benchmark_params()["debug"]:
        print(f"Initial state set to: {initial_state}") if initial_state is not None else print("No initial state provided.")
        print(f"Classical warm start cut set to: {classical_cut}") if classical_cut is not None else print("No classical warm start cut provided.")
    if warm_start_correlations is not None and benchmark_params["warm_start_correlations"]:
        QAOA.warm_start_correlations = warm_start_correlations
    elif benchmark_params["warm_start_correlations"]:
        raise RuntimeError("Warm start correlations are not available for this benchmark.")

    if benchmark_params["debug"]:
        print(f"Initial state: {initial_state}") if initial_state is not None else print("No initial state provided.")
    QAOA.self_init_linegraph = self_init_linegraph
    QAOA.build_qaoa_maxcut_circuit(add_measurements=False) # TODO check parameter (changed from True to False)
    initial_ws_energy = QAOA.initial_ws_energy

    assert sum([optimiser_bayesian, optimiser_cobyla, gridsearch]) == 1
    minimum_energy = None
    used_initial_point = None
    if optimiser == "bayesian":
        BO = BayesianOptimiser()
        minimum_energy = BO.bayesian_optimisation(QAOA=QAOA, N_bayes=N_bayes, no_layers=p)
    elif optimiser == "cobyla":
        correlations=warm_start_correlations if benchmark_params["use_correlations_as_initial_params"] else None
        returned_energy = optimise_cobyla(
            QAOA=QAOA,
            no_layers=p,
            max_iter=N_bayes,
            correlations=correlations,
            x0=fixed_initial_point,
        )
        minimum_energy = -returned_energy.fun
        used_initial_point = getattr(returned_energy, "initial_point", None)
    elif optimiser == "adam":
        returned_energy = optimise_adam(QAOA=QAOA, no_layers=p, steps=N_bayes, x0=fixed_initial_point, learning_rate=benchmark_params["learning_rate_adam"])
        minimum_energy = -returned_energy.fun
        used_initial_point = getattr(returned_energy, "initial_point", None)
    elif optimiser == "gridsearch":
        minimum_energy = grid_search(QAOA, p, precision=precision)
    else:
        throw_error(f"No valid optimiser specified. Received: {optimiser}.")

    if initial_ws_energy is not None and minimum_energy is not None and minimum_energy < initial_ws_energy:
        print(
            "Warm-start fallback: optimiser result was below the SDP warm-start energy; "
            "using the warm-start energy instead."
        )
        minimum_energy = initial_ws_energy

    print(minimum_energy)

    if return_initial_point:
        return minimum_energy, used_initial_point, initial_ws_energy, initial_energy_prodStates
    return minimum_energy, initial_ws_energy, initial_energy_prodStates

def return_optimal_line(n):
    df = pan.read_csv("optimal_results_qaoa.csv", skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

def return_optimal_cycle(n):
    df = pan.read_csv("optimal_results_qaoa_circle.csv", skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

def return_optimal_fully_connected(n):
    df = pan.read_csv("optimal_results_qaoa_complete.csv", skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

if __name__ == "__main__":
    # Run QAOA benchmark:
    # python Main.py <precision/iterations> <p: #layers/circuit depth> <n_start> <n_end> [output_csv]
    global_starttime = time.time()
    time_section = time.time()
    time_sections = {}
    print(f"Starting QAOA benchmark with global start time: {global_starttime}")

    # Load benchmark parameters from json
    parameter_settings = get_benchmark_params()
    singlet_injection = parameter_settings["singlet_injection"]
    warm_start = parameter_settings["warm_start"]
    assert not (singlet_injection and warm_start), "Singlet injection and warm start cannot be used simultaneously, as they both modify the initial state preparation. Please choose one of the two options for a valid benchmark configuration."
    print(f"Benchmark parameters: {parameter_settings}, Running QAOA with equal superposition")
    # Initialise some variables to store results and metadata about the benchmark run.
    results = {}
    results_010101 = {}
    duration = {}
    precision = float(sys.argv[1])
    p = int(sys.argv[2])
    
    # Keep one shared CSV file and append safely across parallel runs.
    debug_csv_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else f"qaoa_results_{optimiser}"
    csv_filename = f"{debug_csv_path}.csv"
    
    # Generate unique hash ID for this benchmark run
    run_id = str(uuid.uuid4())[:8]
    processor_name = get_processor_name()
    hostname = socket.gethostname()
    total_ram_gb = get_total_ram_gb()
    physical_cores = get_physical_cores()
    logical_cores = get_logical_cores()
    python_version = platform.python_version()
    print(f"Benchmark run ID: {run_id}")
    print(f"Processor: {processor_name}")
    print(f"Hostname: {hostname}")
    print(f"Total RAM (GB): {total_ram_gb}")
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")
    print(f"Python version: {python_version}")

    base_fieldnames = ['run_id', 'n', 'm', 'p', 'precision/iterations', 'singlet_injection', 'warm_start',
                       'parameter_vector', 'initial_ws_energy', 'initial_ws_energy_prodStates', 'initial_ws_energy_010101', 'QAOA_improvement_over_SDP', 'result', 'result_010101', 'approx_ratio', 'approx_ratio_010101', 'diff. approx. ratio', 'sdp ws greater', 'duration_seconds', 'finished_at',
                       'processor', 'hostname', 'total_ram_gb', 'physical_cores', 'logical_cores',
                       'python_version', 'peak_ram_mb']

    # start from base_fieldnames and insert scalar params next; reproducibility columns appended last
    fieldnames = list(base_fieldnames)
    for key in parameter_settings:
        if key not in fieldnames and isinstance(parameter_settings[key], (str, int, float, bool)):
            fieldnames.append(key)

    scalar_params = {
        k: v for k, v in parameter_settings.items()
        if isinstance(v, (str, int, float, bool))
    }

    # reproducibility columns (may be large) should be last
    fieldnames.extend(['hog_graph_index', 'edges', 'weights'])

    # XXX TIME: log time taken for initialisation and parameter loading
    time_sections["initialisation"] = time.time() - time_section
    print(f"Initialisation time: {time_sections['initialisation']:.2f} seconds; started at {time_sections} and finished at {time.time()}")

    for n in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
        time_section = time.time()
        # make graph_generation_type check case-insensitive
        assert isinstance(parameter_settings.get("graph_generation_type"), str), "graph_generation_type must be a string"
        graph_generation_type = parameter_settings["graph_generation_type"].lower()
        assert graph_generation_type in ("line", "cycle", "complete", "random", "hog")

        # Determine whether instances are considered weighted in this benchmark
        weighted_flag = bool(parameter_settings.get("weighted", False))

        # Delegate all instance generation (including HOG) to instance_generator
        edges, weights = instance_generator(
            type=graph_generation_type,
            n=n,
            weighted=weighted_flag,
            random_weights=bool(parameter_settings.get("random_weights", False)),
        )
        m = len(edges)
        node_count = len({i for edge in edges for i in edge})

        # XXX Time
        time_sections[f"instance_generation_n_{n}"] = time.time() - time_section
        print(f"Instance generation for n={n} took {time_sections[f'instance_generation_n_{n}']:.2f} seconds; started at {time_section} and finished at {time.time()}")

        start_time = time.time()
        print(f"Running QAOA for n={node_count} nodes, m={len(edges)} edges...; Max iterations: {int(precision)}")
        # XXX Time
        time_section = time.time()
 
        results[n], shared_initial_point, initial_ws_energy, initial_energy_prodStates = main(
            p=p,
            N_bayes=int(precision),
            self_init_linegraph=singlet_injection,
            init_initial_state=warm_start,
            edges=edges,
            weights=weights,
            return_initial_point=True,
        )
        # XXX Time
        time_sections[f"qaoa_optimisation_n_{n}"] = time.time() - time_section
        print(f"QAOA optimisation for n={n} took {time_sections[f'qaoa_optimisation_n_{n}']:.2f} seconds; started at {time_section} and finished at {time.time()}")


        if parameter_settings["compare_with_010101"]:
            # XXX Time
            time_section = time.time()

            bitstring = ''.join(['1' if i % 2 else '0' for i in range(node_count)])

            state = np.zeros(2**node_count, dtype=complex)
            index = int(bitstring, 2)
            state[index] = 1.0

            initial_state = state

            print("Reusing optimiser initial parameters for 010101 comparison run.")
            results_010101[n], initial_ws_energy_010101, __initial_energy_prodStates = main(
                p=p,
                N_bayes=int(precision),
                edges=edges,
                weights=weights,
                __initial_state__=initial_state,
                fixed_initial_point=shared_initial_point,
            )
            # XXX Time
            time_sections[f"qaoa_optimisation_010101_n_{n}"] = time.time() - time_section
            print(f"QAOA optimisation for 010101 state at n={n} took {time_sections[f'qaoa_optimisation_010101_n_{n}']:.2f} seconds; started at {time_section} and finished at {time.time()}")  

        elapsed_time = time.time() - start_time
        # XXX Time
        time_section = time.time()

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        peak_ram_mb = get_peak_ram_mb()
        print(f"Finished QAOA for n={node_count} nodes, m={m} edges at {finished_at} on {processor_name}. Time taken: {elapsed_time:.2f} seconds. Hours: {elapsed_time / 3600:.2f} hours. Peak RAM: {peak_ram_mb} MB.")
        duration[node_count] = elapsed_time
        approx_ratio = None
        approx_ratio_010101 = None
        # BUG illegally classifies single edge line graphs as complete graphs
        try:
            if classify_graph(edges) == "line":
                print("Computing line approximation ratio")
                approx_ratio = results[n] / return_optimal_line(node_count)
                approx_ratio_010101 = results_010101[n] / return_optimal_line(node_count) if parameter_settings["compare_with_010101"] else None
            elif classify_graph(edges) == "cycle":
                print("Computing cycle approximation ratio")
                approx_ratio = results[n] / return_optimal_cycle(node_count)
                approx_ratio_010101 = results_010101[n] / return_optimal_cycle(node_count) if parameter_settings["compare_with_010101"] else None
                #pass
            elif classify_graph(edges) == "complete":
                print("Computing complete graph approximation ratio")
                approx_ratio = results[n] / return_optimal_fully_connected(node_count)
                approx_ratio_010101 = results_010101[n] / return_optimal_fully_connected(node_count) if parameter_settings["compare_with_010101"] else None
                #pass
            else:
                print("Unknown graph type for approximation ratio calculation; skipping approx ratio computation.")
        except Exception as e:
            print(f"Error during approximation ratio calculation: {e}. Skipping approx ratio computation for n={n}.")

        row = {
            'run_id': run_id,
            'n': node_count,
            'm': m,
            'p': p,
            'precision/iterations': precision,
            'singlet_injection': singlet_injection,
            'warm_start': warm_start,
            'parameter_vector': str(parameter_settings["parameter_vector"]),
            'initial_ws_energy': initial_ws_energy,
            'initial_ws_energy_prodStates': initial_energy_prodStates,
            'initial_ws_energy_010101': initial_ws_energy_010101 if parameter_settings["compare_with_010101"] else None,
            'QAOA_improvement_over_SDP': results[n] - initial_ws_energy if initial_ws_energy is not None else None,
            'result': results[n],
            'result_010101': results_010101[n] if parameter_settings["compare_with_010101"] else None,
            'approx_ratio': approx_ratio,
            'approx_ratio_010101': approx_ratio_010101,
            'diff. approx. ratio': round(approx_ratio, 6) - round(approx_ratio_010101, 6) if parameter_settings["compare_with_010101"] else None,
            'sdp ws greater': round(approx_ratio, 6) >= round(approx_ratio_010101, 6) if parameter_settings["compare_with_010101"] else None,
            'duration_seconds': elapsed_time,
            'finished_at': finished_at,
            'processor': processor_name,
            'hostname': hostname,
            'total_ram_gb': total_ram_gb,
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'python_version': python_version,
            'peak_ram_mb': peak_ram_mb,
            'hog_graph_index': n if graph_generation_type == "hog" else None
        }

        # include edges and weights (weights only meaningful when benchmark flagged as weighted)
        row['edges'] = str(edges)
        row['weights'] = str(weights) if weighted_flag else None

        # inject scalar params automatically
        for key, value in scalar_params.items():
            if key not in row and key in fieldnames:
                row[key] = value

        # Simple append to CSV file
       # write_header = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
        
        # Append to CSV under an exclusive file lock so parallel processes
        # cannot both decide to write the header at the same time.
        with open(csv_filename, 'a+', newline='') as csvfile:
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
            try:
                csvfile.seek(0, os.SEEK_END)
                write_header = csvfile.tell() == 0
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
                os.fsync(csvfile.fileno())
            finally:
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

        # XXX Time
        time_sections[f"csv_writing_n_{n}"] = time.time() - time_section
        print(f"CSV writing for n={n} took {time_sections[f'csv_writing_n_{n}']:.2f} seconds; started at {time_section} and finished at {time.time()}")

    # XXX Time
    time_section = time.time()

    print(f"results: {results}")
    print(f"durations: {duration}")
    print(f"Results saved to: {csv_filename}")

    global_endtime = time.time()
    print(f"Global end time: {global_endtime}")

    # XXX Time
    time_sections["finalisation"] = time.time() - time_section
    print(f"Finalisation time: {time_sections['finalisation']:.2f} seconds; started at {time_section} and finished at {time.time()}")
    sum_sections_time = sum(time_sections.values())
    print(f"Sum of all section times: {sum_sections_time:.2f} seconds")
    time_sections["total_time"] = global_endtime - global_starttime
    print(f"run_id: {run_id}")
    with open(f"time/{run_id}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['section', 'duration_seconds'])
        writer.writeheader()
        for section, duration_sec in time_sections.items():
            writer.writerow({'section': section, 'duration_seconds': round(duration_sec, 2)})
    # Run shell file: sudo nohup zsh qaoa_benchmarks.sh > logs/launcher_$(date +'%Y%m%d_%H%M%S').log 2>&1 &