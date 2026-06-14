import time
import csv
import os
from typing import Any, Literal
import uuid
import fcntl
import platform
import socket
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
import pandas as pan
from numpy.f2py.auxfuncs import throw_error
from qiskit.quantum_info import Statevector

# Local imports
from Circuit import QAOACircuit
from ParamOptimisation import BayesianOptimiser, optimise_cobyla, grid_search, optimise_adam, optimise_exact
from Utils import classify_graph, get_benchmark_params
from Utils import get_processor_name, get_total_ram_gb, get_physical_cores, get_logical_cores, get_peak_ram_mb
from Utils import get_exact_result_from_misc, is_triangle_free, is_3_regular
from WarmStart import get_warm_start_state, extract_correlations
from InstanceGenerator import instance_generator

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

precision = None
last_sdp_objective_value = None

benchmark_params: dict = get_benchmark_params()
parameters = benchmark_params["parameter_vector"]
debug = benchmark_params.get("debug", False)

benchmark_repeat_idx = os.environ.get("BENCHMARK_REPEAT_IDX")
benchmark_repeat_idx = int(benchmark_repeat_idx) if benchmark_repeat_idx not in (None, "") else None

sdp_seed_override = os.environ.get("SDP_SEED_OVERRIDE")
sdp_seed_override = int(sdp_seed_override) if sdp_seed_override not in (None, "") else None

optimiser = benchmark_params["optimiser"].lower()
normalisation_factor = benchmark_params["lasserre_level"] if optimiser != "exact" else 1


def _optimal_results_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "optimal_results" / filename


# --- Utility functions for config handling ---
def config_bool(params: dict, key: str, default: bool = False) -> bool:
    value = params.get(key)
    return default if value is None else bool(value)


def resolve_graph_generation_type(params: dict) -> str:
    graph_generation_type = params.get("graph_generation_type")
    assert isinstance(graph_generation_type, str), "graph_generation_type must be a string"
    graph_generation_type = graph_generation_type.lower()
    assert graph_generation_type in ("line", "cycle", "complete", "random", "hog")
    return graph_generation_type

def main(p: int, N_bayes: int | float, m = None, init_initial_state=False, self_init_linegraph=False, edges=None,
    weights=None, __initial_state__=None, fixed_initial_point=None, return_initial_point=False, graph_generation_type: str = "unknown") -> tuple[float | Literal[0] | None, Any | None, float | None, float | Any | None, float | None] | tuple[float | Literal[0] | None, float | None, float | Any | None, float | None]:
    global last_sdp_objective_value
    """
    This method builds and optimises a QAOA instance on a line graph.

    :param p: number of QAOA layers
    :param N_bayes: number of optimisation iterations
    :param m: number of edges for the generated line graph;
    :param init_initial_state: whether to inject an SDP-derived warm-start state
    :param self_init_linegraph: whether to use line-graph singlet state preparation
    :param edges: optional edge list to override instance generation; if None, instance will be generated according to benchmark parameters
    :param weights: optional weight list to override instance generation; if None, instance will be generated according to benchmark parameters
    :param __initial_state__: optional custom initial statevector to override both the SDP warm start and the line-graph singlet state preparation; if provided, this statevector will be used as the initial state for QAOA instead of any warm-start state;
    :param fixed_initial_point: optional initial parameter point for COBYLA or Adam optimisers; if None, optimiser will use its default initialisation strategy
    :param return_initial_point: whether to return the initial parameter point used by the optimiser (relevant for COBYLA and Adam optimisers when fixed_initial_point is None, as they will use a default initialisation strategy in that case)
    """
    assert m is not None or edges is not None # XXX sanity check
    edges = edges if edges is not None else [(i, i + 1) for i in range(m)] # Optionally replace by desired edge list, if a linegraph is not desired
    weights = weights if weights is not None else [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    assert not (init_initial_state and __initial_state__), "Cannot provide both init_initial_state=True and a custom __initial_state__. Please choose one of the two options for a valid benchmark configuration."

    (initial_state, product_states, classical_cut), moment_matrix, warm_start_result = (get_warm_start_state((edges, weights), n) if init_initial_state and not __initial_state__ else ((None, None, None), None, None))
    if __initial_state__ is not None:
        initial_state = __initial_state__
        if str(benchmark_params.get("circuit_type") or "").lower() == "hamqaoa":
            raise Warning("A custom initial state was provided for a HAMQAOA circuit. The classical cut from the SDP warm start will not be available for this run, and any benchmark results should be interpreted accordingly, especially when comparing against runs that do use the SDP warm start.")

    warm_start_correlations = extract_correlations(moment_matrix, edges) if moment_matrix is not None else None
    print(f"warm_start_correlations: {warm_start_correlations}") if debug else None
    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None # XXX sanity check
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n} nodes, p: {p} layers, N_bayes: {N_bayes} iterations")

    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights) # Create QAOA circuit instance with specified parameters

    QAOA.params = [1, 1, 1] if optimiser == "exact" and benchmark_params.get("parameter_vector") is None else benchmark_params["parameter_vector"]
    warm_start_mode = str(benchmark_params.get("warm_start_mode") or "standard").lower()
    energy_audit = config_bool(benchmark_params, "debug")

    initial_energy_prodStates = None
    initial_sdp_statevector_energy = None
    sdp_objective_value = warm_start_result.get("sdp_objective_value") if warm_start_result is not None else None
    sdp_objective_value_normalized = sdp_objective_value
    algorithm17_actual_energy = warm_start_result.get("actual_energy") if warm_start_result is not None else None
    algorithm17_lower_bound_energy = warm_start_result.get("lower_bound_energy") if warm_start_result is not None else None
    rounded_solution_energy = warm_start_result.get("rounded_solution_energy") if warm_start_result is not None else None
    last_sdp_objective_value = sdp_objective_value
    if product_states is not None:
        edge_marginals = {
            (i, j): np.kron(product_states[i], product_states[j])
            for (i, j) in edges
        }
        initial_energy_prodStates = QAOACircuit.qaoa_compute_energy(
            edge_marginals,
            edges,
            weights,
            params=tuple(QAOA.params),
            lasserre_level=benchmark_params.get("lasserre_level")
        ).real
        if energy_audit:
            print(f"Initial energy from SDP product states: {initial_energy_prodStates}")

    if initial_state is not None and (not init_initial_state or warm_start_mode != "entangled"):
        initial_sdp_statevector_energy = QAOA.compute_energy_from_statevector(
            Statevector(np.asarray(initial_state, dtype=complex))
        )
        if energy_audit:
            print(f"Initial energy from warm-start statevector: {initial_sdp_statevector_energy}")

    QAOA.initial_state = initial_state # assign circuit parameter: initial statevector for statevector-based energy evaluation and warm-starting; if None, circuit will use equal superposition initial state
    QAOA.classical_WS_cut = classical_cut # assign circuit parameter: classical warm start cut from SDP solution, used for certain Circuit setups (influences rotation angles in QAOA) and for audit comparisons; if None, no classical warm start cut will be used
    if debug:
        print(f"Initial state set to: {initial_state}") if initial_state is not None else print("No initial state provided.")
        print(f"Classical warm start cut set to: {classical_cut}") if classical_cut is not None else print("No classical warm start cut provided.")

    use_correlations = init_initial_state and (
        warm_start_mode in {"amplified", "entangled"}
        or config_bool(benchmark_params, "use_correlations_as_initial_params")
    )
    if warm_start_correlations is not None and use_correlations:
        QAOA.warm_start_correlations = warm_start_correlations # assign circuit parameter: assign warm start correlations which can then be used for warm start initialisation or as initial parameters
    elif use_correlations:
        raise RuntimeError("Warm start correlations are required for warm_start_mode or correlation-based initial params, but none were available.")

    print(f"Initial state: {initial_state}") if initial_state is not None else print("No initial state provided.") if debug else None

    QAOA.self_init_linegraph = self_init_linegraph # assign circuit parameter: whether to use line-graph singlet state preparation

    """[1] Construct the QAOA circuit"""
    QAOA.build_qaoa_maxcut_circuit(add_measurements=False)

    initial_ws_energy = QAOA.initial_ws_energy # assign circuit parameter: initial warm-start energy from SDP solution, used for audit comparisons
    print(f"Energy audit summary: statevector={initial_sdp_statevector_energy}, product_states={initial_energy_prodStates}, initial_ws_energy={initial_ws_energy}, sdp_objective={sdp_objective_value}, normalized_sdp_objective={sdp_objective_value_normalized}, rounded_solution_energy={rounded_solution_energy}, algorithm17_actual_energy={algorithm17_actual_energy}, algorithm17_lower_bound_energy={algorithm17_lower_bound_energy}") if energy_audit else None

    minimum_energy = None
    used_initial_point = None
    exact_mode = False
    """[2] Start the QAOA evaluation loop with the specified optimiser"""
    if optimiser == "bayesian":
        BO = BayesianOptimiser()
        minimum_energy = BO.bayesian_optimisation(QAOA=QAOA, N_bayes=N_bayes, no_layers=p)
    elif optimiser == "cobyla":
        correlations=warm_start_correlations if config_bool(benchmark_params, "use_correlations_as_initial_params") else None
        returned_energy = optimise_cobyla(QAOA=QAOA, no_layers=p, max_iter=N_bayes, correlations=correlations, x0=fixed_initial_point)
        minimum_energy = -returned_energy.fun
        used_initial_point = getattr(returned_energy, "initial_point", None)
    elif optimiser == "adam":
        returned_energy = optimise_adam(QAOA=QAOA, no_layers=p, steps=N_bayes, x0=fixed_initial_point, learning_rate=benchmark_params["learning_rate_adam"])
        minimum_energy = -returned_energy.fun
        used_initial_point = getattr(returned_energy, "initial_point", None)
    elif optimiser == "gridsearch":
        minimum_energy = grid_search(QAOA, p, precision=precision)
    elif optimiser == "exact":
        exact_mode = True
        minimum_energy = optimise_exact(QAOA, graph_generation_type=graph_generation_type)
    else:
        throw_error(f"No valid optimiser specified. Received: {optimiser}.")

    if (not exact_mode
        and initial_ws_energy is not None
        and minimum_energy is not None
        and minimum_energy < initial_ws_energy):
        print(
            "Warm-start fallback: optimiser result was below the SDP warm-start energy; "
            "using the warm-start energy instead."
        )
        minimum_energy = initial_ws_energy

    print(minimum_energy)

    if return_initial_point:
        return (
            minimum_energy,
            used_initial_point,
            initial_ws_energy,
            initial_energy_prodStates,
            initial_sdp_statevector_energy,
            warm_start_result,
        )
    return minimum_energy, initial_ws_energy, initial_energy_prodStates, initial_sdp_statevector_energy, warm_start_result

def return_optimal_line(n):
    df = pan.read_csv(_optimal_results_path("optimal_results_qaoa.csv"), skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

def return_optimal_cycle(n):
    df = pan.read_csv(_optimal_results_path("optimal_results_qaoa_circle.csv"), skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

def return_optimal_fully_connected(n):
    df = pan.read_csv(_optimal_results_path("optimal_results_qaoa_complete.csv"), skipinitialspace=True)
    val = df.loc[df["n"] == n, "result"].item()
    return val

if __name__ == "__main__":
    # Run QAOA benchmark:
    # python Main.py <precision/iterations> <p: #layers/circuit depth> <n_start> <n_end> [output_csv]
    global_starttime = time.time()
    time_section = time.time()
    time_sections = {}
    lasserre_level = benchmark_params.get("lasserre_level")
    print(f"Starting QAOA benchmark with global start time: {global_starttime}")

    # Use the already loaded benchmark parameters.
    parameter_settings = benchmark_params
    singlet_injection = config_bool(parameter_settings, "singlet_injection")
    warm_start = config_bool(parameter_settings, "warm_start")
    compare_with_010101 = config_bool(parameter_settings, "compare_with_010101")
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
    print(f"Benchmark repeat index: {benchmark_repeat_idx}")
    print(f"SDP seed override: {sdp_seed_override}")
    print(f"Processor: {processor_name}")
    print(f"Hostname: {hostname}")
    print(f"Total RAM (GB): {total_ram_gb}")
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")
    print(f"Python version: {python_version}")

    base_fieldnames = ['run_id', 'n', 'm', 'p', 'precision/iterations', 'singlet_injection', 'warm_start',
                       'parameter_vector', 'optimal_result', 'sdp_objective_value_step1', 'sdp_objective_value_king_normalized_step1', 'algorithm17_actual_energy', 'algorithm17_lower_bound_energy', 'initial_ws_energy_prodStates_step2', 'initial_sdp_statevector_energy', 'initial_sdp_statevec_ratio', 'initial_ws_energy_010101', 'QAOA_improvement_over_SDP_statevectorEnergy', 'QAOA_improvement_over_SDP_prodStatesEnergy', 'result', 'result_010101', 'approx_ratio', 'approx_ratio_010101', 'diff. approx. ratio', 'sdp ws greater', 'duration_seconds', 'full duration_seconds', 'finished_at',
                       'processor', 'hostname', 'total_ram_gb', 'physical_cores', 'logical_cores',
                       'python_version', 'peak_ram_mb', 'Instance_is_triangle_free', 'Instance_is_3_regular']

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
    fieldnames.extend(['benchmark_repeat_idx', 'sdp_seed', 'hog_graph_index', 'edges', 'weights'])

    # XXX TIME: log time taken for initialisation and parameter loading
    time_sections["initialisation"] = time.time() - time_section
    print(f"Initialisation time: {time_sections['initialisation']:.2f} seconds; started at stardate {time_sections} and finished at stardate{time.time()}")

    for n in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
        time_section = time.time()
        graph_generation_type = resolve_graph_generation_type(parameter_settings)

        # Determine whether instances are considered weighted in this benchmark
        weighted_flag = config_bool(parameter_settings, "weighted")

        # Delegate all instance generation (including HOG) to instance_generator
        edges, weights = instance_generator(
            type=graph_generation_type,
            n=n,
            weighted=weighted_flag,
            random_weights=config_bool(parameter_settings, "random_weights"),
        )
        m = len(edges)
        node_count = len({i for edge in edges for i in edge})

        # XXX Time
        time_sections[f"instance_generation_n_{n}"] = time.time() - time_section
        print(f"Instance generation for n={n} took {time_sections[f'instance_generation_n_{n}']:.2f} seconds; started at stardate {time_section} and finished at stardate {time.time()}")

        start_time = time.time()
        print(f"Running QAOA for n={node_count} nodes, m={len(edges)} edges...; Max iterations: {int(precision)}")
        # XXX Time
        time_section = time.time()

        results[n], shared_initial_point, initial_ws_energy, initial_energy_prodStates, initial_sdp_statevector_energy, warm_start_result = main(
            p=p,
            N_bayes=int(precision),
            self_init_linegraph=singlet_injection,
            init_initial_state=warm_start,
            edges=edges,
            weights=weights,
            return_initial_point=True,
            graph_generation_type=graph_generation_type
        )
        sdp_objective_value = (
            warm_start_result.get("sdp_objective_value")
            if warm_start_result is not None
            else last_sdp_objective_value
        )

        sdp_objective_value_normalized = (
            warm_start_result.get("sdp_objective_value_normalized")
            if warm_start_result is not None
            else sdp_objective_value
        )

        algorithm17_actual_energy = (
            warm_start_result.get("actual_energy")
            if warm_start_result is not None
            else None
        )

        algorithm17_lower_bound_energy = (
            warm_start_result.get("lower_bound_energy")
            if warm_start_result is not None
            else None
        )
        # XXX Time
        time_sections[f"qaoa_optimisation_n_{n}"] = time.time() - time_section
        print(f"QAOA optimisation for n={n} took {time_sections[f'qaoa_optimisation_n_{n}']:.2f} seconds; started at stardate {time_section} and finished at stardate {time.time()}")


        initial_ws_energy_010101 = None
        if compare_with_010101:
            # XXX Time
            time_section = time.time()

            bitstring = ''.join(['1' if i % 2 else '0' for i in range(node_count)])

            state = np.zeros(2**node_count, dtype=complex)
            index = int(bitstring, 2)
            state[index] = 1.0

            initial_state = state

            print("Reusing optimiser initial parameters for 010101 comparison run.")
            results_010101[n], initial_ws_energy_010101, __initial_energy_prodStates, __initial_energy_statevector, _ = main(
                p=p,
                N_bayes=int(precision),
                edges=edges,
                weights=weights,
                __initial_state__=initial_state,
                fixed_initial_point=shared_initial_point
            )
            # XXX Time
            time_sections[f"qaoa_optimisation_010101_n_{n}"] = time.time() - time_section
            print(f"QAOA optimisation for 010101 state at n={n} took {time_sections[f'qaoa_optimisation_010101_n_{n}']:.2f} seconds; started at stardate {time_section} and finished at stardate {time.time()}")

        elapsed_time = time.time() - start_time
        # XXX Time
        time_section = time.time()

        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        peak_ram_mb = get_peak_ram_mb()
        print(f"Finished QAOA for n={node_count} nodes, m={m} edges at {finished_at} on {processor_name}. Time taken: {elapsed_time:.2f} seconds. Hours: {elapsed_time / 3600:.2f} hours. Peak RAM: {peak_ram_mb} MB.")
        duration[node_count] = elapsed_time
        _optimal = None
        approx_ratio = None
        approx_ratio_010101 = None
        # BUG illegally classifies single edge line graphs as complete graphs
        try:
            graph_type = classify_graph(edges)
            if graph_type == "line":
                print("Computing line approximation ratio")
                optimal = return_optimal_line(node_count)
                _optimal = optimal
                if optimal is None:
                    print("Missing optimal_results_qaoa.csv; skipping line approximation ratio.")
                else:
                    approx_ratio = (results[n] / normalisation_factor) / optimal
                    approx_ratio_010101 = (results_010101[n] / normalisation_factor) / optimal if compare_with_010101 else None
            elif graph_type == "cycle":
                print("Computing cycle approximation ratio")
                optimal = return_optimal_cycle(node_count)
                _optimal = optimal
                if optimal is None:
                    print("Missing optimal_results_qaoa_circle.csv; skipping cycle approximation ratio.")
                else:
                    approx_ratio = (results[n] / normalisation_factor) / optimal
                    approx_ratio_010101 = (results_010101[n] / normalisation_factor) / optimal if compare_with_010101 else None
            elif graph_type == "complete":
                print("Computing complete graph approximation ratio")
                optimal = return_optimal_fully_connected(node_count)
                _optimal = optimal
                if optimal is None:
                    print("Missing optimal_results_qaoa_complete.csv; skipping complete approximation ratio.")
                else:
                    approx_ratio = (results[n] / normalisation_factor) / optimal
                    approx_ratio_010101 = (results_010101[n] / normalisation_factor) / optimal if compare_with_010101 else None
            else:
                print("Unknown graph type for approximation ratio calculation; checking misc CSV.")
        except Exception as e:
            print(f"Error during approximation ratio calculation: {e}. Checking misc CSV as fallback.")

        optimal_result = None
        # Fallback: check misc CSV for matching edges and weights
        if approx_ratio is None:
            print(f"Checking optimal_results_misc.csv for matching edges and weights...")
            optimal_from_misc = get_exact_result_from_misc(edges, weights)
            optimal_result = optimal_from_misc
            if optimal_from_misc is not None:
                print(f"Found matching exact result in misc CSV: {optimal_from_misc}")
                approx_ratio = (results[n] / normalisation_factor) / optimal_from_misc
                approx_ratio_010101 = (results_010101[n] / normalisation_factor) / optimal_from_misc if compare_with_010101 else None
                _optimal = optimal_from_misc
            else:
                print("No matching result found in misc CSV; skipping approx ratio computation.")

        row = {
            'run_id': run_id,
            'n': node_count,
            'm': m,
            'p': p,
            'precision/iterations': precision,
            'singlet_injection': singlet_injection,
            'warm_start': warm_start,
            'parameter_vector': str(parameter_settings.get("parameter_vector")),
            'optimal_result': optimal_result,
            'sdp_objective_value_step1': sdp_objective_value,
            'sdp_objective_value_king_normalized_step1': sdp_objective_value_normalized / normalisation_factor if sdp_objective_value_normalized is not None else None,
            'algorithm17_actual_energy': algorithm17_actual_energy / normalisation_factor if lasserre_level == 2 and algorithm17_actual_energy is not None else None,
            'algorithm17_lower_bound_energy': algorithm17_lower_bound_energy / normalisation_factor if lasserre_level == 2 and algorithm17_lower_bound_energy is not None else None,
            'initial_ws_energy_prodStates_step2': initial_energy_prodStates / normalisation_factor if initial_energy_prodStates is not None else None,
            'initial_sdp_statevector_energy': initial_sdp_statevector_energy / normalisation_factor if initial_sdp_statevector_energy is not None else None,
            'initial_sdp_statevec_ratio': (initial_sdp_statevector_energy / normalisation_factor) / _optimal if _optimal is not None and initial_sdp_statevector_energy is not None else None,
            'initial_ws_energy_010101': initial_ws_energy_010101 / normalisation_factor if compare_with_010101 and initial_ws_energy_010101 is not None else None,
            'QAOA_improvement_over_SDP_statevectorEnergy': results[n] - initial_sdp_statevector_energy if initial_sdp_statevector_energy is not None else None,
            'QAOA_improvement_over_SDP_prodStatesEnergy': results[n] - initial_energy_prodStates if initial_energy_prodStates is not None else None,
            'result': results[n] / normalisation_factor,
            'result_010101': results_010101[n] / normalisation_factor if compare_with_010101 else None,
            'approx_ratio': approx_ratio,
            'approx_ratio_010101': approx_ratio_010101,
            'diff. approx. ratio': round(approx_ratio, 6) - round(approx_ratio_010101, 6) if compare_with_010101 and approx_ratio is not None and approx_ratio_010101 is not None else None,
            'sdp ws greater': round(approx_ratio, 6) >= round(approx_ratio_010101, 6) if compare_with_010101 and approx_ratio is not None and approx_ratio_010101 is not None else None,
            'duration_seconds': elapsed_time,
            'full duration_seconds': time.time() - global_starttime,
            'finished_at': finished_at,
            'processor': processor_name,
            'hostname': hostname,
            'total_ram_gb': total_ram_gb,
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'python_version': python_version,
            'peak_ram_mb': peak_ram_mb,
            'benchmark_repeat_idx': benchmark_repeat_idx,
            'sdp_seed': sdp_seed_override,
            'hog_graph_index': n if graph_generation_type == "hog" else None,
            'Instance_is_triangle_free': is_triangle_free(edges),
            'Instance_is_3_regular': is_3_regular(edges)
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
        print(f"CSV writing for n={n} took {time_sections[f'csv_writing_n_{n}']:.2f} seconds; started at stardate {time_section} and finished at stardate {time.time()}")

    # XXX Time
    time_section = time.time()

    print(f"results: {results}")
    print(f"durations: {duration}")
    print(f"Results saved to: {csv_filename}")

    global_endtime = time.time()
    print(f"Global end time: {global_endtime}")

    # XXX Time
    time_sections["finalisation"] = time.time() - time_section
    print(f"Finalisation time: {time_sections['finalisation']:.2f} seconds; started at stardate {time_section} and finished at stardate {time.time()}")
    sum_sections_time = sum(time_sections.values())
    print(f"Sum of all section times: {sum_sections_time:.2f} seconds")
    time_sections["total_time"] = global_endtime - global_starttime
    print(f"run_id: {run_id}")
    time_dir = Path(__file__).resolve().parents[1] / "Results" / "time"
    time_dir.mkdir(parents=True, exist_ok=True)
    with (time_dir / f"{run_id}.csv").open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['section', 'duration_seconds'])
        writer.writeheader()
        for section, duration_sec in time_sections.items():
            writer.writerow({'section': section, 'duration_seconds': round(duration_sec, 2)})
    # Run shell file: sudo nohup zsh qaoa_benchmarks.sh > Results/logs/launcher_$(date +'%Y%m%d_%H%M%S').log 2>&1 &