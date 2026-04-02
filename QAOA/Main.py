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

import pandas as pan
from numpy.f2py.auxfuncs import throw_error

from Circuit import QAOACircuit
from ParamOptimisation import BayesianOptimiser, optimise_cobyla, grid_search
import sys
from pathlib import Path
from utils import get_benchmark_params
from WarmStart import get_warm_start_state, extract_correlations

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from SPD.Main import main as SDP_main

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


def get_total_ram_gb():
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return None


def get_physical_cores():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count()


def get_logical_cores():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count()


def get_peak_ram_mb():
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 2)
    except Exception:
        return None

def main(m = None, p=20, N_bayes=200, init_initial_state = False, self_init_linegraph = False):
    """
    This method builds and optimises a QAOA instance on a line graph.

    :param m: number of edges for the generated line graph
    :param p: number of QAOA layers
    :param N_bayes: number of optimisation iterations
    :param init_initial_state: whether to inject an SDP-derived warm-start state
    :param self_init_linegraph: whether to use line-graph singlet state preparation
    :return: None
    """
    assert m is not None
    edges = [(i, i + 1) for i in range(m)] # Optionally replace by desired edge list, if a linegraph is not desired
    weights = [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    initial_state, moment_matrix = get_warm_start_state((edges, weights), n) if init_initial_state else (None, None)
    warm_start_correlations = extract_correlations(moment_matrix, edges) if moment_matrix is not None else None

    print(f"warm_start_correlations: {warm_start_correlations}")

    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n} nodes, p: {p} layers, N_bayes: {N_bayes} iterations")
    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights)

    benchmark_params: dict = get_benchmark_params()
    QAOA.params = benchmark_params["parameter_vector"]

    QAOA.initial_state = initial_state
    if warm_start_correlations is not None and benchmark_params["warm_start_correlations"]:
        QAOA.warm_start_correlations = warm_start_correlations
    elif benchmark_params["warm_start_correlations"]:
        raise RuntimeError("Warm start correlations are not available for this benchmark.")

    print(f"Initial state: {initial_state}") if initial_state is not None else print("No initial state provided.")
    QAOA.self_init_linegraph = self_init_linegraph
    QAOA.build_qaoa_maxcut_circuit(add_measurements=False) # TODO check parameter (changed from True to False)

    BO = BayesianOptimiser()

    assert sum([optimiser_bayesian, optimiser_cobyla, gridsearch]) == 1
    minimum_energy = None
    if optimiser_bayesian:
        minimum_energy = BO.bayesian_optimisation(QAOA=QAOA, N_bayes=N_bayes, no_layers=p)
    elif optimiser_cobyla:
        returned_energy = optimise_cobyla(QAOA=QAOA, no_layers=p, max_iter=N_bayes)
        minimum_energy = -returned_energy.fun
    elif gridsearch:
        minimum_energy = grid_search(QAOA, p, precision=precision)

    print(minimum_energy)

    return minimum_energy

def return_optimal_line(m):
    df = pan.read_csv("optimal_results_qaoa.csv")
    val = df.loc[df["m"] == m, "result"].item()
    return val


if __name__ == "__main__":
    parameter_settings = get_benchmark_params()
    singlet_injection = parameter_settings["singlet_injection"]
    warm_start = parameter_settings["warm_start"]
    assert not (singlet_injection and warm_start), "Singlet injection and warm start cannot be used simultaneously, as they both modify the initial state preparation. Please choose one of the two options for a valid benchmark configuration."
    print(f"Benchmark parameters: {parameter_settings}, Running QAOA with equal superposition")
    results = {}
    duration = {}
    precision = float(sys.argv[1])
    p = int(sys.argv[2])
    
    # Keep one shared CSV file and append safely across parallel runs.
    csv_filename = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else "qaoa_results_COBYLA.csv"
    
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

    fieldnames = ['run_id', 'm', 'p', 'precision/iterations', 'singlet_injection', 'warm_start',
                  'parameter_vector', 'result', 'duration_seconds', 'finished_at', 'approx_ratio',
                  'processor', 'hostname', 'total_ram_gb', 'physical_cores', 'logical_cores',
                  'python_version', 'peak_ram_mb']

    for m in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
        start_time = time.time()
        print(f"Running QAOA for m={m} edges...; Max iterations: {int(precision)}")
        results[m] = main(m=m, p=p, N_bayes=int(precision), self_init_linegraph=singlet_injection, init_initial_state=warm_start)
        elapsed_time = time.time() - start_time
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        peak_ram_mb = get_peak_ram_mb()
        print(f"Finished QAOA for m={m} edges at {finished_at} on {processor_name}. Time taken: {elapsed_time:.2f} seconds. Hours: {elapsed_time / 3600:.2f} hours. Peak RAM: {peak_ram_mb} MB.")
        duration[m] = elapsed_time

        row = {
            'run_id': run_id,
            'm': m,
            'p': p,
            'precision/iterations': precision,
            'singlet_injection': singlet_injection,
            'warm_start': warm_start,
            'parameter_vector': str(parameter_settings["parameter_vector"]),
            'result': results[m],
            'duration_seconds': elapsed_time,
            'finished_at': finished_at,
            'approx_ratio': results[m] / return_optimal_line(m),
            'processor': processor_name,
            'hostname': hostname,
            'total_ram_gb': total_ram_gb,
            'physical_cores': physical_cores,
            'logical_cores': logical_cores,
            'python_version': python_version,
            'peak_ram_mb': peak_ram_mb
        }

        # Simple append to CSV file
       # write_header = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
        
        # Append to CSV under an exclusive file lock so parallel processes
        # cannot both decide to write the header at the same time.
        with open(csv_filename, 'a+', newline='') as csvfile:
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
            try:
                csvfile.seek(0, os.SEEK_END)
                write_header = csvfile.tell() == 0
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                csvfile.flush()
                os.fsync(csvfile.fileno())
            finally:
                fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)


    print(f"results: {results}")
    print(f"durations: {duration}")
    print(f"Results saved to: {csv_filename}")