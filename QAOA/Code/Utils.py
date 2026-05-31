import json
from pathlib import Path
import numpy as np
import platform
import subprocess
import os
import resource
import random
from collections import Counter
import csv

def set_random_params(p: int, range: tuple[float, float], seed: int | None = None, init_close_to_zero: bool = False):
    """
    This function samples random QAOA angle parameters for a given circuit depth p.

    :param p: number of QAOA layers
    :param range: the range (min, max) for the random sampling
    :param seed: optional random seed for reproducible sampling
    :param init_close_to_zero: if True, initialise parameters close to zero
    :return: tuple (gamma_values, beta_values) as numpy arrays
    """
    rng = np.random.default_rng(seed)
    gamma_values = rng.uniform(range[0], range[1], size=p) if not init_close_to_zero else rng.uniform(1e-10, 0.1, size=p)
    #beta_values = rng.uniform(1e-10, 0.1 if init_close_to_zero else np.pi, size=p)

    return gamma_values#, beta_values

def random_instance_generator(nodes: int, weights_static: bool, sparse: bool):
    """
    This function generates a random connected undirected graph instance.

    Guarantees connectivity via a random spanning tree, then adds additional
    edges stochastically based on sparsity. Sparse graphs use denser edge thresholds.

    :param nodes: number of vertices
    :param weights_static: if True, all edges have weight 1.0; otherwise random [1e-10, 1.0]
    :param sparse: if True, use high edge threshold (0.8-0.99); otherwise random threshold
    :return: tuple (edges, weights, nodes) describing the graph
    """
    if nodes <= 0:
        return [], [], nodes
    if nodes == 1:
        return [], [], nodes

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    edge_set: set[tuple[int, int]] = set()  # store as (min(u,v), max(u,v))

    # Keep the original "threshold" semantics:
    # add a candidate edge with probability ~ (1 - threshold)
    threshold = random.random() if not sparse else random.uniform(0.8, 0.99)

    def add_edge(u: int, v: int):
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            return
        if (a, b) in edge_set:
            return
        edge_set.add((a, b))
        edges.append((a, b))
        weights.append(random.uniform(1e-10, 1.0) if not weights_static else 1.0)

    # 1) Ensure connectivity via a random spanning tree
    perm = list(range(nodes))
    random.shuffle(perm)
    for idx in range(1, nodes):
        u = perm[idx]
        v = random.choice(perm[:idx])  # connect to any previous node
        add_edge(u, v)

    # 2) Add extra random edges
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if (i, j) in edge_set:
                continue
            if random.random() > threshold:
                add_edge(i, j)

    if classify_graph(edges) in {"line", "complete"} and not nodes <= 3:
        edges, weights, nodes = random_instance_generator(nodes, weights_static, sparse)  

    return edges, weights, nodes

def sample_initial_qaoa_params(n: int, p: int) -> list[tuple[list[float], list[float]]]:
    """
    This function samples initial QAOA parameter tuples for Bayesian optimisation. I realise it is basically a duplicate of the above function I already used before in a different place.

    :param n: number of parameter points to sample
    :param p: number of QAOA layers per parameter vector
    :return: list of tuples (gamma_values, beta_values)
    """
    return [
        (
            [random.uniform(0, 3.141592653589793) for _ in range(p)],
            [random.uniform(0, 3.141592653589793 / 2) for _ in range(p)],
        )
        for _ in range(n)
    ]

def build_qaoa_warm_start_state(states: list[np.ndarray]):
    """
    This function builds a global warm-start statevector from local density matrices.

    :param states: list of local density matrices as returned from the SDP solver, one for each edge in the graph; each state is a 2x2 numpy array, representing the respective vertex
    :return: normalised global warm-start statevector
    """
    statevec = np.array([1.0 + 0.0j])
    if debug := get_benchmark_params().get("debug", False): #XXX
        print("Building warm-start statevector from local density matrices:")
        print(f"Number of local states (edges): {len(states)}")
        print(f"Dimensionality of each local state: {states[0].shape if states else 'N/A'}")

    for state in states:
        if debug: #XXX
            print(f"Dimensionality of local state: {state.shape}")
        ew, ev = np.linalg.eigh(state)
        idx = np.argmax(ew) # maximum eigenvalue index
        max_ew = float(ew[idx]) # maximum eigenvalue, used to find out how pure the state is
        x = ev[:, idx] / np.linalg.norm(ev[:, idx])

        if max_ew < 1 - 1e-10: raise ValueError("State is not pure enough for unique ket extraction; maximum Eigenvalue is " + str(max_ew))
        statevec = np.kron(statevec, x)
    
    if debug: #XXX
        print(f"Unnormalised warm-start statevector: {statevec}")
        print(f"Norm of unnormalised statevector: {np.linalg.norm(statevec)}")
        print("Finished building warm-start statevector.")

    return_vec = statevec / np.linalg.norm(statevec)
    if debug: print(f"Size of normalised warm-start statevector: {return_vec.shape}, norm: {np.linalg.norm(return_vec)}") #XXX
    return return_vec

def get_benchmark_params() -> dict:
    """
    This function loads benchmark parameters from JSON.

    :return: benchmark parameter dictionary exactly as stored in JSON
    """
    config_path = Path(__file__).resolve().parent / "benchmark_config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)

def classify_graph(edges):
    verts = {u for e in edges for u in e}
    n = len(verts)
    m = len(edges)

    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    d = list(deg.values())

    if m == n * (n - 1) // 2:
        return "complete"
    if n >= 3 and all(x == 2 for x in d):
        return "cycle"
    if d.count(1) == 2 and d.count(2) == n - 2:
        return "line"
    return None

def log_exact_result(energy: float, n: int, m: int, edges: list, weights: list, graph_type: str) -> None:
    """Log exact solver result to optimal_results_misc.csv with metadata."""
    csv_path = Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"
    fieldnames = ["energy", "n", "m", "edges", "weights", "graph_type"]
    row = {
        "energy": energy,
        "n": n,
        "m": m,
        "edges": str(edges),
        "weights": str(weights),
        "graph_type": graph_type,
    }
    
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def get_exact_result_from_misc(edges: list, weights: list) -> float | None:
    """Look up exact result from optimal_results_misc.csv by matching edges and weights.
    
    :param edges: edge list to match
    :param weights: weight list to match
    :return: exact energy if found, None otherwise
    """
    csv_path = Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"optimal_results_misc.csv not found at {csv_path}")
    
    edges_str = str(edges)
    weights_str = str(weights)
    
    try:
        with csv_path.open("r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("edges") == edges_str and row.get("weights") == weights_str:
                    return float(row["energy"])
    except Exception as e:
        print(f"Error reading optimal_results_misc.csv: {e}")
    
    return None

def read_graphs_as_edge_lists(path: str = get_benchmark_params()["relative_graph_adjList_path"]) -> list[list[tuple[int, int]]]:
    # Used to decompose "House of Graphs" adjacency lists and format into edge lists
    graphs = []

    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parent / path_obj

    with path_obj.open("r") as f:
        content = f.read().strip()
    
    # split graphs by blank lines
    raw_graphs = content.split("\n\n")
    
    for raw_graph in raw_graphs:
        edges = set()
        vertices = set()
        
        for line in raw_graph.splitlines():
            node, neighbors = line.split(":")
            u = int(node.strip())
            vertices.add(u)
            
            for v_str in neighbors.strip().split():
                v = int(v_str)
                vertices.add(v)
                
                # avoid duplicate edges (u,v) and (v,u)
                edge = tuple(sorted((u, v)))
                edges.add(edge)

        # Relabel vertices to zero-based indices so downstream SDP/QAOA code
        # can safely use idx(i, k) with i in range(n_vertices).
        relabel = {vertex: index for index, vertex in enumerate(sorted(vertices))}
        normalized_edges = [tuple(sorted((relabel[u], relabel[v]))) for u, v in edges]
        
        graphs.append(normalized_edges)
    
    return graphs

"""Methods to retrieve benchmarking metrics"""
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

