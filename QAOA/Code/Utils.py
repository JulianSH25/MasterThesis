import json
import ast
import hashlib
from pathlib import Path
import numpy as np
import platform
import subprocess
import os
import resource
import random
from collections import Counter
import csv

EXACT_RESULT_FIELDNAMES = ["energy", "n", "m", "edges", "weights", "graph_hash", "graph_type"]

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
    This function loads benchmark parameters from the JSON config snapshot
    passed by the benchmark shell script via BENCHMARK_CONFIG_FILE.

    :return: benchmark parameter dictionary exactly as stored in JSON,
             except that sdp_seed may be overridden by SDP_SEED_OVERRIDE
    """
    config_path_env = os.environ.get("BENCHMARK_CONFIG_FILE")

    if not config_path_env:
        raise RuntimeError(
            "BENCHMARK_CONFIG_FILE is not set. "
            "Run Main.py through qaoa_benchmarks.sh with an explicit config file."
        )

    config_path = Path(config_path_env).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as config_file:
        params = json.load(config_file)

    if "SDP_SEED_OVERRIDE" in os.environ:
        params["sdp_seed"] = int(os.environ["SDP_SEED_OVERRIDE"])

    return params

def classify_graph(edges):
    verts = {u for e in edges for u in e}
    n = len(verts)
    m = len(edges)

    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    d = list(deg.values())

    if m == n * (n - 1) // 2 and n != 2:
        return "complete"
    if n >= 3 and all(x == 2 for x in d):
        return "cycle"
    if d.count(1) == 2 and d.count(2) == n - 2:
        return "line"
    return None

def normalise_edge_weights(edges: list, weights: list | str | None = None) -> list[float]:
    if weights is None:
        return [1.0] * len(edges)
    if isinstance(weights, str):
        if weights.strip() == "":
            return [1.0] * len(edges)
        weights = ast.literal_eval(weights)
    if len(weights) == 0:
        return [1.0] * len(edges)
    if len(edges) != len(weights):
        raise ValueError(
            f"Edge/weight length mismatch: {len(edges)} edges, {len(weights)} weights"
        )
    return [float(weight) for weight in weights]

def canonical_edge_weight_items(edges: list, weights: list | str | None = None) -> list[tuple[tuple[int, int], float]]:
    """
    Canonical representation of an undirected weighted graph.
    Makes matching independent of edge order and endpoint orientation.
    Example: (0, 1) and (1, 0) become identical.
    """

    weights = normalise_edge_weights(edges, weights)

    return sorted(
        (tuple(sorted((int(u), int(v)))), float(w))
        for (u, v), w in zip(edges, weights)
    )

def canonical_edge_weight_strings(edges: list, weights: list | str | None = None) -> tuple[str, str]:
    canonical_items = canonical_edge_weight_items(edges, weights)
    canonical_edges = [edge for edge, _ in canonical_items]
    canonical_weights = [weight for _, weight in canonical_items]
    return str(canonical_edges), str(canonical_weights)

def graph_instance_hash(edges: list, weights: list | str | None = None) -> str:
    """Stable hash for a canonical undirected weighted graph instance."""
    payload = [
        {"edge": [u, v], "weight": weight}
        for (u, v), weight in canonical_edge_weight_items(edges, weights)
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _row_with_graph_hash(row: dict) -> dict:
    parsed_edges = ast.literal_eval(row["edges"])
    parsed_weights = normalise_edge_weights(parsed_edges, row.get("weights"))
    edges_str, weights_str = canonical_edge_weight_strings(parsed_edges, parsed_weights)
    return {
        "energy": row.get("energy"),
        "n": row.get("n"),
        "m": row.get("m"),
        "edges": edges_str,
        "weights": weights_str,
        "graph_hash": row.get("graph_hash") or graph_instance_hash(parsed_edges, parsed_weights),
        "graph_type": row.get("graph_type"),
    }

def ensure_exact_results_misc_has_hashes(csv_path: Path) -> None:
    """Backfill graph_hash for existing exact-result CSVs before appending new rows."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames and "graph_hash" in reader.fieldnames:
            return
        rows = [_row_with_graph_hash(row) for row in reader]

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EXACT_RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

def log_exact_result(energy: float, n: int, m: int, edges: list, weights: list, graph_type: str) -> None:
    """Log exact solver result to optimal_results_misc.csv unless already present."""
    csv_path = Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_exact_results_misc_has_hashes(csv_path)

    existing_energy = get_exact_result_from_misc(edges, weights, n, m, raise_if_missing=False)
    if existing_energy is not None:
        print(f"Exact result already logged for this instance; skipping duplicate. Existing energy: {existing_energy}")
        return

    edges_str, weights_str = canonical_edge_weight_strings(edges, weights)
    graph_hash = graph_instance_hash(edges, weights)

    row = {
        "energy": energy,
        "n": n,
        "m": m,
        "edges": edges_str,
        "weights": weights_str,
        "graph_hash": graph_hash,
        "graph_type": graph_type,
    }

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EXACT_RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def get_exact_result_from_misc(
    edges: list,
    weights: list | None,
    n: int,
    m: int,
    raise_if_missing: bool = True
) -> float | None:
    """Look up exact result from optimal_results_misc.csv by n, m, and graph hash."""
    csv_path = Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"

    if not csv_path.exists():
        if raise_if_missing:
            raise FileNotFoundError(f"optimal_results_misc.csv not found at {csv_path}")
        return None

    edges_str, weights_str = canonical_edge_weight_strings(edges, weights)
    target_hash = graph_instance_hash(edges, weights)

    try:
        with csv_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    if int(row.get("n", "")) != int(n) or int(row.get("m", "")) != int(m):
                        continue
                except (TypeError, ValueError):
                    continue

                row_edges = row.get("edges")
                row_weights = row.get("weights")
                row_hash = row.get("graph_hash")

                if row_hash and row_hash == target_hash:
                    return float(row["energy"])

                # Preferred new canonical format.
                if row_edges == edges_str and row_weights == weights_str:
                    return float(row["energy"])

                # Backward compatibility for old non-canonical rows.
                try:
                    old_edges = ast.literal_eval(row_edges)
                    old_weights = normalise_edge_weights(old_edges, row_weights)
                    old_edges_str, old_weights_str = canonical_edge_weight_strings(old_edges, old_weights)

                    if old_edges_str == edges_str and old_weights_str == weights_str:
                        return float(row["energy"])
                except Exception:
                    pass

    except Exception as e:
        print(f"Error reading optimal_results_misc.csv: {e}")

    return None

def deduplicate_exact_results_misc() -> None:
    """
    Remove duplicate graph entries from optimal_results_misc.csv in place.

    Keeps the first occurrence of each canonical (edges, weights) pair.
    Rewrites legacy edge/weight rows into canonical form.
    """
    csv_path = Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"

    if not csv_path.exists():
        print(f"No exact-results file found at {csv_path}; nothing to deduplicate.")
        return

    unique_rows = []
    seen_keys: set[str] = set()
    removed_count = 0

    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                normalised_row = _row_with_graph_hash(row)
                edges_str = normalised_row["edges"]
                weights_str = normalised_row["weights"]
                graph_hash = normalised_row["graph_hash"]
            except Exception:
                normalised_row = {
                    "energy": row.get("energy"),
                    "n": row.get("n"),
                    "m": row.get("m"),
                    "edges": row.get("edges", ""),
                    "weights": row.get("weights", ""),
                    "graph_hash": row.get("graph_hash", ""),
                    "graph_type": row.get("graph_type"),
                }
                edges_str = normalised_row["edges"]
                weights_str = normalised_row["weights"]
                graph_hash = normalised_row["graph_hash"]

            key = graph_hash or f"{edges_str}|{weights_str}"

            if key in seen_keys:
                removed_count += 1
                continue

            seen_keys.add(key)
            unique_rows.append(normalised_row)

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EXACT_RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"Deduplicated {csv_path}: kept {len(unique_rows)} rows, removed {removed_count} duplicates.")

def read_graphs_as_edge_lists(path: str | None = None) -> list[list[tuple[int, int]]]:
    # Used to decompose "House of Graphs" adjacency lists and format into edge lists
    graphs = []
    if path is None:
        path = get_benchmark_params()["relative_graph_adjList_path"]

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


"""The below graph type verification methods are intended for logging purposes only, to assure that the benchmarking instance was indeed of a desired type)"""
def is_triangle_free(edge_list: list[tuple[int, int]]) -> bool:
    """
    Return True iff the undirected graph described by edge_list is triangle-free.

    Self-loops are treated as invalid and therefore return False.
    Parallel edges are ignored for the triangle check, since they do not create
    triangles in a simple-graph sense.
    """
    adjacency: dict[int, set[int]] = {}

    for u, v in edge_list:
        if u == v:
            return False
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    for u in adjacency:
        neighbours = list(adjacency[u])
        for i, v in enumerate(neighbours):
            for w in neighbours[i + 1:]:
                if w in adjacency[v]:
                    return False

    return True


def is_3_regular(edge_list: list[tuple[int, int]]) -> bool:
    """
    Return True iff the undirected graph described by edge_list is 3-regular.

    The graph is interpreted as a simple undirected graph. Self-loops and
    parallel edges are treated as invalid and therefore return False.
    """
    adjacency: dict[int, set[int]] = {}
    seen_edges: set[tuple[int, int]] = set()

    for u, v in edge_list:
        if u == v:
            return False

        edge = tuple(sorted((u, v)))
        if edge in seen_edges:
            return False
        seen_edges.add(edge)

        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    if not adjacency:
        return False

    return all(len(neighbours) == 3 for neighbours in adjacency.values())
