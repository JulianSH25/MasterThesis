"""Provide configuration, indexing, graph, and persistence helpers for SDP runs.

The SDP solver and standalone benchmark pipeline share this module to preserve
the Pauli-block index convention and obtain consistent benchmark configuration
and graph representations.
"""

import random
from datetime import datetime
import csv
import os
from pathlib import Path
import json
import numpy as np

if __package__ in (None, ""):
    from Lasserre_level_2 import P, PP
else:
    from .Lasserre_level_2 import P, PP

def get_benchmark_params() -> dict:
    """
    This function loads benchmark parameters from the JSON config snapshot
    passed by the benchmark shell script via BENCHMARK_CONFIG_FILE.

    Returns:
        Benchmark parameters with an optional runtime SDP seed override.

    Raises:
        RuntimeError: If the launcher did not define ``BENCHMARK_CONFIG_FILE``.
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

def idx(i: int, k: int) -> int:
    """
    Map a vertex and Pauli direction to a Level-1 moment-matrix index.

    Args:
        i: Zero-based vertex index.
        k: Pauli index ``0 -> X``, ``1 -> Y``, or ``2 -> Z``.

    Returns:
        Linear index ``3 * i + k`` in the ``3n x 3n`` ordering.
    """
    # k: 0->X, 1->Y, 2->Z
    return 3 * i + k
    #return i * k

def extract_level1_submatrix_from_level2(M_level2, pidx, n_vertices):
    """
    Extract the Level-1 Pauli block used by GP/GW rounding.

    Args:
        M_level2: Full Level-2 moment matrix.
        pidx: Mapping from Pauli strings to Level-2 matrix indices.
        n_vertices: Number of graph vertices.

    Returns:
        ``3n x 3n`` matrix ordered as ``X_0, Y_0, Z_0, X_1, ...``.
    """
    level1_indices = []

    for i in range(n_vertices):
        for k in range(3):
            level1_indices.append(pidx[P(i, k)])

    M_level1 = M_level2[np.ix_(level1_indices, level1_indices)]

    return M_level1

def random_instance_generator(nodes: int, weights_static: bool, sparse: bool):
    """
    This function generates a random connected undirected graph instance.

    Guarantees connectivity via a random spanning tree, then adds additional
    edges stochastically based on sparsity. Sparse graphs use denser edge thresholds.

    Args:
        nodes: Number of vertices.
        weights_static: Use weight one for every edge when true.
        sparse: Use the sparse graph-generation regime when true.

    Returns:
        ``(edges, weights, nodes)`` for a connected random graph.
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
        """Add an unseen non-self-loop edge and its associated weight.

        Args:
            u: First candidate endpoint.
            v: Second candidate endpoint.
        """
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            return
        if (a, b) in edge_set:
            return
        edge_set.add((a, b))
        edges.append((a, b))
        weights.append(random.uniform(1e-10, 1.0) if not weights_static else 1)

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

    return edges, weights, nodes

def line_instance_generator(nodes: int, weights_static: bool, _ = None):
    """
    Create one weighted or unweighted path graph.

    Args:
        nodes: Number of vertices.
        weights_static: Use weight one for every edge when true.
        _: Unused compatibility placeholder.

    Returns:
        ``(edges, weights, nodes)`` for the path graph.
    """
    edges = []
    weights = []
    for i in range(nodes):
        if i + 1 < nodes:
            edges.append((i, i + 1))
            weights.append(random.uniform(1e-10, 1.0) if not weights_static else 1)

    return edges, weights, nodes

def get_edges_in_cut(cut, edges):
    """
    Identify the edges crossing a binary cut.

    Args:
        cut: Vertex assignments indexed by vertex number.
        edges: Undirected graph edges.

    Returns:
        Number and list of crossing edges.
    """
    edge_count = 0
    edges_in_cut = []
    print(f'Check the cut: {cut}') if get_benchmark_params().get("debug", False) else None
    for i in range(len(cut)):
        for j in range(len(cut)):
            if i != j and cut[i] != cut[j] and (i, j) in edges:
                edge_count += 1
                edges_in_cut.append((i, j))

    return edge_count, edges_in_cut

def save_benchmark_csv(sol_sdp, sol_grb, name_addition = ""):
    """
    Append combined SDP and Gurobi fields to a dated benchmark CSV.

    Args:
        sol_sdp: SDP benchmark fields.
        sol_grb: Exact-solver benchmark fields.
        name_addition: Optional filename suffix.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    date = datetime.now().strftime("%Y-%m-%d")
    #filename = f'Benchmarks/benchmark_{sol_sdp["n_vertices"]}_{sol_sdp["n_edges"]}_{now}.csv'
    filename = f'benchmarks_{date}_{name_addition}.csv'
    merged = {**sol_sdp, **sol_grb, "current time": now}  # dict2 overwrites dict1 if keys overlap

    # If the CSV does not exist yet, write headers
    try:
        with open(filename, "x", newline="") as f:  # 'x' fails if file exists
            writer = csv.DictWriter(f, fieldnames=merged.keys())
            writer.writeheader()
            writer.writerow(merged)
    except FileExistsError:
        with open(filename, "a", newline="") as f:  # append
            writer = csv.DictWriter(f, fieldnames=merged.keys())
            writer.writerow(merged)
