import random
from datetime import datetime
import csv
import json


def idx(i: int, k: int) -> int:
    """
    This function maps a vertex index and Pauli operator to a linear index.

    Used to ensure consistent indexing in the 3n x 3n moment matrix.
    Mapping: k=0->X, k=1->Y, k=2->Z.

    :param i: vertex index
    :param k: Pauli operator index (0, 1, or 2)
    :return: linear index 3*i + k for position in flattened structure
    """
    # k: 0->X, 1->Y, 2->Z
    return 3 * i + k
    #return i * k

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
    This function generates a path graph (line graph) instance.

    Creates edges (i, i+1) for each consecutive pair of nodes.

    :param nodes: number of vertices
    :param weights_static: if True, all edges have weight 1.0; otherwise random [1e-10, 1.0]
    :param _: placeholder argument (unused)
    :return: tuple (edges, weights, nodes) describing the line graph
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
    This function identifies edges whose endpoints are in different cut parts.

    Given a vertex cut assignment and an edge list, counts edges that cross
    the cut (one endpoint in each part).

    :param cut: vertex assignment mapping, indexed by vertex number
    :param edges: list of edges as tuples (i, j)
    :return: tuple (edge_count, edges_in_cut) with the count and list of crossing edges
    """
    edge_count = 0
    edges_in_cut = []
    print(f'Check the cut: {cut}')
    for i in range(len(cut)):
        for j in range(len(cut)):
            if i != j and cut[i] != cut[j] and (i, j) in edges:
                edge_count += 1
                edges_in_cut.append((i, j))

    return edge_count, edges_in_cut

def save_benchmark_csv(sol_sdp, sol_grb, name_addition = ""):
    """
    This function saves benchmark results to a CSV file.

    Creates a new file with headers if it does not exist, otherwise appends
    a row. Filenames are timestamped by date.

    :param sol_sdp: dictionary of SDP solution fields
    :param sol_grb: dictionary of Gurobi solution fields
    :param name_addition: optional suffix for the filename
    :return: None
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