import random
from Utils import random_instance_generator, read_graphs_as_edge_lists


def instance_generator(type, n: int, weighted: bool = False, random_weights: bool = False):
    type_lower = type.lower()
    if type_lower == "line":
        edges = make_line_graph(n)
    elif type_lower == "cycle":
        edges = make_cycle_graph(n)
    elif type_lower == "complete":
        edges = make_complete_graph(n)
    elif type_lower == "random":
        edges, _, _ = random_instance_generator(n, weights_static=not weighted, sparse=True)
        print(f"Generated random graph with {n} nodes and {len(edges)} edges.")
    elif type_lower == "hog":
        edges = make_hog_graph(n)
    else:
        raise ValueError("Unknown graph type: " + type + "; expected 'line', 'cycle', 'complete', 'random' or 'HOG'.")

    if weighted and random_weights:
        weights = [random.uniform(1e-3, 1.0) for _ in edges]
    else:
        weights = [1.0] * len(edges)

    return edges, weights

def make_line_graph(n: int):
    print(f"Generating line graph with {n} nodes...")
    return [(i, i + 1) for i in range(n - 1)]

def make_cycle_graph(n: int):
    print(f"Generating cycle graph with {n} nodes...")
    if n < 3:
        raise ValueError("cycle graph needs at least 3 nodes")
    return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

def make_complete_graph(n: int):
    print(f"Generating complete graph with {n} nodes...")
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def make_hog_graph(n: int):
    hog_graphs = read_graphs_as_edge_lists()
    if n < 0 or n >= len(hog_graphs):
        raise IndexError(f"HOG graph index {n} is out of range for {len(hog_graphs)} available graphs")
    edges = hog_graphs[n]
    print(f"Loaded HOG graph index {n} with {len(edges)} edges.")
    return edges