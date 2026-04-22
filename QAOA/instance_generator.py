import random
from tokenize import String
from utils import random_instance_generator


def instance_generator(type, n: int, weighted: bool = False):
    if type == "line":
        edges = make_line_graph(n)
    elif type == "cycle":
        edges = make_cycle_graph(n)
    elif type == "complete":
        edges = make_complete_graph(n)
    elif type == "random":
        edges, _, _ = random_instance_generator(n, weights_static=not weighted, sparse=True)
        print(f"Generated random graph with {n} nodes and {len(edges)} edges.")
    else:
        raise ValueError("Unknown graph type: " + type + "; expected 'line', 'cycle', or 'complete'.")
    weights = [random.uniform(1e-3, 1.0) for _ in edges] if weighted else [1.0] * len(edges)

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