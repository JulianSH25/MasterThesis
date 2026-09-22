"""Create benchmark graph instances for the QAOA entry point.

This module is the graph-source layer of the QAOA pipeline.  It either builds
the simple named graph families used in controlled benchmarks or reads a House
of Graphs instance before the QAOA and SDP stages receive the resulting edges.
"""

import random
from Utils import random_instance_generator, read_graphs_as_edge_lists


def instance_generator(type, n: int, weighted: bool = False, random_weights: bool = False):
    """Create or load one graph instance and align its edge weights.

    Args:
        type: Graph source or family: ``line``, ``cycle``, ``complete``,
            ``random``, or ``hog``.
        n: Number of vertices for generated families, or House of Graphs index
            when ``type`` is ``hog``.
        weighted: Whether the requested benchmark uses weighted edges.
        random_weights: Whether weighted edges should receive random weights.

    Returns:
        A pair ``(edges, weights)`` with weights ordered consistently with
        ``edges``.
    """
    type_lower = type.lower()
    if type_lower == "line":
        edges = make_line_graph(n)
    elif type_lower == "cycle":
        edges = make_cycle_graph(n)
    elif type_lower == "complete":
        edges = make_complete_graph(n)
    elif type_lower == "random":
        edges, _, _ = random_instance_generator(n, weights_static=not weighted, sparse=True) # Creates random but connected graph; Does not create line or complete graphs, only those inbetween (restricted to)
        print(f"Generated random graph with {n} nodes and {len(edges)} edges.")
    elif type_lower == "hog":
        edges = make_hog_graph(n) # Read in an adjacency list; Currently only unweighted graphs (05/05/2026)
    else:
        raise ValueError("Unknown graph type: " + type + "; expected 'line', 'cycle', 'complete', 'random' or 'HOG'.")

    if weighted and random_weights:
        weights = [random.uniform(1e-3, 1.0) for _ in edges]
    else:
        weights = [1.0] * len(edges)

    return edges, weights

def make_line_graph(n: int):
    """Construct the path graph ``P_n``.

    Args:
        n: Number of vertices.

    Returns:
        Consecutive vertex pairs ``(i, i + 1)``.
    """
    print(f"Generating line graph with {n} nodes...")
    return [(i, i + 1) for i in range(n - 1)]

def make_cycle_graph(n: int):
    """Construct the cycle graph ``C_n``.

    Args:
        n: Number of vertices; must be at least three.

    Returns:
        Consecutive vertex pairs plus the closing edge.
    """
    print(f"Generating cycle graph with {n} nodes...")
    if n < 3:
        raise ValueError("cycle graph needs at least 3 nodes")
    return [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

def make_complete_graph(n: int):
    """Construct the complete graph ``K_n``.

    Args:
        n: Number of vertices.

    Returns:
        Every undirected vertex pair exactly once.
    """
    print(f"Generating complete graph with {n} nodes...")
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def make_hog_graph(n: int):
    """Load one House of Graphs edge list by its configured index.

    Args:
        n: Zero-based index in the parsed House of Graphs data set.

    Returns:
        The selected graph's undirected edge list.

    Raises:
        IndexError: If ``n`` does not identify an available graph.
    """
    hog_graphs = read_graphs_as_edge_lists()
    if n < 0 or n >= len(hog_graphs):
        raise IndexError(f"HOG graph index {n} is out of range for {len(hog_graphs)} available graphs")
    edges = hog_graphs[n]
    print(f"Loaded HOG graph index {n} with {len(edges)} edges.")
    return edges
