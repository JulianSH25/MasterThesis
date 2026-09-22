"""Classify graph instances used by the QAOA benchmark pipeline.

The instance generators and result-analysis scripts use these small predicates to
identify structural graph families directly from an undirected edge list.
"""

from itertools import combinations

import networkx as nx


#class GraphCharacteristics:
    #"""Check graph characteristics directly from an edge list."""

@staticmethod
def _graph(edges: list[tuple[int, int]]) -> nx.Graph:
    """Build a NetworkX graph from an undirected edge list.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        A NetworkX graph containing every supplied edge.
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph

@staticmethod
def is_bipartite(edges: list[tuple[int, int]]) -> bool:
    """Check whether an edge-list graph is bipartite.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` when the graph admits a two-colouring.
    """
    return nx.is_bipartite(_graph(edges))

@staticmethod
def is_regular(edges: list[tuple[int, int]]) -> bool:
    """Check whether every vertex incident to an edge has the same degree.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` for a non-empty regular graph.
    """
    graph = _graph(edges)
    degrees = [degree for _, degree in graph.degree()]
    return bool(degrees) and len(set(degrees)) == 1

@staticmethod
def regular_degree(edges: list[tuple[int, int]]) -> int | None:
    """Return the common degree, or ``None`` for an irregular graph.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        The shared vertex degree, or ``None`` when no such degree exists.
    """

    graph = _graph(edges)
    degrees = [degree for _, degree in graph.degree()]

    if not degrees or len(set(degrees)) != 1:
        return None
    return degrees[0]

@staticmethod
def is_claw_free(edges: list[tuple[int, int]]) -> bool:
    """Check whether the graph contains no induced claw ``K_{1,3}``.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` if no vertex has three pairwise non-adjacent neighbours.
    """
    graph = _graph(edges)

    for centre in graph:
        neighbours = list(graph.neighbors(centre))
        for a, b, c in combinations(neighbours, 3):
            if (
                not graph.has_edge(a, b)
                and not graph.has_edge(a, c)
                and not graph.has_edge(b, c)
            ):
                return False

    return True

@staticmethod
def is_twin_free(edges: list[tuple[int, int]]) -> bool:
    """Check that the graph has neither true nor false twin vertices.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` when every pair of vertices has a distinct neighbourhood.
    """

    graph = _graph(edges)

    for u, v in combinations(graph.nodes, 2):
        neighbours_u = set(graph.neighbors(u))
        neighbours_v = set(graph.neighbors(v))

        if graph.has_edge(u, v):
            if neighbours_u - {v} == neighbours_v - {u}:
                return False
        elif neighbours_u == neighbours_v:
            return False

    return True

@staticmethod
def is_planar(edges: list[tuple[int, int]]) -> bool:
    """Check whether an edge-list graph is planar.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` if NetworkX finds a planar embedding.
    """
    planar, _ = nx.check_planarity(_graph(edges))
    return planar

@staticmethod
def is_triangle_free(edges: list[tuple[int, int]]) -> bool:
    """Check whether an edge-list graph contains no triangles.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` when the graph contains no 3-cycle.
    """
    graph = _graph(edges)
    return sum(nx.triangles(graph).values()) == 0

@staticmethod
def is_eulerian(edges: list[tuple[int, int]]) -> bool:
    """Check whether a non-empty connected graph has an Eulerian cycle.

    Args:
        edges: Vertex pairs describing the graph.

    Returns:
        ``True`` if the graph is connected and all degrees are even.
    """
    graph = _graph(edges)

    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return False

    return all(degree % 2 == 0 for _, degree in graph.degree())
