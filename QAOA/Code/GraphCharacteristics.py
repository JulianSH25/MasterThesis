"""Static checks for graph characteristics."""

from itertools import combinations

import networkx as nx


#class GraphCharacteristics:
    #"""Check graph characteristics directly from an edge list."""

@staticmethod
def _graph(edges: list[tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph

@staticmethod
def is_bipartite(edges: list[tuple[int, int]]) -> bool:
    return nx.is_bipartite(_graph(edges))

@staticmethod
def is_regular(edges: list[tuple[int, int]]) -> bool:
    graph = _graph(edges)
    degrees = [degree for _, degree in graph.degree()]
    return bool(degrees) and len(set(degrees)) == 1

@staticmethod
def regular_degree(edges: list[tuple[int, int]]) -> int | None:
    """Return the common degree, or None if the graph is irregular."""

    graph = _graph(edges)
    degrees = [degree for _, degree in graph.degree()]

    if not degrees or len(set(degrees)) != 1:
        return None
    return degrees[0]

@staticmethod
def is_claw_free(edges: list[tuple[int, int]]) -> bool:
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
    """Check that the graph has neither true nor false twin vertices."""

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
    planar, _ = nx.check_planarity(_graph(edges))
    return planar

@staticmethod
def is_triangle_free(edges: list[tuple[int, int]]) -> bool:
    graph = _graph(edges)
    return sum(nx.triangles(graph).values()) == 0

@staticmethod
def is_eulerian(edges: list[tuple[int, int]]) -> bool:
    graph = _graph(edges)

    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return False

    return all(degree % 2 == 0 for _, degree in graph.degree())
