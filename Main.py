from SDP_solver import ABCParams, SDP_Solver_
from Rounding import round_sdp_with_cholesky
from Utilities import random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv
#from testing import visualize_cut


def main_benchmark(n_vertices, params: ABCParams, instance, sparse: bool):
    solver_sdp = SDP_Solver_()

    edges, weights = instance

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params)

    cut = round_sdp_with_cholesky(M_optimal, parameters=params)
    print("Rounded cut:")
    print(cut)

    print("Optimal moment matrix:")
    print(M_optimal)

    print("Rounding...")
    cuts = [round_sdp_with_cholesky(M_optimal, parameters=params) for _ in range(10)]
    print(cuts)

    edge_count, edges_in_cut = get_edges_in_cut(cut, edges)
    print(f"{edge_count} in cut out of a total of {len(edges)} edges")

    #visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")

    return edge_count, edges_in_cut, cut, M_optimal


if __name__ == "__main__":
    n_vertices = 20
    params: ABCParams = {"a": 0, "b": 0, "c": 1}
    sparse = False

    edges = [
        (6, 11), (6, 12), (2, 6), (5, 11), (0, 2), (0, 8),
        (3, 11), (6, 10), (6, 7), (7, 13), (3, 4), (3, 9),
        (1, 9), (1, 12), (2, 3), (2, 4), (2, 5), (12, 13),
    ]
    weights = [1.0] * len(edges)

    edge_count, _, _, _ = main_benchmark(n_vertices, params, (edges, weights), sparse)
