from MasterThesis.SDP_solver import ABCParams, SDP_Solver_
from MasterThesis.Rounding import round_sdp_with_cholesky
from Utilities import random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv
from testing import visualize_cut


def main_benchmark(n_vertices, params: ABCParams, sparse: bool):
    solver_sdp = SDP_Solver_()

    edges, weights, nodes = line_instance_generator(nodes=n_vertices, weights_static=True)#, sparse=sparse)

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

    visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")


if __name__ == "__main__":
    n_vertices = 20
    params: ABCParams = {"a": 1, "b": 1, "c": 1}
    sparse = False

    main_benchmark(n_vertices, params, sparse)
