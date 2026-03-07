from SDP_solver import ABCParams, SDP_Solver_
from Rounding import round_sdp_with_cholesky
from Utilities import random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv
#from testing import visualize_cut
import datetime, random, secrets, uuid
from datetime import datetime

benchmark_roundings = True
fixed_seed = False

def main_benchmark(n_vertices, params: ABCParams, instance, sparse: bool, benchm_filename = None, uuid__ = None):
    solver_sdp = SDP_Solver_()

    edges, weights = instance

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params)

    #cut = round_sdp_with_cholesky(M_optimancbsncbletel, parameters=params)
    #print("Rounded cut:")
    #print(cut)

    print("Optimal moment matrix:")
    print(M_optimal)

    edge_count, edges_in_cut, cuts = None, None, None

    cut_variations = set()
    if benchmark_roundings:
        if not benchm_filename:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            hash2 = f"{random.randrange(100):02d}"
            name = f"roundings_1Rounds_{now}_{hash2}"
        else:
            name = benchm_filename
        name += "fixed_seed" if fixed_seed else "random_seed"

        edge_count_3 = False

        seed = secrets.randbits(64) if fixed_seed else None

        best_edge_count, best_edges_in_cut = 0, []

        for k in range(1):
            for _ in range(1):
                cuts = round_sdp_with_cholesky(M_optimal, parameters=params, seed=seed)

                edge_count__, edges_in_cut__ = get_edges_in_cut(cuts, edges)
                if edge_count__ > best_edge_count:
                    best_edge_count, best_edges_in_cut = edge_count__, edges_in_cut__

            edge_count, edges_in_cut = best_edge_count, best_edges_in_cut

            cut_variations.add(edge_count)

            sol_sdp_round = {
                "uuid": uuid__,
                "method": "SDP+GW",
                "params": params,
                "n_vertices": n_vertices,
                "n_edges": len(edges),
                "round_id": k,
                "cut_edges": edge_count,
                "cut_assignment": cuts,
                "edges_in_cut": edges_in_cut,
                "seed": seed,
            }

            save_benchmark_csv(sol_sdp_round, {}, name_addition=name)

        return edge_count, edges_in_cut, cuts, M_optimal, cut_variations
    else:
        print("Rounding...")
        cuts = round_sdp_with_cholesky(M_optimal, parameters=params)
        print(cuts)
        edge_count, edges_in_cut = get_edges_in_cut(cuts, edges)
        print(f"{edge_count} in cut out of a total of {len(edges)} edges")
        return edge_count, edges_in_cut, cuts, M_optimal, {}

    #visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")


if __name__ == "__main__":
    n_vertices = 14
    params: ABCParams = {"a": 1, "b": 1, "c": 1}
    sparse = False

    #edges, weights, nodes = random_instance_generator(n_vertices, weights_static=True, sparse=sparse)
    #edges = [(0, 2), (0, 3), (1, 3), (0, 1), (2, 3)]
    #edges = [(1, 3), (1, 2), (0, 2), (0, 1), (0, 3), (2, 3)]
    edges = [(6, 11), (6, 12), (2, 6), (5, 11), (0, 2), (0, 8), (3, 11), (6, 10), (6, 7), (7, 13), (3, 4), (3, 9), (1, 9), (1, 12), (2, 3), (2, 4), (2, 5), (12, 13),]
    weights = [1.0] * len(edges)

    print(f"Edges: {edges}")

    objectives = set()

    for _ in range(1):
        edge_count, _, _, _ = main_benchmark(n_vertices, params, (edges, weights), sparse)
        objectives.add(edge_count)

    print(f"Objectives found: {objectives}")
