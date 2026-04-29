from .SDP_solver import ABCParams, SDP_Solver_
import numpy as np
from .Rounding import round_sdp_with_cholesky
from .Utilities import random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv, idx
#from testing import visualize_cut
import datetime, random, secrets, uuid
from datetime import datetime

benchmark_roundings = True
fixed_seed = False

def main_benchmark(n_vertices, params: ABCParams, instance, sparse: bool, benchm_filename = None, uuid__ = None):
    """
    This function solves and rounds one SDP instance, optionally persisting benchmark rows.

    :param n_vertices: number of vertices in the graph instance
    :param params: SDP Hamiltonian coefficients as a, b, c bits
    :param instance: tuple (edges, weights) describing the graph
    :param sparse: whether the benchmark configuration uses sparse graph generation
    :param benchm_filename: optional benchmark filename prefix override
    :param uuid__: optional run identifier for benchmark row tracking
    :return: tuple (edge_count, edges_in_cut, cuts, M_optimal, cut_variations)
    """
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

def main(instance, n_vertices, params: dict, debug: bool = False):
    """
    NOTE: Use this method to run a single SDP solve without benchmarking.
    This function runs the SDP solve and rounding flow without benchmark persistence.

    :param instance: tuple (edges, weights) describing the graph
    :param n_vertices: number of vertices in the graph instance
    :param params: SDP Hamiltonian coefficients as a, b, c bits
    :return: tuple (edge_count, edges_in_cut, cuts, M_optimal, states)
    """
    # Same functionality as main_benchmark, but without benchmarking. TODO streamline both functions.
    solver_sdp = SDP_Solver_()

    edges, weights = instance

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params, debug=debug)

    print("Rounding...")
    cuts, states = round_sdp_with_cholesky(M_optimal, parameters=params)
    print(cuts)
    energy = solver_sdp.compute_energy(states, edges=edges, weights=weights, params=params)
    print(f"Energy of rounded cut: {energy}")
    edge_count, edges_in_cut = get_edges_in_cut(cuts, edges)
    print(f"{edge_count} in cut out of a total of {len(edges)} edges")
    # BUG for quantum maxcut neither an edge count nor cut should be returned; This is currently not critical but should be fixed, if just for clean code 
    #return edge_count, edges_in_cut, cuts, M_optimal, states
    return energy, M_optimal, states, cuts

if __name__ == "__main__":
    # NOTE: Run this to run the main function that executes one SDP solve and rounding without benchmarking. To run the benchmark pipeline, run the `run_single_benchmark` function in `benchmark_pipeline.py` instead. (i.e. change function call below)
    n_vertices = 14
    params = {"a": 1, "b": 1, "c": 1}
    sparse = False

    #edges, weights, nodes = random_instance_generator(n_vertices, weights_static=True, sparse=sparse)
    edges = [(0, 2), (0, 3), (1, 3), (0, 1), (2, 3)]
    #edges = [(1, 3), (1, 2), (0, 2), (0, 1), (0, 3), (2, 3)]
    # NOTE: Define your instance here
    #edges = [(6, 11), (6, 12), (2, 6), (5, 11), (0, 2), (0, 8), (3, 11), (6, 10), (6, 7), (7, 13), (3, 4), (3, 9), (1, 9), (1, 12), (2, 3), (2, 4), (2, 5), (12, 13),]
    weights = [1.0] * len(edges)

    print(f"Edges: {edges}")

    objectives = set()

    for _ in range(1):
        edge_count, edges_in_cut, cuts, M, states = main(instance=(edges, weights), n_vertices=n_vertices, params=params) # NOTE: change this function call to `main_benchmark` to run the benchmark pipeline instead of the single-run flow
        #edge_count, _, _, _ = main_benchmark(n_vertices, params, (edges, weights), sparse)
        objectives.add(edge_count)
        print(f"State: {states}")

        for i, j in edges:
            corr = (float(np.real(M[idx(i, 0), idx(j, 0)]))
                    + float(np.real(M[idx(i, 1), idx(j, 1)]))
                    + float(np.real(M[idx(i, 2), idx(j, 2)]))
                    )

            print(f"idx 0: {M[idx(i, 0), idx(j, 0)]}, idx({i}, 0): {idx(i, 0)}, idx({j}, 0): {idx(j, 0)},")
            print(f"idx 1: {M[idx(i, 1), idx(j, 1)]}, idx(i, 1): {idx(i, 1)}, idx(j, 1): {idx(j, 1)},")
            print(f"idx 2: {M[idx(i, 2), idx(j, 2)]}, idx(i, 2): {idx(i, 2)}, idx(j, 2): {idx(j, 2)},")

            print(f"idx 01: {M[idx(i, 0), idx(j, 1)]}")
            print(f"idx 10: {M[idx(i, 1), idx(j, 0)]}")
            print(f"idx 02: {M[idx(i, 0), idx(j, 2)]}")
            print(f"idx 20: {M[idx(i, 2), idx(j, 0)]}")
            print(f"idx 12: {M[idx(i, 1), idx(j, 2)]}")
            print(f"idx 21: {M[idx(i, 2), idx(j, 1)]}")

            print(f"Correlation: {corr}")
            print(f"Trace: {np.trace(M)}")

    print(f"Objectives found: {objectives}")
