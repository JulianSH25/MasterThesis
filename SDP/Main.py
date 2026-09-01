if __package__ in (None, ""):
    from Lasserre_level_2 import Level_2_Rounding
    from SDP_solver import ABCParams, SDP_Solver_
else:
    from .Lasserre_level_2 import Level_2_Rounding
    from .SDP_solver import ABCParams, SDP_Solver_
import numpy as np
if __package__ in (None, ""):
    from Rounding import round_sdp_with_cholesky
    from Utilities import extract_level1_submatrix_from_level2, random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv, idx, get_benchmark_params
else:
    from .Rounding import round_sdp_with_cholesky
    from .Utilities import extract_level1_submatrix_from_level2, random_instance_generator, line_instance_generator, get_edges_in_cut, save_benchmark_csv, idx, get_benchmark_params
#from testing import visualize_cut
import datetime, random, secrets, uuid
from datetime import datetime

benchmark_roundings = True
fixed_seed = False
lasserre_level = 2
debug = get_benchmark_params().get("debug", False)

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

def main(
    instance,
    n_vertices,
    lasserre_level,
    params: dict,
    debug: bool = False,
    initial_solver_level_M: int = 2,
    seed: int | None = None,
    algorithm17_num_seeds: int = 1,
    algorithm17_seed_start: int | None = None,
):
    """
    Run a single SDP solve and rounding flow.

    The two level arguments have separate responsibilities:
        - initial_solver_level_M selects the SDP relaxation that is solved.
        - lasserre_level selects the downstream rounding pipeline.

    Supported combinations:
        - L1M1: solve the 3n x 3n level-1 SDP and run GP rounding.
        - L2M1: solve the 3n x 3n level-1 SDP and heuristically run
          Algorithm 17 using its correlations.
        - L2M2: solve the full level-2 SDP, run GP rounding on its extracted
          level-1 submatrix, and run Algorithm 17 using the full matrix.

    This function runs the SDP solve and rounding flow without benchmark persistence.

    :param instance: tuple (edges, weights) describing the graph
    :param n_vertices: number of vertices in the graph instance
    :param lasserre_level: downstream rounding level (1 for GP, 2 for Algorithm 17)
    :param params: SDP Hamiltonian coefficients as a, b, c bits
    :param initial_solver_level_M: SDP relaxation to solve (1 for the 3n level-1
        matrix, 2 for the full level-2 King matrix)
    :param seed: optional seed for seeded GP rounding
    :param algorithm17_num_seeds: number of consecutive Algorithm 17 rotation
        seeds to evaluate; the highest-energy state is retained
    :param algorithm17_seed_start: first Algorithm 17 rotation seed; defaults
        to seed + 1 when seed is provided
    :return: tuple (energy, M_optimal, states, cuts, sdp_result)
    """
    solver_sdp = SDP_Solver_(lasserre_level=lasserre_level)
    edges, weights = instance

    if lasserre_level not in (1, 2):
        raise ValueError(f"Unsupported Lasserre level: {lasserre_level}")

    if initial_solver_level_M not in (1, 2):
        raise ValueError(f"Unsupported initial_solver_level_M: {initial_solver_level_M}")

    if lasserre_level == 1 and initial_solver_level_M != 1:
        raise ValueError(
            "Unsupported combination L1M2: level-1 rounding requires "
            "initial_solver_level_M=1."
        )

    if initial_solver_level_M == 1:
        M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(
            edges,
            weights,
            n_vertices,
            params=params,
            debug=debug,
        )

        M_for_rounding = M_optimal
        basis = None
        pidx = None

    else:
        M_level2_full, basis, pidx = solver_sdp.QMC_SDP_solver_antiFerro_level_2(
            edges,
            weights,
            n_vertices,
            params=params,
            debug=debug,
        )

        M_optimal = M_level2_full
        M_for_rounding = extract_level1_submatrix_from_level2(
            M_level2=M_level2_full,
            pidx=pidx,
            n_vertices=n_vertices,
        )

    gp_rounding_seed = None if seed is None else int(seed)
    if algorithm17_num_seeds < 1:
        raise ValueError(
            "algorithm17_num_seeds must be at least 1, got "
            f"{algorithm17_num_seeds}."
        )

    if algorithm17_seed_start is None and seed is not None:
        algorithm17_seed_start = int(seed) + 1

    print("Rounding...")
    cuts, states, bloch_vectors = round_sdp_with_cholesky(
        M_for_rounding,
        parameters=params,
        seed=gp_rounding_seed,
    )

    print(cuts) if debug else None

    energy = solver_sdp.compute_energy(
        states,
        edges=edges,
        weights=weights,
        params=params,
    )

    print(f"Energy of rounded cut/product state: {energy}")

    sdp_objective_value = getattr(solver_sdp, "last_objective_value", None)

    sdp_result = None
    if lasserre_level == 2:
        rounder = Level_2_Rounding()
        rounder.edges = edges
        rounder.weights = weights
        rounder.n_vertices = n_vertices
        rounder.M_level2 = M_optimal
        rounder.basis = basis
        rounder.pidx = pidx
        rounder.bloch_vectors = bloch_vectors
        rounder.beta_star = 0.390
        if algorithm17_seed_start is None:
            algorithm17_seeds = [None] * algorithm17_num_seeds
        else:
            algorithm17_seeds = [
                int(algorithm17_seed_start) + offset
                for offset in range(algorithm17_num_seeds)
            ]

        best_result = None
        best_seed = None
        candidate_energies = []
        for algorithm17_seed in algorithm17_seeds:
            candidate = rounder.QMC_rounding(
                seed=algorithm17_seed,
                max_vertices=16,
            )
            candidate_energy = float(candidate["actual_energy"])
            candidate_energies.append(
                {
                    "seed": algorithm17_seed,
                    "actual_energy": candidate_energy,
                }
            )
            if (
                best_result is None
                or candidate_energy > float(best_result["actual_energy"])
            ):
                best_result = candidate
                best_seed = algorithm17_seed

        sdp_result = best_result
        sdp_result["algorithm17_num_seeds"] = algorithm17_num_seeds
        sdp_result["algorithm17_seed_start"] = algorithm17_seed_start
        sdp_result["algorithm17_seeds_tried"] = algorithm17_seeds
        sdp_result["algorithm17_candidate_energies"] = candidate_energies
        sdp_result["algorithm17_seed"] = best_seed
        if sdp_result["analytic_F_bound_applicable"]:
            print(f"Algorithm 17 lower-bound energy: {sdp_result['lower_bound_energy']}")
        else:
            print(
                "Algorithm 17 analytical F-bound is not applicable because "
                "the graph is not triangle-free."
            )
        print(f"Algorithm 17 actual entangled-state energy: {sdp_result['actual_energy']}")

    if sdp_objective_value is not None or lasserre_level == 2:
        sdp_result = sdp_result or {}
        sdp_result["rounded_solution_energy"] = float(energy)
        if lasserre_level == 1:
            sdp_result["actual_energy"] = None
            sdp_result["lower_bound_energy"] = None
        if sdp_objective_value is not None:
            sdp_result["sdp_objective_value"] = float(sdp_objective_value)
            sdp_result["sdp_objective_value_normalized"] = float(sdp_objective_value)
        sdp_result["initial_solver_level_M"] = initial_solver_level_M
        sdp_result["lasserre_level"] = lasserre_level
        sdp_result["gp_rounding_seed"] = gp_rounding_seed
        if lasserre_level == 1:
            sdp_result["algorithm17_seed"] = None
        if sdp_objective_value is not None:
            print(f"SDP objective value: {sdp_objective_value}")

    edge_count, edges_in_cut = get_edges_in_cut(cuts, edges)
    print(f"{edge_count} in cut out of a total of {len(edges)} edges")

    if sdp_result is not None:
        if "actual_energy" in sdp_result:
            SDP_Solver_.GP_rounding_energy = sdp_result["actual_energy"]
        elif "rounded_solution_energy" in sdp_result:
            SDP_Solver_.GP_rounding_energy = sdp_result["rounded_solution_energy"]

    return energy, M_optimal, states, cuts, sdp_result


if __name__ == "__main__":
    # Tiny sanity test for the level-2 / Algorithm 17 pipeline.
    # Start with n=2 before trying larger instances.
    n_vertices = 6
    params = {"a": 1, "b": 1, "c": 1}
    lasserre_level_to_run = 2

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
    weights = [1.0] * len(edges)

    print(f"Edges: {edges}")
    print(f"n_vertices: {n_vertices}")
    print(f"Lasserre level: {lasserre_level_to_run}")

    product_energy, M, product_states, cuts, level2_result = main(
        instance=(edges, weights),
        n_vertices=n_vertices,
        params=params,
        lasserre_level=lasserre_level_to_run,
        debug=True,
    )

    print("\n=== Product-state / GP rounding result ===")
    print(f"Cuts: {cuts}") if debug else None
    print(f"Product-state energy: {product_energy}")
    print(f"Number of local product states: {len(product_states)}")

    if level2_result is not None:
        final_state = level2_result["final_state_vector"]
        lower_bound_energy = level2_result["lower_bound_energy"]
        actual_energy = level2_result["actual_energy"]

        if debug:
            print("\n=== Algorithm 17 entangled-state result ===")
            print(f"Final state-vector shape: {final_state.shape}")
            print(f"Final state-vector norm: {np.linalg.norm(final_state)}")
            print(f"Algorithm 17 lower-bound energy: {lower_bound_energy}")
            print(f"Actual entangled-state energy: {actual_energy}")
            print(f"x_ij values: {level2_result['x_dict']}")
            print(f"theta_ij values: {level2_result['theta_dict']}")
            print(f"epsilon_ij signs: {level2_result['epsilon_dict']}")

        assert final_state.shape == (2 ** n_vertices,)
        assert np.isclose(np.linalg.norm(final_state), 1.0, atol=1e-6)
        assert np.isfinite(actual_energy)
        assert actual_energy >= -1e-6

        # For one QMC edge with h = 1/2(I - XX - YY - ZZ), energy should be in [0, 2].
        if n_vertices == 2 and edges == [(0, 1)]:
            assert actual_energy <= 2.0 + 1e-5

    print("\nSanity test completed.")

"""if __name__ == "__main__":
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

    print(f"Objectives found: {objectives}")"""
