from Utils import get_benchmark_params, build_qaoa_warm_start_state
import numpy as np
import sys
from pathlib import Path
import time
import secrets

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from SDP.Main import main as SDP_main
from SDP.Utilities import idx

debug = get_benchmark_params().get("debug", False)

"""This is the interface between QAOA and the SDP solver in order to obtain the SDP warm start solution and pass it on to the QAOA circuit preparation"""

def initial_state_rotation(product_state):
    pass

def extract_correlations(M, edges):
    correlations = {}

    for i, j in edges:
        corr = (float(np.real(M[idx(i, 0), idx(j, 0)]))
                + float(np.real(M[idx(i, 1), idx(j, 1)]))
                + float(np.real(M[idx(i, 2), idx(j, 2)]))
        )
        
        """print(f"idx 0: {M[idx(i, 0), idx(j, 0)]}")
        print(f"idx 1: {M[idx(i, 1), idx(j, 1)]}")
        print(f"idx 2: {M[idx(i, 2), idx(j, 2)]}")

        print(f"idx 01: {M[idx(i, 0), idx(j, 1)]}")
        print(f"idx 10: {M[idx(i, 1), idx(j, 0)]}")
        print(f"idx 02: {M[idx(i, 0), idx(j, 2)]}")
        print(f"idx 20: {M[idx(i, 2), idx(j, 0)]}")
        print(f"idx 12: {M[idx(i, 1), idx(j, 2)]}")
        print(f"idx 21: {M[idx(i, 2), idx(j, 1)]}")"""

        correlations[(i, j)] = corr

    return correlations

def get_warm_start_state(instance, n_vertices):
    """
    This method computes a QAOA warm-start state from an SDP solution, to be optionally provided to QAOA.

    :param instance: tuple containing graph edges and edge weights, i.e. instance = (edges, weights)
    :param n_vertices: number of graph vertices in the instance
    :return: warm-start statevector prepared from the SDP states
    """

    #params = {"a": 1, "b": 1, "c": 1}
    benchmark_params: dict = get_benchmark_params()
    parameters = benchmark_params["parameter_vector"]

    print(f"Warm-start: using parameter_vector={parameters}", flush=True) if debug else None

    parameters = {"a": parameters[0], "b": parameters[1], "c": parameters[2]}
    sdp_start = time.time()
    print(f"Warm-start: starting SDP solve for n_vertices={n_vertices}", flush=True)
    lasserre_level = int(benchmark_params.get("lasserre_level"))
    if lasserre_level not in (1, 2):
        raise ValueError(f"Expected lasserre_level to be 1 or 2, got {lasserre_level}")

    initial_solver_level_M = int(benchmark_params.get("initial_solver_level_M", lasserre_level))
    if initial_solver_level_M not in (1, 2):
        raise ValueError(f"Expected initial_solver_level_M to be 1 or 2, got {initial_solver_level_M}")

    sdp_seed = benchmark_params.get("sdp_seed", None)
    if sdp_seed is None:
        sdp_seed = secrets.randbits(64)
        print(f"Warm-start: no sdp_seed in config, generated fallback seed {sdp_seed}", flush=True)
    else:
        sdp_seed = int(sdp_seed)
        print(f"Warm-start: using configured sdp_seed={sdp_seed}", flush=True)

    energy, M_optimal, states, classical_cut, sdp_result = SDP_main(
        instance=instance,
        n_vertices=n_vertices,
        params=parameters,
        debug=bool(benchmark_params.get("debug", False)),
        lasserre_level=lasserre_level,
        initial_solver_level_M=initial_solver_level_M,
        seed=sdp_seed,
    )
    print(f"Warm-start: SDP solve + rounding finished in {time.time() - sdp_start:.2f} seconds", flush=True)

    build_start = time.time()
    if lasserre_level == 2:
        if sdp_result is None or "final_state_vector" not in sdp_result:
            raise RuntimeError("lasserre_level=2 was requested, but SDP_main returned no level-2 result.")
        warmstart = np.asarray(sdp_result["final_state_vector"], dtype=complex)
        warmstart = warmstart / np.linalg.norm(warmstart)
        print(
            f"Warm-start: using level-2 Algorithm 17 entangled statevector with shape {warmstart.shape}",
            flush=True,
        )
    else:
        warmstart = build_qaoa_warm_start_state(states=states)
        print(
            f"Warm-start: built level-1 GP/GW product statevector in {time.time() - build_start:.2f} seconds",
            flush=True,
        )
    if debug:
        print(f"Warm start state: {warmstart}")
        print("Product states:")
        for idx in range(len(states)):
            print(f"Product state {idx}: {states[idx]}")
        if sdp_result is not None:
            print(f"SDP objective value: {sdp_result.get('sdp_objective_value')}")
            if "actual_energy" in sdp_result:
                print(f"Algorithm 17 actual energy: {sdp_result.get('actual_energy')}")
                print(f"Algorithm 17 lower-bound energy: {sdp_result.get('lower_bound_energy')}")
        print(f"Moment Matrix: {M_optimal}")
    warm_start_mode = str(benchmark_params.get("warm_start_mode", "standard")).lower()
    need_correlations = warm_start_mode in {"amplified", "entangled", "amplified_king", "entangled_king"} or bool(benchmark_params.get("use_correlations_as_initial_params", False))
    Moment_matrix = M_optimal if need_correlations else None
    return (warmstart, states, classical_cut), Moment_matrix, sdp_result
