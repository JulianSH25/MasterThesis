from Utils import get_benchmark_params, build_qaoa_warm_start_state
import numpy as np
import sys
from pathlib import Path
import time

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from SDP.Main import main as SDP_main
from SDP.Utilities import idx

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

        print(f"idx 0: {M[idx(i, 0), idx(j, 0)]}")
        print(f"idx 1: {M[idx(i, 1), idx(j, 1)]}")
        print(f"idx 2: {M[idx(i, 2), idx(j, 2)]}")

        print(f"idx 01: {M[idx(i, 0), idx(j, 1)]}")
        print(f"idx 10: {M[idx(i, 1), idx(j, 0)]}")
        print(f"idx 02: {M[idx(i, 0), idx(j, 2)]}")
        print(f"idx 20: {M[idx(i, 2), idx(j, 0)]}")
        print(f"idx 12: {M[idx(i, 1), idx(j, 2)]}")
        print(f"idx 21: {M[idx(i, 2), idx(j, 1)]}")

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

    print(f"Warm-start: using parameter_vector={parameters}", flush=True)

    parameters = {"a": parameters[0], "b": parameters[1], "c": parameters[2]}
    sdp_start = time.time()
    print(f"Warm-start: starting SDP solve for n_vertices={n_vertices}", flush=True)
    lasserre_level = int(benchmark_params.get("lasserre_level", 1))
    if lasserre_level not in (1, 2):
        raise ValueError(f"Expected lasserre_level to be 1 or 2, got {lasserre_level}")

    energy, M_optimal, states, classical_cut, level2_result = SDP_main(
        instance=instance,
        n_vertices=n_vertices,
        params=parameters,
        debug=bool(benchmark_params.get("debug", False)),
        lasserre_level=lasserre_level,
    )
    print(f"Warm-start: SDP solve + rounding finished in {time.time() - sdp_start:.2f} seconds", flush=True)

    build_start = time.time()
    if lasserre_level == 2:
        if level2_result is None:
            raise RuntimeError("lasserre_level=2 was requested, but SDP_main returned no level2_result.")
        warmstart = np.asarray(level2_result["final_state_vector"], dtype=complex)
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
    if benchmark_params["debug"]:
        print(f"Warm start state: {warmstart}")
        print("Product states:")
        for idx in range(len(states)):
            print(f"Product state {idx}: {states[idx]}")
        if level2_result is not None:
            print(f"Algorithm 17 actual energy: {level2_result.get('actual_energy')}")
            print(f"Algorithm 17 lower-bound energy: {level2_result.get('lower_bound_energy')}")
        print(f"Moment Matrix: {M_optimal}")
    warm_start_mode = str(benchmark_params.get("warm_start_mode", "standard")).lower()
    need_correlations = warm_start_mode in {"amplified", "entangled"} or bool(benchmark_params.get("use_correlations_as_initial_params", False))
    Moment_matrix = M_optimal if need_correlations else None
    return (warmstart, states, classical_cut), Moment_matrix