from utils import get_benchmark_params, build_qaoa_warm_start_state
import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from SPD.Main import main as SDP_main
from SPD.Utilities import idx

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

    print(parameters)

    parameters = {"a": parameters[0], "b": parameters[1], "c": parameters[2]}
    edge_count, edges_in_cut, cuts, M_optimal, states = SDP_main(instance=instance, n_vertices=n_vertices, params=parameters)

    warmstart = build_qaoa_warm_start_state(states=states)
    print(f"Warm start state: {warmstart}")
    return warmstart, M_optimal