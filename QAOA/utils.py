import json
from pathlib import Path
import numpy as np
import random
from collections import Counter

def set_random_params(p: int, seed: int | None = None, init_close_to_zero: bool = False):
    """
    This function samples random QAOA angle parameters for a given circuit depth p.

    :param p: number of QAOA layers
    :param seed: optional random seed for reproducible sampling
    :return: tuple (gamma_values, beta_values) as numpy arrays
    """
    rng = np.random.default_rng(seed)
    gamma_values = rng.uniform(1e-10, 0.1 if init_close_to_zero else 2*np.pi, size=p)
    beta_values = rng.uniform(1e-10, 0.1 if init_close_to_zero else np.pi, size=p)

    return gamma_values, beta_values

def sample_initial_qaoa_params(n: int, p: int) -> list[tuple[list[float], list[float]]]:
    """
    This function samples initial QAOA parameter tuples for Bayesian optimisation. I realise it is basically a duplicate of the above function I already used before in a different place.

    :param n: number of parameter points to sample
    :param p: number of QAOA layers per parameter vector
    :return: list of tuples (gamma_values, beta_values)
    """
    return [
        (
            [random.uniform(0, 3.141592653589793) for _ in range(p)],
            [random.uniform(0, 3.141592653589793 / 2) for _ in range(p)],
        )
        for _ in range(n)
    ]

def build_qaoa_warm_start_state(states: list[np.ndarray]):
    """
    This function builds a global warm-start statevector from local density matrices.

    :param states: list of local density matrices as returned from the SDP solver, one for each edge in the graph; each state is a 4x4 numpy array representing the two-qubit density matrix for the corresponding edge
    :return: normalised global warm-start statevector
    """
    statevec = np.array([1.0 + 0.0j])

    for state in states:
        ew, ev = np.linalg.eigh(state)
        idx = np.argmax(ew) # maximum eigenvalue index
        max_ew = float(ew[idx]) # maximum eigenvalue, used to find out how pure the state is
        x = ev[:, idx] / np.linalg.norm(ev[:, idx])

        if max_ew < 1 - 1e-10: raise ValueError("State is not pure enough for unique ket extraction; maximum Eigenvalue is " + str(max_ew))
        statevec = np.kron(statevec, x)

    return statevec / np.linalg.norm(statevec)

def get_benchmark_params() -> dict:
    """
    This function loads benchmark parameters from JSON.

    :return: benchmark parameter dictionary exactly as stored in JSON
    """
    config_path = Path(__file__).resolve().parent / "benchmark_config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)

def classify_graph(edges):
    verts = {u for e in edges for u in e}
    n = len(verts)
    m = len(edges)

    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    d = list(deg.values())

    if m == n * (n - 1) // 2:
        return "complete"
    if n >= 3 and all(x == 2 for x in d):
        return "cycle"
    if d.count(1) == 2 and d.count(2) == n - 2:
        return "path"
    return None

