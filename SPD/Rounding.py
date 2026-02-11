import numpy as np
from Utilities import idx, save_benchmark_csv

debugging = False
benchmark_roundings = False


def parameter_check(parameters: dict, required_keys=None) -> dict:

    # Accept either {a,b,c} (preferred) or {a,b,y} (legacy). Normalize to {a,b,c}.
    keys = set(parameters.keys())
    if keys == {"a", "b", "y"}:
        parameters = {"a": parameters["a"], "b": parameters["b"], "c": parameters["y"]}
        keys = set(parameters.keys())

    if required_keys is None:
        required_keys = {"a", "b", "c"}

    if keys != required_keys:
        raise ValueError(
            f"parameters must have exactly keys {required_keys}, got {keys}"
        )

    for k in required_keys:
        if parameters[k] not in (0, 1):
            raise ValueError(
                f"parameters['{k}'] must be 0 or 1, got {parameters[k]}"
            )

    return parameters

def cholesky_psd(M, eps=1e-12):
    """Step 2: Factorisation of M"""

    """
    Robust Cholesky for Hermitian PSD matrices that may have small
    negative eigenvalues from numerical noise.

    Steps:
      1) Symmetrize A
      2) Project to PSD by clipping eigenvalues
      3) Standard Cholesky
    """
    # 1. Force exact Hermitian
    M_ = (M + M.conj().T) / 2

    # 2. Eigenvalue repair
    w, V = np.linalg.eigh(M_)
    w_clipped = np.maximum(w, eps)  # eliminate small negative pivots
    M_psd = V @ np.diag(w_clipped) @ V.conj().T

    # 3. Cholesky on repaired matrix
    M_cholesky = np.linalg.cholesky(M_psd)
    print(f"Cholesky decomposition of M: {M_cholesky}")
    return M_cholesky

def concat_pauli_blocks(v1, v2, v3, parameters: dict):
    """Step 3.1: Concatenate the vectors retrieved from the factorisation of M

    u_i = (a v_i1) * (b v_i2) * (c v_i3) with the operator * defined as in the paper
    (concatenation except in the 0-vector case; hence no padding)"""

    #parameters = parameter_check(parameters) # TODO: NOTE: This was removed for debugging purposes only

    blocks = []
    if parameters["a"] == 1: blocks.append(v1)
    if parameters["b"] == 1: blocks.append(v2)
    if parameters["c"] == 1: blocks.append(v3)
    assert len(blocks) > 0
    return np.concatenate(blocks)

def normalise_vector(u: np.ndarray):
    """e.g. Step 3.2: concatenated vector u is normalised to unit length"""
    x = u / np.linalg.norm(u)

    return x

def round_normalised_vector(x: np.ndarray, R: np.ndarray):
    """STEP 3.3: Rounding step"""
    assert x.ndim == 1 and x.size > 0  # make sure x is a vector and not 'empty'

    y__ = R @ x
    y = y__ / np.linalg.norm(y__)

    return y


def init_random_matrix_for_x(x: np.ndarray, r: int, seed, rng=None):
    """STEP 3.3a: Generate a random matrix for rounding"""

    """Sanity checks:"""
    if r not in (1, 2, 3):
        raise ValueError("r must be 1, 2, or 3")
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if x.size % r != 0:
        raise ValueError(
            f"x dimension {x.size} not divisible by r={r}"
        )

    rng = np.random.default_rng(seed=seed)
    print(f"Using seed {seed} to generate random matrix for x.")
    R = rng.normal(loc=0.0, scale=1.0, size=(r, x.size)) # Std. deviation 1 and mean 0, as defined in the paper by Parekh and Gharibian
    return R


def round_sdp_with_cholesky(M, parameters: dict, seed = None, debugging: bool = False):
    """
    Round an SDP solution using Goemans-Williamson random hyperplane rounding.

    Given M (moment matrix) where M[i,j] represents the inner product between
    vectors v_i and v_j, we:
    1. Decompose M = L L^T via Choleskyt
    2. Extract vectors v_i as rows of L
    3. Use random hyperplane rounding
    """
    if seed: print(f"Setting seed to {seed}")
    debugging = debugging

    M = (M + M.T) / 2 # symmetrising matrix
    n = M.shape[0]

    # STEP 2: Get Cholesky decomposition: M = L L^T
    V = cholesky_psd(M)

    R: np.ndarray = None
    g: np.ndarray = None

    # If a full 3n x 3n moment matrix is provided and parameters are given,
    # build one vector per vertex by concatenating the Pauli-block vectors v_{iX}, v_{iY}, v_{iZ}.
    parameters = parameter_check(parameters)
    r = parameters["a"] + parameters["b"] + parameters["c"]
    assert r == 1 or r == 2 or r == 3
    if n % 3 != 0:
        raise ValueError(f"Expected M to have dimension 3n x 3n when parameters are provided, got {n}x{n}.")
    n_vertices = n // 3
    print(f"n_vertices: {n_vertices} for n = {n}")

    cuts = []
    y_scalar: bool = False
    for i in range(n_vertices):
        v1 = V[idx(i, 0), :]  # X block
        v2 = V[idx(i, 1), :]  # Y block
        v3 = V[idx(i, 2), :]  # Z block

        u = concat_pauli_blocks(v1, v2, v3, parameters)

        x = normalise_vector(u)

        if R is None:
            R = init_random_matrix_for_x(x=x, r=r, seed=seed)
            if debugging: print(f"Initializing random matrix for x: {R}")
        y = round_normalised_vector(x, R)
        if debugging: print(f"y: {y} with shape {y.shape}")

        if len(y) == 1:
            y_scalar = True
            cuts.append(y[0])
        else:
            cuts.append(y)

    print(f"y_scalar: {y_scalar}")

    return cuts