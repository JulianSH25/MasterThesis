import numpy as np
from Utilities import idx

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
    return np.linalg.cholesky(M_psd)

def concat_pauli_blocks(v1, v2, v3, parameters: dict):
    """Step 3.1: Concatenate the vectors retrieved from the factorisation of M

    u_i = (a v_i1) * (b v_i2) * (c v_i3) with the operator * defined as in the paper
    (concatenation except in the 0-vector case; hence no padding)"""

    parameters = parameter_check(parameters)

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
    assert x.ndim == 1 and x.size > 0  # make sure x is a vector and not 'empty'

    y__ = R @ x
    y = y__ / np.linalg.norm(y__)

    return y


def init_random_matrix_for_x(x: np.ndarray, r: int, rng=None):
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

    rng = rng or np.random.default_rng()
    R = rng.normal(loc=0.0, scale=1.0, size=(r, x.size)) # Std. deviation 1 and mean 0, as defined in the paper by Parekh and Gharibian
    return R


def round_sdp_with_cholesky(M, num_rounds=1, parameters: dict | None = None):
    """
    Round an SDP solution using Goemans-Williamson random hyperplane rounding.

    Given M (moment matrix) where M[i,j] represents the inner product between
    vectors v_i and v_j, we:
    1. Decompose M = L L^T via Choleskyt
    2. Extract vectors v_i as rows of L
    3. Use random hyperplane rounding
    """
    n = M.shape[0]

    # Get Cholesky decomposition: M = L L^T
    L = cholesky_psd(M)

    # The vectors are the ROWS of L (each row i is the vector v_i)
    V = L  # shape: (n_vertices, embedding_dimension)

    # If a full 3n x 3n moment matrix is provided and parameters are given,
    # build one vector per vertex by concatenating the Pauli-block vectors v_{iX}, v_{iY}, v_{iZ}.
    if parameters is not None:
        parameters = parameter_check(parameters)
        if n % 3 != 0:
            raise ValueError(f"Expected M to have dimension 3n x 3n when parameters are provided, got {n}x{n}.")
        n_vertices = n // 3

        per_vertex_vectors = []
        for i in range(n_vertices):
            v1 = V[idx(i, 0), :]  # X block
            v2 = V[idx(i, 1), :]  # Y block
            v3 = V[idx(i, 2), :]  # Z block
            u = concat_pauli_blocks(v1, v2, v3, parameters)
            x = normalise_vector(u)
            per_vertex_vectors.append(x)

        V_round = np.vstack(per_vertex_vectors)  # shape: (n_vertices, dim)
    else:
        V_round = V

    # Hyperplane rounding: we still need this step to convert the embedding vectors into a {0,1} cut.
    def one_round() -> np.ndarray:
        random_normal = np.random.randn(V_round.shape[1])
        random_normal /= np.linalg.norm(random_normal)
        projections = V_round @ random_normal
        return (projections >= 0).astype(int)  # {0,1} assignment

    if num_rounds == 1:
        return one_round()

    cuts = [one_round() for _ in range(num_rounds)]
    return cuts