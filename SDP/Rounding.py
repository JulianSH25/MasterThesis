import numpy as np
if __package__ in (None, ""):
    from Utilities import idx, save_benchmark_csv
else:
    from .Utilities import idx, save_benchmark_csv
#from utils import get_benchmark_params

debugging = False
benchmark_roundings = False


def parameter_check(parameters: dict, required_keys=None) -> dict:
    """
    This function validates and normalises Hamiltonian parameter flags.

    It accepts either the preferred key set {a, b, c} or the legacy
    set {a, b, y}, and always returns the normalised {a, b, c} format.

    :param parameters: parameter dictionary with binary values
    :param required_keys: expected key set, defaults to {"a", "b", "c"}
    :return: validated and normalised parameter dictionary
    """

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
    """
    Step 2: factorise the moment matrix with a robust PSD repair.

    For Hermitian PSD matrices with small negative eigenvalues caused by
    numerical noise, this routine symmetrises M, clips eigenvalues, rebuilds
    a repaired PSD matrix, and then applies standard Cholesky.

    :param M: matrix to factorise
    :param eps: minimum eigenvalue after clipping
    :return: Cholesky factor of the repaired matrix
    """
    # 1. Force exact Hermitian
    M_ = (M + M.conj().T) / 2

    # 2. Eigenvalue repair
    w, V = np.linalg.eigh(M_)
    w_clipped = np.maximum(w, eps)  # eliminate small negative pivots
    M_psd = V @ np.diag(w_clipped) @ V.conj().T

    # 3. Cholesky on repaired matrix
    M_cholesky = np.linalg.cholesky(M_psd)
    #print(f"Cholesky decomposition of M: {M_cholesky}")
    return M_cholesky

def concat_pauli_blocks(v1, v2, v3, parameters: dict):
    """
    Step 3.1: concatenate Pauli-block vectors selected by parameters.

    The per-vertex vector u_i is built from active blocks among v1, v2, v3,
    controlled by binary switches a, b, c.

    :param v1: X-block vector for one vertex
    :param v2: Y-block vector for one vertex
    :param v3: Z-block vector for one vertex
    :param parameters: binary selector dictionary with keys a, b, c
    :return: concatenated vector without zero padding
    """

    #parameters = parameter_check(parameters) # TODO: NOTE: This was removed for debugging purposes only

    blocks = []
    if parameters["a"] == 1: blocks.append(v1)
    if parameters["b"] == 1: blocks.append(v2)
    if parameters["c"] == 1: blocks.append(v3)
    assert len(blocks) > 0
    return np.concatenate(blocks)

def normalise_vector(u: np.ndarray):
    """
    Step 3.2: normalise a concatenated vector to unit length.

    :param u: input vector
    :return: normalised vector x
    """
    x = u / np.linalg.norm(u)

    return x

def round_normalised_vector(x: np.ndarray, R: np.ndarray):
    """
    Step 3.3: apply random projection and re-normalise the result.

    :param x: normalised input vector
    :param R: random projection matrix
    :return: rounded and normalised vector y
    """
    assert x.ndim == 1 and x.size > 0  # make sure x is a vector and not 'empty'

    y__ = R @ x
    y = y__ / np.linalg.norm(y__)

    return y


def init_random_matrix_for_x(x: np.ndarray, r: int, seed, rng=None):
    """
    Step 3.3a: generate a Gaussian random matrix for rounding.

    The method validates r and x dimensions before sampling entries from
    a normal distribution with mean 0 and standard deviation 1.

    :param x: input vector used to determine matrix width
    :param r: active Pauli dimension, must be 1, 2, or 3
    :param seed: optional random seed for reproducibility
    :param rng: optional RNG object (currently unused)
    :return: random matrix R with shape (r, x.size)
    """
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


def build_single_qubit_state(y, parameters):
    """
    This function constructs one single-qubit state.

    The rounded vector y is mapped to Bloch components according to active
    parameters a, b, c, and then converted into a density matrix.

    :param y: rounded vector for one vertex
    :param parameters: binary selector dictionary with keys a, b, c
    :return: tuple (r_i, state) with Bloch vector and 2x2 density matrix
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex) # Pauli X
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex) # Pauli Y
    Z = np.array([[1, 0], [0, -1]], dtype=complex) # Pauli Z

    r_i = np.zeros(3, dtype=float)

    index = 0

    if parameters["a"] == 1:
        r_i[0] = y[index]
        index += 1
    if parameters["b"] == 1:
        r_i[1] = y[index]
        index += 1
    if parameters["c"] == 1: r_i[2] = y[index]

    state = (I + r_i[0] * X + r_i[1] * Y + r_i[2] * Z)/2

    return r_i, state

def map_product_state_to_cut(product_state):
    """
    This function maps a single-qubit state to a binary cut label.

    The assignment compares diagonal probabilities in the computational basis.

    :param product_state: 2x2 density matrix
    :return: +1 if p(0) > p(1), otherwise -1
    """
    assert product_state.ndim == 2
    x = np.real(np.diag(product_state))

    return 1 if x[0] > x[1] else -1

def round_sdp_with_cholesky(M, parameters: dict, seed = None, debugging: bool = False):
    """
    This function rounds an SDP moment matrix using Cholesky-based projection.

    Pipeline:
    1. Symmetrise M and compute a robust Cholesky factorisation.
    2. Build one per-vertex vector by concatenating active Pauli blocks.
    3. Apply random projection rounding and reconstruct local states.
    4. Map local states to cut assignments.

    :param M: SDP moment matrix, expected in 3n x 3n Pauli-block form
    :param parameters: binary selector dictionary with keys a, b, c
    :param seed: optional random seed for reproducible rounding
    :param debugging: enables additional debug logging
    :return: tuple (cuts, states) with cut labels and single-qubit states
    """
    if seed is not None:
        print(f"Setting seed to {seed}")
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
    product_state = None
    states: list = [None] * n_vertices
    bloch_vectors: list = [None] * n_vertices
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

        r_i, state = build_single_qubit_state(y, parameters)
        bloch_vectors[i] = r_i
        states[i] = state
        product_state = state if product_state is None else np.kron(product_state, state)

        if len(y) == 1:
            y_scalar = True
            cuts.append(y[0])
        else:
            cuts.append(map_product_state_to_cut(state))
        if debugging:
            print(f"r_{i}: {r_i}")
            print(f"Product state qubit {i}: {state}")

            print(f"Diagonal entries of product state {i}: {np.real(np.diag(state))}")

    if debugging:
        print(f"y_scalar: {y_scalar}")
        print(f"Overall Product State: {product_state}")

    return cuts, states, bloch_vectors