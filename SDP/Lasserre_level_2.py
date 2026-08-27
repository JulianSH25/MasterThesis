from itertools import combinations
from scipy.special import hyp2f1
from numpy import pi
import numpy as np
import cvxpy as cp

# Pauli labels:
# 0 -> X
# 1 -> Y
# 2 -> Z

def canonical_pauli_string(items):
    """
    Canonical representation of a Pauli string.

    items: iterable of (vertex, pauli)
    returns: sorted tuple ((i, k), ...)
    """
    return tuple(sorted(items, key=lambda x: x[0]))


def P(i: int, k: int):
    """
    One-body Pauli string P_i.
    """
    return ((i, k),)


def PP(i: int, k: int, j: int, l: int):
    """
    Two-body Pauli string P_i Q_j.
    Assumes i != j.
    """
    if i == j:
        raise ValueError("Use Pauli multiplication for same-vertex products.")
    return canonical_pauli_string(((i, k), (j, l)))


def build_pauli_basis_level_2(n_vertices: int):
    """
    Build P_n^(2): identity, all one-body Paulis, all two-body Paulis.
    """
    basis = [()]  # identity

    # Weight-1 Paulis
    for i in range(n_vertices):
        for k in range(3):
            basis.append(P(i, k))

    # Weight-2 Paulis
    for i, j in combinations(range(n_vertices), 2):
        for k in range(3):
            for l in range(3):
                basis.append(PP(i, k, j, l))

    pidx = {pauli_string: idx for idx, pauli_string in enumerate(basis)}
    return basis, pidx


def single_pauli_multiply(a: int, b: int):
    """
    Multiply single-qubit Paulis.

    Pauli encoding:
        0 -> X
        1 -> Y
        2 -> Z

    Returns:
        phase, result_pauli

    If result is identity, result_pauli is None.
    """
    if a == b:
        return 1, None

    # X Y = i Z
    if (a, b) == (0, 1):
        return 1j, 2
    # Y Z = i X
    if (a, b) == (1, 2):
        return 1j, 0
    # Z X = i Y
    if (a, b) == (2, 0):
        return 1j, 1

    # Reverse order gives negative phase
    if (a, b) == (1, 0):
        return -1j, 2
    if (a, b) == (2, 1):
        return -1j, 0
    if (a, b) == (0, 2):
        return -1j, 1

    raise ValueError(f"Invalid Pauli product: {a}, {b}")


def multiply_pauli_strings(A, B):
    """
    Multiply two Pauli strings A * B.

    A, B are tuples like:
        ()
        ((0, 0),)
        ((0, 0), (3, 2))

    Returns:
        phase, C

    where phase in {1, -1, 1j, -1j}
    and C is the canonical resulting Pauli string.
    """
    phase = 1
    result = {}

    for i, k in A:
        result[i] = k

    for i, k_b in B:
        if i not in result:
            result[i] = k_b
        else:
            k_a = result[i]
            local_phase, k_c = single_pauli_multiply(k_a, k_b)
            phase *= local_phase

            if k_c is None:
                del result[i]
            else:
                result[i] = k_c

    C = canonical_pauli_string(result.items())

    # Normalize phase to avoid weird Python complex comparison issues
    if phase == 1:
        phase = 1
    elif phase == -1:
        phase = -1
    elif phase == 1j:
        phase = 1j
    elif phase == -1j:
        phase = -1j
    else:
        raise ValueError(f"Unexpected phase: {phase}")

    return phase, C

class Level_2_Rounding:

    def __init__(self):
        self.edges: list[tuple[int, int]] = []
        self.weights: list[float] = []
        self.n_vertices: int = 0
        self.M_level2: np.ndarray | None = None
        self.basis: list[tuple] | None = None
        self.pidx: dict[tuple, int] | None = None
        self.x_dict: dict = {}
        self.theta_dict: dict = {}
        self.bloch_vectors: list[np.ndarray] = []
        self.n_vectors: list = []
        self.P_matrices: list = []
        self.epsilon_dict: dict = {}
        self.analytic_F_value: float | None = None
        self.analytic_F_bound_applicable: bool = False
        self.output: dict | None = None

    def build_x_vars(self):
        I = ()

        use_reduced_matrix = self.pidx is None

        for (i, j) in self.edges:
            if use_reduced_matrix:
                if self.M_level2.shape[0] != 3 * self.n_vertices:
                    raise ValueError(
                        "Reduced-matrix Algorithm 17 mode expects a 3n x 3n matrix."
                    )

                term = (
                    1.0
                    + self.M_level2[3 * i + 0, 3 * j + 0]
                    + self.M_level2[3 * i + 1, 3 * j + 1]
                    + self.M_level2[3 * i + 2, 3 * j + 2]
                )
            else:
                XiXj = PP(i, 0, j, 0)
                YiYj = PP(i, 1, j, 1)
                ZiZj = PP(i, 2, j, 2)
                term = (
                    1.0
                    + self.M_level2[self.pidx[I], self.pidx[XiXj]]
                    + self.M_level2[self.pidx[I], self.pidx[YiYj]]
                    + self.M_level2[self.pidx[I], self.pidx[ZiZj]]
                )

            self.x_dict[(i, j)] = float(np.real(-0.5 * term))

    @staticmethod
    def F(beta, x):
        """
        Function F(beta; x) from Algorithm 17, paper by Robbie King.
        F(beta; x) = f(x) * (0.5 + 0.5(1 - beta^2(1-x)^2) + (2/pi) beta x (sqrt(1-beta^2) + (1-sqrt(1-beta^2)) x))
        """
        x = float(np.clip(np.real(x), -1.0, 1.0))

        def f(x_ij):
            t = -(1.0 + 2.0 * x_ij) / 3.0
            return 0.5 - (4.0 / (3.0 * pi)) * t * hyp2f1(
                0.5, 0.5, 2.5, t**2
            )

        sqrt_term = np.sqrt(max(0.0, 1.0 - beta**2))
        second_term = (
            0.5
            + 0.5 * (1.0 - beta**2 * (1.0 - x)**2)
            + (2.0 / np.pi) * beta * x * (
                sqrt_term + (1.0 - sqrt_term) * x
            )
        )

        return float(f(x_ij=x) * second_term)

    # Step 5
    beta_star = 0.390 # approximate optimal beta from paper

    # Step 6
        # Step 6
    def build_theta_vars(self):
        """
        Step 6 of Algorithm 17.

        theta_ij = 1/2 * arcsin(beta_star * x_ij) if x_ij >= 0
                 = 0 otherwise
        """
        if not self.x_dict:
            self.build_x_vars()

        self.theta_dict = {}

        for edge, x_ij in self.x_dict.items():
            if x_ij >= 0:
                arg = np.clip(self.beta_star * x_ij, -1.0, 1.0)
                theta_ij = 0.5 * np.arcsin(arg)
            else:
                theta_ij = 0.0

            self.theta_dict[edge] = float(theta_ij)

        return self.theta_dict
    
    @staticmethod
    def pauli_matrices():
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return X, Y, Z

    # Step 7
    def sample_orthogonal_n_vectors(self, seed=None):
        """
        Step 7 of Algorithm 17.

        For each vertex k, sample a random unit vector n_k satisfying

            n_k · v_k = 0

        where v_k is the Bloch vector of the GP-rounded product state.
        """
        rng = np.random.default_rng(seed)
        self.n_vectors = []

        for k, v_k in enumerate(self.bloch_vectors):
            v_k = np.asarray(v_k, dtype=float)
            norm_v = np.linalg.norm(v_k)

            if norm_v < 1e-10:
                raise ValueError(
                    f"Bloch vector at vertex {k} has near-zero norm. "
                    "Algorithm 17 expects pure-state Bloch vectors."
                )

            v_k = v_k / norm_v

            # Draw random vector and project away the component parallel to v_k
            r = rng.normal(size=3)
            n_k = r - np.dot(r, v_k) * v_k

            while np.linalg.norm(n_k) < 1e-10:
                r = rng.normal(size=3)
                n_k = r - np.dot(r, v_k) * v_k

            n_k = n_k / np.linalg.norm(n_k)
            self.n_vectors.append(n_k)

        return self.n_vectors

    # Step 8
    def build_P_matrices(self):
        """
        Step 8 of Algorithm 17.

        P_k = n_k · sigma_k
        """
        if not hasattr(self, "n_vectors") or not self.n_vectors:
            self.sample_orthogonal_n_vectors()

        X, Y, Z = self.pauli_matrices()
        self.P_matrices = []

        for n_k in self.n_vectors:
            P_k = n_k[0] * X + n_k[1] * Y + n_k[2] * Z
            self.P_matrices.append(P_k)

        return self.P_matrices
    
    @staticmethod
    def pure_state_from_bloch(v):
        """
        Convert a unit Bloch vector into one compatible pure state vector.

        v = (x, y, z)
        |v> = cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>
        """
        v = np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)

        x, y, z = v

        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)

        return np.array(
            [
                np.cos(theta / 2.0),
                np.exp(1j * phi) * np.sin(theta / 2.0),
            ],
            dtype=complex,
        )

    # Step 9
    def build_epsilon_signs(self):
        """
        Step 9 of Algorithm 17.

        Choose epsilon_ij based on the phase of

            <v_i|P_j|v_j> <v_j|P_i|v_i>

        If arg is in [0, pi), epsilon = +1.
        If arg is in [-pi, 0), epsilon = -1.
        """
        if not hasattr(self, "P_matrices") or not self.P_matrices:
            self.build_P_matrices()

        state_vectors = [
            self.pure_state_from_bloch(v_k)
            for v_k in self.bloch_vectors
        ]

        self.epsilon_dict = {}

        for i, j in self.edges:
            v_i = state_vectors[i]
            v_j = state_vectors[j]

            P_i = self.P_matrices[i]
            P_j = self.P_matrices[j]

            overlap_1 = np.vdot(v_i, P_j @ v_j)
            overlap_2 = np.vdot(v_j, P_i @ v_i)

            phase = np.angle(overlap_1 * overlap_2)

            epsilon_ij = +1 if phase >= 0 else -1
            self.epsilon_dict[(i, j)] = epsilon_ij

        return self.epsilon_dict
    
    # Step 10
    def build_final_state(self, max_vertices: int = 16):
        """
        Step 10 of Algorithm 17.

        Builds the final entangled state vector

            prod_ij exp(i epsilon_ij theta_ij P_i ⊗ P_j) ⊗_k |v_k>

        and computes:
            - analytic_F_value = sum_ij w_ij F(beta_star; x_ij)
            - lower_bound_energy = analytic_F_value for triangle-free graphs
            - actual_energy = <psi|H|psi>

        This explicitly materialises a 2^n state vector, so it is only practical
        for small n.
        """
        if self.n_vertices > max_vertices:
            raise ValueError(
                f"Refusing to build explicit state vector for n={self.n_vertices}. "
                f"State dimension would be 2^{self.n_vertices}. "
                f"Increase max_vertices if intentional."
            )

        if not hasattr(self, "theta_dict") or not self.theta_dict:
            self.build_theta_vars()

        if not hasattr(self, "P_matrices") or not self.P_matrices:
            self.build_P_matrices()

        if not hasattr(self, "epsilon_dict") or not self.epsilon_dict:
            self.build_epsilon_signs()

        if not hasattr(self, "x_dict") or not self.x_dict:
            self.build_x_vars()

        # (removed local pure_state_from_bloch)

        def kron_all(vectors):
            result = vectors[0]
            for vec in vectors[1:]:
                result = np.kron(result, vec)
            return result

        def apply_two_qubit_gate(state, gate, i, j):
            if i == j:
                raise ValueError("Cannot apply two-qubit gate to same qubit.")

            psi = state.reshape([2] * self.n_vertices)

            axes = [i, j] + [q for q in range(self.n_vertices) if q not in (i, j)]
            inv_axes = np.argsort(axes)

            psi_perm = np.transpose(psi, axes)
            psi_perm = psi_perm.reshape(4, -1)

            psi_perm = gate @ psi_perm

            psi_perm = psi_perm.reshape([2, 2] + [2] * (self.n_vertices - 2))
            psi = np.transpose(psi_perm, inv_axes)

            return psi.reshape(-1)

        def two_qubit_expectation(state, operator, i, j):
            psi = state.reshape([2] * self.n_vertices)

            axes = [i, j] + [q for q in range(self.n_vertices) if q not in (i, j)]
            psi_perm = np.transpose(psi, axes)
            psi_perm = psi_perm.reshape(4, -1)

            op_psi = operator @ psi_perm

            return float(np.vdot(psi_perm, op_psi).real)

        # Initial GP product state ⊗_k |v_k>
        product_state_vectors = [
            self.pure_state_from_bloch(v_k)
            for v_k in self.bloch_vectors
        ]

        state = kron_all(product_state_vectors)

        # Apply gates exp(i epsilon_ij theta_ij P_i ⊗ P_j)
        I4 = np.eye(4, dtype=complex)

        for i, j in self.edges:
            theta_ij = self.theta_dict[(i, j)]
            epsilon_ij = self.epsilon_dict[(i, j)]

            generator = np.kron(self.P_matrices[i], self.P_matrices[j])

            # Since generator^2 = I:
            # exp(i a G) = cos(a) I + i sin(a) G
            angle = epsilon_ij * theta_ij
            gate = np.cos(angle) * I4 + 1j * np.sin(angle) * generator

            state = apply_two_qubit_gate(state, gate, i, j)

        # Theorem 3 in King applies this analytical bound to triangle-free graphs.
        neighbours = [set() for _ in range(self.n_vertices)]
        for i, j in self.edges:
            neighbours[i].add(j)
            neighbours[j].add(i)
        graph_is_triangle_free = not any(
            neighbours[i].intersection(neighbours[j])
            for i, j in self.edges
        )

        analytic_F_value = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            analytic_F_value += w * self.F(self.beta_star, self.x_dict[(i, j)])
        lower_bound_energy = analytic_F_value if graph_is_triangle_free else None

        # Actual QMC energy <psi|H|psi>
        I = np.eye(2, dtype=complex)
        X, Y, Z = self.pauli_matrices()

        # King/QMC convention: h_ij = 1/2(I - XX - YY - ZZ).
        h_qmc = 0.5 * (
            np.kron(I, I)
            - np.kron(X, X)
            - np.kron(Y, Y)
            - np.kron(Z, Z)
        )

        actual_energy = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            actual_energy += w * two_qubit_expectation(state, h_qmc, i, j)

        self.product_state_vectors = product_state_vectors
        self.final_state_vector = state
        self.analytic_F_value = float(analytic_F_value)
        self.analytic_F_bound_applicable = graph_is_triangle_free
        self.lower_bound_energy = float(lower_bound_energy) if lower_bound_energy is not None else None
        self.actual_energy = float(actual_energy)

        return {
            "product_state_vectors": product_state_vectors,
            "final_state_vector": state,
            "analytic_F_value": self.analytic_F_value,
            "analytic_F_bound_applicable": self.analytic_F_bound_applicable,
            "lower_bound_energy": self.lower_bound_energy,
            "actual_energy": self.actual_energy,
            "x_dict": self.x_dict,
            "theta_dict": self.theta_dict,
            "epsilon_dict": self.epsilon_dict,
        }
    
    def QMC_rounding(self, seed=None, max_vertices: int = 16):
        """
        Wrapper for King's Algorithm 17 postprocessing for QMC.

        Assumes Step 1 and Step 2 are already done externally:
            Step 1: Solve Lasserre2(H)
            Step 2: Run GP rounding on the level-1 submatrix to obtain Bloch vectors.

        This wrapper performs:
            Step 3: build x_ij values
            Step 5: use beta_star
            Step 6: build theta_ij values
            Step 7: sample n_k orthogonal to v_k
            Step 8: build P_k = n_k · sigma
            Step 9: choose epsilon_ij signs
            Step 10: build final entangled state and compute energy metrics

        :param seed: optional RNG seed for Step 7
        :param max_vertices: max n for explicit 2^n state-vector construction
        :return: dict containing final state, initial product vectors, energies, and intermediate values
        """
        required_attrs = [
            "edges",
            "weights",
            "n_vertices",
            "M_level2",
            "bloch_vectors",
            "beta_star",
        ]

        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"QMC_rounding requires self.{attr} to be set.")

        if self.M_level2 is None:
            raise ValueError("self.M_level2 is None.")

        if self.bloch_vectors is None:
            raise ValueError("self.bloch_vectors is None.")

        if len(self.bloch_vectors) == 0:
            raise ValueError("self.bloch_vectors is empty.")

        # Step 3
        self.build_x_vars()

        # Step 6
        self.build_theta_vars()

        # Step 7
        self.sample_orthogonal_n_vectors(seed=seed)

        # Step 8
        self.build_P_matrices()

        # Step 9
        self.build_epsilon_signs()

        # Step 10
        output = self.build_final_state(max_vertices=max_vertices)

        # Add intermediate quantities explicitly, in case Step 10 does not already include all of them
        output.update(
            {
                "x_dict": self.x_dict,
                "theta_dict": self.theta_dict,
                "P_matrices": self.P_matrices,
                "epsilon_dict": self.epsilon_dict,
            }
        )
        
        self.output = output

        return output
