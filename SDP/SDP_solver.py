import cvxpy as cp
import numpy as np
from typing import Literal, TypedDict
import time

#from MasterThesis.SDP.Rounding import compute_energy
if __package__ in (None, ""):
    from Rounding import round_sdp_with_cholesky
else:
    from .Rounding import round_sdp_with_cholesky
from numpy.matlib import empty
from itertools import permutations
#from testing import visualize_cut

if __package__ in (None, ""):
    from Lasserre_level_2 import build_pauli_basis_level_2, multiply_pauli_strings, P, PP
else:
    from .Lasserre_level_2 import build_pauli_basis_level_2, multiply_pauli_strings, P, PP

if __package__ in (None, ""):
    from Utilities import idx
else:
    from .Utilities import idx

Lasserre_level = 1
Bit = Literal[0, 1]

class ABCParams(TypedDict):
    a: Bit
    b: Bit
    c: Bit

class SDP_Solver_():
    def SDP_setup_level_2(self, edges, weights, n_vertices, parameters: tuple, debug: bool = False):

        a, b, c = parameters
        assert all(x in (0, 1) for x in parameters)

        active_k = [k for k, flag in enumerate((a, b, c)) if flag == 1]
        assert len(active_k) > 0

        # Step 1: build level-2 Pauli basis
        basis, pidx = build_pauli_basis_level_2(n_vertices) # pidx is the index dictionary/hashmap of the matrix M
        dimensionality = len(basis)

        M = cp.Variable((dimensionality, dimensionality), symmetric=True)

        # Step 2: objective
        objective = 0

        scale = 1 / (1 + a + b + c)

        for (i, j), w in zip(edges, weights):
            term = 1

            if a == 1:
                term -= M[pidx[P(i, 0)], pidx[P(j, 0)]]

            if b == 1:
                term -= M[pidx[P(i, 1)], pidx[P(j, 1)]]

            if c == 1:
                term -= M[pidx[P(i, 2)], pidx[P(j, 2)]]

            objective += scale * w * term

        # Step 3: constraints
        constraints = [M >> 0] # (1): M PSD constraint

        # 3.1 Diagonal normalization: M(A,A) = 1
        constraints += [
            M[r, r] == 1
            for r in range(dimensionality)
        ] # (5): Diagonal entries normalised to 1, M(A,A) = 1 for all A in the basis P_n^(2)

        # 3.2 General Pauli-product consistency constraints
        product_groups = {}

        for r, A in enumerate(basis):
            for s, B in enumerate(basis):
                phase, C = multiply_pauli_strings(A, B)  # A^†B = AB since Paulis are Hermitian

                if phase == 1:
                    key = C
                    sign = 1 # (2): M(A,B) = M(B,A) if A^†B is Hermitian and commutes (i.e. has phase +1)
                elif phase == -1:
                    key = C
                    sign = -1 # (3): M(A,B) = -M(B,A) if A^†B is Hermitian and anti-commutes (i.e. has phase -1)
                elif phase == 1j or phase == -1j: # non hermitian
                    # Product is non-Hermitian, so the real symmetric moment entry is zero
                    constraints.append(M[r, s] == 0) # (4): M(A,B) = 0 if A^†B is anti-Hermitian (i.e. has imaginary phase)
                    continue
                else:
                    raise ValueError(f"Unexpected phase {phase}")

                product_groups.setdefault(key, []).append((r, s, sign)) # (2) and (3): Group entries by product C = A^†B, enforcing M(A,B) = ±M(B,A) within each group (as long as A^†B = C is Hermitian))

        # Within each group, enforce signed equality
        for key, entries in product_groups.items():
            r0, s0, sign0 = entries[0]
            reference = sign0 * M[r0, s0]

            for r, s, sign in entries[1:]:
                constraints.append(reference == sign * M[r, s])

        if debug:
            print(
                f"Level-2 SDP setup complete: n_vertices={n_vertices}, "
                f"matrix_dim={dimensionality}, edges={len(edges)}, "
                f"constraints={len(constraints)}",
                flush=True,
            )

        problem = cp.Problem(cp.Maximize(objective), constraints)

        return problem, M, constraints, basis, pidx
   
    def SDP_setup(self, edges, weights, n_vertices, parameters: tuple, debug: bool = False):
        """
        This method sets up an SDP for Max-Cut with Pauli-block structure.

        Pipeline:
        1. Declare moment matrix M as a 3n x 3n symmetric variable.
        2. Build objective function: maximise sum of weighted edge terms.
        3. Define constraints: M PSD, diagonal entries normalised, anti-commutation.

        :param edges: edge list as tuples (i, j)
        :param weights: edge weights
        :param n_vertices: number of vertices
        :param parameters: tuple (a, b, c) with binary flags for Pauli operators
        :return: tuple (problem, M, constraints)
        """
        # Step 1: Declaring M as a variable
        M = cp.Variable((n_vertices * 3, n_vertices * 3), symmetric=True)

        # Step 2: Set up of the objective function
        objective = 0
        a, b, c = parameters
        assert all(x in (0, 1) for x in parameters)

        # We index M by (vertex i, Pauli k) with k in {0,1,2} (X,Y,Z) using a bijection into {0,...,3n-1}.
        # This avoids collisions that occur with i*k indexing.
        active_k = [k for k, flag in enumerate((a, b, c)) if flag == 1]
        assert len(active_k) > 0


        # build the objective function:
        scale = 1 / (1 + a + b + c)

        for (i, j), w in zip(edges, weights):
            term = 1
            if a == 1:
                term -= M[idx(i, 0), idx(j, 0)]
            if b == 1:
                term -= M[idx(i, 1), idx(j, 1)]
            if c == 1:
                term -= M[idx(i, 2), idx(j, 2)]
            objective += scale * w * term
        # Step 3: Defining constraints
        #
        # 3.3 M PSD:
        constraints = [
            M >> 0,
        ]
        # 3.2 Anti commutation:
        # added check to avoid adding anti-commutation constraints when only one Pauli operator is active, since in that case the anti-commutation constraints are not relevant and only add overhead as well as possibly unwanted side effects.
        #if not (a == 0 and b == 0 and c == 1):
        """constraints += [
            # With symmetric M, antisymmetry implies these entries must be 0.
            M[idx(i, k), idx(i, l)] == - M[idx(i, l), idx(i, k)]
            for i in range(n_vertices)
            for k in range(3)
            for l in range(3)
            if k != l
        ]"""
        for i in range(n_vertices):
            for k in range(3):
                for l in range(3):
                    if k != l:
                        ik = idx(i, k)
                        il = idx(i, l)
                        constraints.append(M[ik, il] == -M[il, ik])
            #else:
         #   print(f"Skipping anti-commutation constraints since only Z is active, which is sufficient for Max-Cut. (parameters={parameters} / a={a}, b={b}, c={c})")

        # 3.1 Diagonal entries normalised to 1/enforcing identity for products of equal pauli operators, i.e. p^+ p = I
        constraints += [M[idx(i,k), idx(i,k)] == 1 for i in range(n_vertices) for k in range(3)]

        if debug:
            print(
                f"SDP setup complete: n_vertices={n_vertices}, matrix_dim={3 * n_vertices}, "
                f"edges={len(edges)}, constraints={len(constraints)}",
                flush=True,
            )

        problem = cp.Problem(cp.Maximize(objective), constraints)

        return problem, M, constraints
    
    def QMC_SDP_solver_antiFerro_level_2(self, edges, weights, n_vertices, params, debug=False):
        try:
            a, b, c = params.get("a"), params.get("b"), params.get("c")
        except AttributeError:
            assert len(params) == 3
            params = {"a": params[0], "b": params[1], "c": params[2]}
            a, b, c = params.get("a"), params.get("b"), params.get("c")

        if any(x not in (0, 1) for x in (a, b, c)):
            raise ValueError(f"Expected params a,b,c in {{0,1}}, got {params}")

        problem, M, _, basis, pidx = self.SDP_setup_level_2(
            edges,
            weights,
            n_vertices,
            parameters=(a, b, c),
            debug=debug,
        )

        try:
            problem.solve(solver=cp.MOSEK, verbose=debug)
        except Exception as e:
            print(f"Warning: MOSEK solver not available: {e};", flush=True)
            #problem.solve(solver=cp.SCS, verbose=debug)
            raise RuntimeError(f"MOSEK solver failed with error: {e}. Please ensure MOSEK is installed and licensed for optimal performance. Falling back to SCS, which may be slower and less accurate.") from e

        if problem.status not in (cp.OPTIMAL, "optimal"):
            raise RuntimeError(f"SDP did not solve to optimality. status={problem.status}")

        self.last_objective_value = problem.value
        return M.value, basis, pidx

    def QMC_SDP_solver_antiFerro(self, edges, weights, n_vertices, params: ABCParams, debug: bool = False):
        """
        This method solves the SDP and returns the optimal moment matrix.

        Uses the MOSEK solver when available, falling back to SCS otherwise.
        Raises an exception if the SDP does not achieve optimal status.

        :param edges: edge list as tuples (i, j)
        :param weights: edge weights
        :param n_vertices: number of vertices
        :param params: SDP Hamiltonian parameters as {"a": Bit, "b": Bit, "c": Bit}
        :return: optimal moment matrix M as a numpy array
        """
        try:
            a, b, c = params.get("a"), params.get("b"), params.get("c")
        except AttributeError:
            assert len(params) == 3
            params = {"a": params[0], "b": params[1], "c": params[2]}
            a, b, c = params.get("a"), params.get("b"), params.get("c")
        if any(x not in (0, 1) for x in (a, b, c)):
            raise ValueError(f"Expected params a,b,c in {{0,1}}, got {params}")

        # Use the full 3n x 3n formulation with correct (i,k) indexing.
        problem, M, _ = self.SDP_setup(edges, weights, n_vertices, parameters=(a, b, c), debug=debug)

        # Prefer MOSEK, but fall back to SCS if MOSEK is not available/licensed.
        start_solve = time.time()
        solver_used = "MOSEK"
        print(
            f"Starting SDP solve (solver preference: MOSEK, fallback: SCS) for n_vertices={n_vertices}, "
            f"matrix_dim={3 * n_vertices}, edges={len(edges)}",
            flush=True,
        )
        try:
            problem.solve(solver=cp.MOSEK, verbose=debug)
        except Exception as e:
            solver_used = "SCS"
            raise RuntimeError(f"MOSEK solver failed with error: {e}. Please ensure MOSEK is installed and licensed for optimal performance. Falling back to SCS, which may be slower and less accurate.") from e
            #print(f"Warning: MOSEK solver not available: {e};", flush=True)
            #problem.solve(solver=cp.SCS, verbose=debug)

        solve_elapsed = time.time() - start_solve
        print(
            f"Finished SDP solve with {solver_used}. status={problem.status}. elapsed_seconds={solve_elapsed:.2f}",
            flush=True,
        )

        if problem.status not in (cp.OPTIMAL, "optimal"): #(cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"SDP did not solve to optimality (status={problem.status}). "
                "If this is unexpected, try checking solver output or relaxing constraints."
            )

        self.last_objective_value = problem.value
        return M.value

    """def compute_energy(product_states, edges, weights=None):
        # H_map = np.zeros((len(edges), len(edges)), dtype=complex)
        weights = weights if weights is not None else np.ones(len(edges))

        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        energy = 0.0
        for (i, j), w in zip(edges, weights):
            H = 0.5 * w(np.kron(I, I) - np.kron(Z, Z))
            p = np.kron(product_states[i], product_states[j])
            energy += np.trace(H @ p)

        return energy
"""
    def compute_energy(self, product_states, edges, weights=None, params=None):
        """
        This method computes the Hamiltonian energy from local single-qubit states.

        Given a list of single-qubit density matrices (one per vertex), computes the
        bilinear energy trace for all edges under the weighted Hamiltonian.

        :param product_states: list of 2x2 density matrices rho_i, one per vertex
        :param edges: edge list as tuples (i, j)
        :param weights: optional edge weights; defaults to 1 per edge
        :param params: binary parameter tuple (a, b, c) controlling Pauli operators
        :return: real energy value of the product state
        """
        weights = weights if weights is not None else np.ones(len(edges))
        if params is None:
            a, b, c = 1, 1, 1
        elif isinstance(params, dict):
            a, b, c = params.get("a"), params.get("b"), params.get("c")
        else:
            a, b, c = params
        if any(v not in (0, 1) for v in (a, b, c)):
            raise ValueError(f"Expected params a,b,c in {{0,1}}, got {(a, b, c)}")

        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        energy = 0.0
        for (i, j), w in zip(edges, weights):
            # Matches the normalized Hamiltonian term used in the SDP objective.
            H_ij = 1/(1+ a+b+c) * w * (
                    np.kron(I, I)
                    - a * np.kron(X, X)
                    - b * np.kron(Y, Y)
                    - c * np.kron(Z, Z)
            )
            #for s in {product_states[i], product_states[j]}: assert np.isclose(np.trace(s), 1.0, atol=1e-8)
            p = np.kron(product_states[i], product_states[j])
            assert np.isclose(np.trace(p), 1.0, atol=1e-8)
            print(f"Hamiltonian equals product state? {H_ij == p}")
            print(f"Hamiltonian: {H_ij}, product state: {p}")
            energy += np.trace(H_ij @ p).real

        return float(energy)

if __name__ == '__main__':
    solver_sdp = SDP_Solver_()
    edges = [(0, 1), (1, 2), (2, 3)]  # A triangle graph
    weights = [1 for _ in edges]
    n_vertices = 4

    params: ABCParams = {"a": 1, "b": 1, "c": 1}

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params)

    print("Optimal moment matrix:")
    print(M_optimal)

    print("Rounding...")
    cut, states = round_sdp_with_cholesky(M_optimal, parameters=params)
    print("Rounded cut:")
    print(cut)
    energy = solver_sdp.compute_energy(states, edges=edges, weights=weights, params=params)
    print(f"Energy of rounded cut: {energy}")
    print("Test")

    #visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")