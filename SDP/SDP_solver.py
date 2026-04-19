import cvxpy as cp
import numpy as np
from typing import Literal, TypedDict
import time

#from MasterThesis.SDP.Rounding import compute_energy
from .Rounding import round_sdp_with_cholesky
from numpy.matlib import empty
from itertools import permutations
#from testing import visualize_cut

from .Utilities import idx

Bit = Literal[0, 1]

class ABCParams(TypedDict):
    a: Bit
    b: Bit
    c: Bit

class SDP_Solver_():
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
        for (i, j), w in zip(edges, weights):
            term = 1
            if a == 1:
                term -= M[idx(i, 0), idx(j, 0)]
            if b == 1:
                term -= M[idx(i, 1), idx(j, 1)]
            if c == 1:
                term -= M[idx(i, 2), idx(j, 2)]
            objective += w * term
            #objective += w * (1 - term) #/2 #  WHY /2? -> # REVIEW: removed /2
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
        except Exception:
            solver_used = "SCS"
            print("Warning: MOSEK solver not available; using SCS instead.", flush=True)
            problem.solve(solver=cp.SCS, verbose=debug)

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
        #assert params is not None
        #a, b, c = params.get("a"), params.get("b"), params.get("c")
        a, b, c = 1, 1, 1

        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        energy = 0.0
        for (i, j), w in zip(edges, weights):
            # Matches your SDP objective term: w * (1 - a<XX> - b<YY> - c<ZZ>)
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