import cvxpy as cp
import numpy as np
from typing import Literal, TypedDict
from Rounding import round_sdp_with_cholesky
from numpy.matlib import empty
from itertools import permutations
from testing import visualize_cut

from Utilities import idx

Bit = Literal[0, 1]

class ABCParams(TypedDict):
    a: Bit
    b: Bit
    c: Bit

class SDP_Solver_():
    def SDP_setup(self, edges, weights, n_vertices, parameters: tuple):
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


        for (i, j), w in zip(edges, weights):
            term = 0
            if a == 1:
                term += M[idx(i, 0), idx(j, 0)]
            if b == 1:
                term += M[idx(i, 1), idx(j, 1)]
            if c == 1:
                term += M[idx(i, 2), idx(j, 2)]
            objective += w * (1 - term) / 2
        # Step 3: Defining constraints
        #
        # 3.3 M PSD:
        constraints = [
            M >> 0,
        ]
        # 3.2 Anti commutation:
        constraints += [
            # With symmetric M, antisymmetry implies these entries must be 0.
            M[idx(i, k), idx(i, l)] == 0
            for i in range(n_vertices)
            for k in active_k
            for l in active_k
            if k != l
        ]

        # 3.1 Diagonal entries normalised to 1/enforcing identity for products of equal pauli operators, i.e. p^+ p = I
        constraints += [M[idx(i, k), idx(i, k)] == 1 for i in range(n_vertices) for k in active_k]

        problem = cp.Problem(cp.Maximize(objective), constraints)

        return problem, M, constraints

    """def QMC_SDP_solver(edges, weights, n_vertices):
    
        #problem, M, _ = SDP_setup(edges, weights, n_vertices, parameters=(1, 0, 0))
        # Step 1: Declaring M as a variable
        M = cp.Variable((n_vertices, n_vertices), symmetric=True)
    
        # Step 2: Set up of the objective function
        objective = 0
        for (i, j), w in zip(edges, weights):
            objective += w * (1 - M[i, j]) / 2
    
        # Step 3: Defining constraints
        # 3.3 M PSD:
        constraints = [
            M >> 0,
        ]
    
        # 3.1 Diagonal entries normalised to 1/enforcing identity for products of equal pauli operators, i.e. p^+ p = I
        for i in range(n_vertices):
            constraints.append(M[i, i] == 1)
    
        # Can neglect second constraint (3.2 here) since for the max cut setting it is never the case that we have one pauli operator on the one qubit but a different one on the other; We always only ever apply Pauli-z in QMC
    
        # Step 4: Create and solve the problem
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.MOSEK)
    
        # Step 5: Return the optimal M found by the solver
        return M.value"""

    def QMC_SDP_solver_antiFerro(self, edges, weights, n_vertices, params: ABCParams):

        a, b, c = params.get("a"), params.get("b"), params.get("c")
        if any(x not in (0, 1) for x in (a, b, c)):
            raise ValueError(f"Expected params a,b,c in {{0,1}}, got {params}")

        # Use the full 3n x 3n formulation with correct (i,k) indexing.
        problem, M, _ = self.SDP_setup(edges, weights, n_vertices, parameters=(a, b, c))

        # Prefer MOSEK, but fall back to SCS if MOSEK is not available/licensed.
        try:
            problem.solve(solver=cp.MOSEK)
        except Exception:
            problem.solve(solver=cp.SCS)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"SDP did not solve to optimality (status={problem.status}). "
                "If this is unexpected, try checking solver output or relaxing constraints."
            )

        return M.value

if __name__ == '__main__':
    solver_sdp = SDP_Solver_()
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0,5), (0,4), (2,4)]  # A triangle graph
    weights = [1 for _ in edges]
    n_vertices = 6

    params: ABCParams = {"a": 1, "b": 1, "c": 1}

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params)

    cut = round_sdp_with_cholesky(M_optimal, parameters=params)
    print("Rounded cut:")
    print(cut)

    print("Optimal moment matrix:")
    print(M_optimal)

    print("Rounding...")
    cuts = [round_sdp_with_cholesky(M_optimal, parameters=params) for _ in range(10)]
    print(cuts)

    visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")