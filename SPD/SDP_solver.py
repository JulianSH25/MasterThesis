import cvxpy as cp
import numpy as np
from typing import Literal, TypedDict
from Rounding import round_sdp_with_cholesky
from numpy.matlib import empty
from itertools import permutations
#from testing import visualize_cut

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

        print(constraints)
        print(objective)

        problem = cp.Problem(cp.Maximize(objective), constraints)

        return problem, M, constraints

    def QMC_SDP_solver_antiFerro(self, edges, weights, n_vertices, params: ABCParams):

        a, b, c = params.get("a"), params.get("b"), params.get("c")
        if any(x not in (0, 1) for x in (a, b, c)):
            raise ValueError(f"Expected params a,b,c in {{0,1}}, got {params}")

        # Use the full 3n x 3n formulation with correct (i,k) indexing.
        problem, M, _ = self.SDP_setup(edges, weights, n_vertices, parameters=(a, b, c))

        # Prefer MOSEK, but fall back to SCS if MOSEK is not available/licensed.
        try:
            problem.solve(solver=cp.MOSEK)#, eps=1e-200, verbose=False)
            #problem.solve(solver=cp.SCS, eps=1e-12, verbose=True)
        except Exception:
            print("Warning: MOSEK solver not available; using SCS instead.")
            problem.solve(solver=cp.SCS)

        if problem.status not in (cp.OPTIMAL, "optimal"): #(cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"SDP did not solve to optimality (status={problem.status}). "
                "If this is unexpected, try checking solver output or relaxing constraints."
            )

        return M.value

if __name__ == '__main__':
    solver_sdp = SDP_Solver_()
    edges = [(0, 1)]#, (1, 2), (2, 3), (3, 4), (4, 5), (0,5), (0,4), (2,4)]  # A triangle graph
    weights = [1 for _ in edges]
    n_vertices = 2

    params: ABCParams = {"a": 1, "b": 1, "c": 1}

    M_optimal = solver_sdp.QMC_SDP_solver_antiFerro(edges, weights, n_vertices, params=params)

    print("Optimal moment matrix:")
    print(M_optimal)

    print("Rounding...")
    cut = round_sdp_with_cholesky(M_optimal, parameters=params)
    print("Rounded cut:")
    print(cut)

    #visualize_cut(edges, cut, weights=weights, title="SDP rounded cut")