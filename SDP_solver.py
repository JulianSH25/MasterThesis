import cvxpy as cp
import numpy as np
from Rounding import round_sdp_with_cholesky
from numpy.matlib import empty


def SDP_setup(edges, weights, n_vertices, parameters: tuple):
    # Step 1: Declaring M as a variable
    M = cp.Variable((n_vertices * 3, n_vertices * 3), symmetric=True)

    # Step 2: Set up of the objective function
    objective = 0
    a, b, c = parameters
    assert all(x in (0, 1) for x in parameters)

    for (i, j), w in zip(edges, weights):
        objective += w * (1 - a * M[i*1, j*1] - b * M[i*2, j*2] - c * M[i*3, j*3]) / 2

    scalars = {a * 1, b * 2, c * 3} - {0}
    assert 0 not in scalars
    assert len(scalars) > 0
    # Step 3: Defining constraints
    #
    # 3.3 M PSD:
    constraints = [
        M >> 0,
    ]
    # 3.2 Anti commutation:
    constraints += [
        M[i * k, i * l] == -M[i * l, i * k]
        for i in range(n_vertices)
        for k in scalars
        for l in scalars
        if k != l
    ]

    # 3.1 Diagonal entries normalised to 1/enforcing identity for products of equal pauli operators, i.e. p^+ p = I
    constraints += [M[i * k, i * k] == 1 for i in range(n_vertices) for k in scalars]

    problem = cp.Problem(cp.Maximize(objective), constraints)

    return problem, M, constraints

def QMC_SDP_solver(edges, weights, n_vertices):

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
    return M.value

if __name__ == '__main__':
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  # A triangle graph
    weights = [1, 1, 1, 1, 1]
    n_vertices = 6

    M_optimal = QMC_SDP_solver(edges, weights, n_vertices)

    print(round_sdp_with_cholesky(M_optimal))
    print("Optimal moment matrix:")
    print(M_optimal)