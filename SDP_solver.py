import cvxpy as cp
import numpy as np

def QMC_SDP_solver(edges, weights, n_vertices):
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
    problem.solve()

    # Step 5: Return the optimal M found by the solver
    return M.value

if __name__ == '__main__':
    edges = [(0, 1), (0, 3), (2, 1), (2, 3), (0, 2)]  # A triangle graph
    weights = [1, 1, 1, 1, 1]
    n_vertices = 4

    M_optimal = QMC_SDP_solver(edges, weights, n_vertices)
    print("Optimal moment matrix:")
    print(M_optimal)