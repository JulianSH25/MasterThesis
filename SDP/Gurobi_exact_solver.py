"""Solve classical Max-Cut instances exactly for legacy SDP benchmarks.

The current thesis uses quantum exact energies elsewhere, but the standalone
SDP benchmark pipeline still uses this Gurobi model to validate its classical
cut-rounding experiments.
"""

import os
#os.environ['GRB_LICENSE_FILE'] = '/Users/julian_dev/Projects/University/MasterThesis/gurobi.lic'#'/Users/julian/PycharmProjects/PythonProject/MasterThesis/gurobi.lic'
#os.environ['GRB_LICENSE_FILE'] = '/Users/julian/PycharmProjects/PythonProject/MasterThesis/gurobi.lic'
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

def gurobi_maxcut(n, edges, weights, time_limit=None, mip_gap=None, verbose=True):
    """
    Solve a weighted classical Max-Cut instance with Gurobi.

    Args:
        n: Number of vertices.
        edges: Undirected graph edges.
        weights: Edge weights aligned with ``edges``.
        time_limit: Optional solver time limit in seconds.
        mip_gap: Optional target relative MIP gap.
        verbose: Enable Gurobi's own log output.

    Returns:
        Objective, vertex assignment, edge-cut indicators, and solver status.
    """

    model = Model("maxcut")
    model.setParam("OutputFlag", 1 if verbose else 0)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    # Vars
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    z = model.addVars(len(edges), vtype=GRB.BINARY, name="z")

    # Constraints: z[k] = |y_i - y_j|
    for k, (i, j) in enumerate(edges):
        model.addConstr(z[k] >= y[i] - y[j])
        model.addConstr(z[k] >= y[j] - y[i])
        model.addConstr(z[k] <= y[i] + y[j])
        model.addConstr(z[k] <= 2 - (y[i] + y[j]))

    # Objective
    model.setObjective(quicksum(weights[k] * z[k] for k in range(len(edges))),
                       GRB.MAXIMIZE)

    model.optimize()

    if model.SolCount == 0:
        return None, None, None

    obj = model.ObjVal
    y_sol = [int(y[i].X) for i in range(n)]
    z_sol = [int(z[k].X) for k in range(len(edges))]

    return obj, y_sol, z_sol, model.Status
