import os
os.environ['GRB_LICENSE_FILE'] = 'Users/julian_dev/Documents/gurobi.lic'#'/Users/julian/PycharmProjects/PythonProject/MasterThesis/gurobi.lic'
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

def gurobi_maxcut(n, edges, weights, time_limit=None, mip_gap=None, verbose=True):
    """
    Solve Max-Cut with separate edges + weights arrays.

    Args:
        n (int)
        edges: list of (i, j)
        weights: list of floats, same length as `edges`
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

    return obj, y_sol, z_sol