import os
#os.environ['GRB_LICENSE_FILE'] = '/Users/julian_dev/Projects/University/MasterThesis/gurobi.lic'#'/Users/julian/PycharmProjects/PythonProject/MasterThesis/gurobi.lic'
#os.environ['GRB_LICENSE_FILE'] = '/Users/julian/PycharmProjects/PythonProject/MasterThesis/gurobi.lic'
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

def gurobi_maxcut(n, edges, weights, time_limit=None, mip_gap=None, verbose=True):
    """
    This function solves a weighted Max-Cut instance with Gurobi.

    :param n: number of vertices
    :param edges: list of edges as tuples (i, j)
    :param weights: list of edge weights aligned with edges
    :param time_limit: optional solver time limit in seconds
    :param mip_gap: optional MIP optimality gap target
    :param verbose: whether Gurobi solver output is printed
    :return: tuple (objective, y_solution, z_solution, status)
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