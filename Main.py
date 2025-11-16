from Utilities import random_instance_generator, get_edges_in_cut, save_benchmark_csv
from SDP_solver import QMC_SDP_solver
from Rounding import round_sdp_with_cholesky
from Gurobi_exact_solver import gurobi_maxcut
import time
import random
import uuid

def benchmark_instance(n_nodes, iid = None):

    weights_prob = random.random()
    weights_static = False if random.random() < 0.5 else True

    edges, weights, n_vertices = random_instance_generator(n_nodes, weights_static)

    while len(edges) == 0:
        edges, weights, n_vertices = random_instance_generator(n_nodes, weights_static)

    print(f'number of vertices = {n_vertices}, number of edges = {len(edges)}')

    start = time.perf_counter()
    M_optimal = QMC_SDP_solver(edges, weights, n_vertices)
    end = time.perf_counter()
    sdp_time = end - start

    print(f"Optimal moment matrix: with dimension: {M_optimal.shape}")
    print(M_optimal)

    start = time.perf_counter()
    rounded_solution = round_sdp_with_cholesky(M_optimal)
    end = time.perf_counter()
    rounding_time = end - start

    print(rounded_solution)
    edge_count, edges_in_cut = get_edges_in_cut(rounded_solution, edges)

    print(edge_count)

    solution_sdp = {
        "instance_id": iid if iid is not None else None,
        "n_vertices": n_vertices,   # vertices in instance
        "n_edges": len(edges),  # number of edges in instance
        "edges": edges,     # edges in instance
        "weights_static": weights_static,
        "weights": weights,     # edge weights of instance
        "SDP_matrix": M_optimal.tolist(),  # M optimal matrix of relaxation
        "sdp_rounded_solution": rounded_solution,   # the solution
        "sdp_edges_in_cut": edges_in_cut,  # edges in solution
        "sdp_cut_value": edge_count,     #objective value of solution
        "sdp_timings": {"sdp": sdp_time, "rounding": rounding_time, "total": sdp_time+rounding_time}
    }

    #Gurobi exact:
    start = time.perf_counter()
    obj, y_sol, z_sol = gurobi_maxcut(n_vertices, edges, weights)
    end = time.perf_counter()
    grb_time = end - start

    grb_edge_count, grb_edges_in_cut = get_edges_in_cut(y_sol, edges)
    print(f'objective is {obj}; Check: edges in cut = {grb_edge_count} vs {edge_count} in the SDP solution')
    print(f'y_sol is {y_sol}')
    print(f'z_sol is {z_sol}')


    solution_gurobi = {
        "n_vertices": n_vertices,  # vertices in instance
        "n_edges": len(edges),  # number of edges in instance
        "edges": edges,  # edges in instance
        "weights": weights,  # edge weights of instance
        "grb_optimal_solution": y_sol,  # the solution
        "grb_edges_in_cut": grb_edges_in_cut,  # edges in solution
        "grb_cut_value": obj,  # objective value of solution
        "grb_timings": grb_time,
        "approx_quality_%": obj/edge_count if edge_count > 0 else 0
    }

    return solution_sdp, solution_gurobi

def automated_benchmark(iid):
    num = int(random.uniform(5, 50))

    sol_sdp, sol_grb = benchmark_instance(num, iid)

    save_benchmark_csv(sol_sdp, sol_grb)

if __name__ == '__main__':
    start = time.perf_counter()
    max_seconds = 24 * 3600
    elapsed = time.perf_counter() - start

    while elapsed <= max_seconds:
        instance_id = str(uuid.uuid4())
        automated_benchmark(instance_id)
        elapsed = time.perf_counter() - start

    print(f'elapsed time = {elapsed}; Terminating Benchmarking!')



