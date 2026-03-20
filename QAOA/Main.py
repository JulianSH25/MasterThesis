from Circuit import QAOACircuit
from ParamOptimisation import BayesianOptimiser, optimise_cobyla, grid_search
import sys
from pathlib import Path
from utils import build_qaoa_warm_start_state, get_benchmark_params

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from SPD.Main import main as SDP_main

#edges = [(0, 1), (1, 2)]  # , (2, 3), (3, 4), (4, 5), (5, 6)]  # ring
#weights = [1.0] * len(edges)
#set_of_nodes = {i for k in edges for i in k}
#n = len(set_of_nodes)
#print(set_of_nodes, n)
#p = 20

optimiser_bayesian = False
optimiser_cobyla = False
gridsearch = True

precision = None

benchmark_params: dict = get_benchmark_params()
parameters = benchmark_params["parameter_vector"]

def get_warm_start_state(instance, n_vertices):
    """
    This method computes a QAOA warm-start state from an SDP solution, to be optionally provided to QAOA.

    :param instance: tuple containing graph edges and edge weights, i.e. instance = (edges, weights)
    :param n_vertices: number of graph vertices in the instance
    :return: warm-start statevector prepared from the SDP states
    """

    #params = {"a": 1, "b": 1, "c": 1}
    benchmark_params: dict = get_benchmark_params()
    parameters = benchmark_params["parameter_vector"]

    print(parameters)

    parameters = {"a": parameters[0], "b": parameters[1], "c": parameters[2]}
    edge_count, edges_in_cut, cuts, M_optimal, states = SDP_main(instance=instance, n_vertices=n_vertices, params=parameters)

    warmstart = build_qaoa_warm_start_state(states=states)
    print(f"Warm start state: {warmstart}")
    return warmstart

def main(m = None, p=20, N_bayes=200, init_initial_state = False, self_init_linegraph = False):
    """
    This method builds and optimises a QAOA instance on a line graph.

    :param m: number of edges for the generated line graph
    :param p: number of QAOA layers
    :param N_bayes: number of optimisation iterations
    :param init_initial_state: whether to inject an SDP-derived warm-start state
    :param self_init_linegraph: whether to use line-graph singlet state preparation
    :return: None
    """
    assert m is not None
    edges = [(i, i + 1) for i in range(m)] # Optionally replace by desired edge list, if a linegraph is not desired
    weights = [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    initial_state = get_warm_start_state((edges, weights), n) if init_initial_state else None

    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n} nodes, p: {p} layers, N_bayes: {N_bayes} iterations")
    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights)

    benchmark_params: dict = get_benchmark_params()
    QAOA.params = benchmark_params["parameter_vector"]

    QAOA.initial_state = initial_state
    print(f"Initial state: {initial_state}") if initial_state is not None else print("No initial state provided.")
    QAOA.self_init_linegraph = self_init_linegraph
    QAOA.build_qaoa_maxcut_circuit(add_measurements=False) # TODO check parameter (changed from True to False)

    BO = BayesianOptimiser()

    assert sum([optimiser_bayesian, optimiser_cobyla, gridsearch]) == 1
    minimum_energy = None
    if optimiser_bayesian:
        minimum_energy = BO.bayesian_optimisation(QAOA=QAOA, N_bayes=N_bayes, no_layers=p)
    elif optimiser_cobyla:
        minimum_energy = optimise_cobyla(QAOA=QAOA, no_layers=p, max_iter=N_bayes)
    elif gridsearch:
        minimum_energy = grid_search(QAOA, p, precision=precision)

    print(minimum_energy)
    if optimiser_cobyla:
        print(minimum_energy.fun)
        return minimum_energy.fun

    return minimum_energy


if __name__ == "__main__":
    #for m in range(5, 15):
    parameter_settings = get_benchmark_params()
    print(f"Benchmark parameters: {parameter_settings}")
    results = []
    precision = 0.1
    for m in range(1, 10):
        print(f"Running QAOA for m={m} edges...; Precision: {precision}")
        results.append(main(m=m, p=2, N_bayes=10, self_init_linegraph=True))
    #results.append(main(m=int(sys.argv[1]), p=20, N_bayes=200, init_initial_state=True))
    #results.append(main(m=3, p=20, N_bayes=10))

    print(f"results: {results}")
    #main(m=sys.argv[1], p=int(sys.argv[2]), N_bayes=int(sys.argv[3]))