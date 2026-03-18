from Circuit import QAOACircuit
from ParamOptimisation import bayesian_optimisation, optimise_cobyla
import sys
from pathlib import Path
from utils import build_qaoa_warm_start_state

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

optimiser_bayesian = True

def get_warm_start_state(instance, n_vertices):

    params = {"a": 1, "b": 1, "c": 1}
    edge_count, edges_in_cut, cuts, M_optimal, states = SDP_main(instance=instance, n_vertices=n_vertices, params=params)

    warmstart = build_qaoa_warm_start_state(states=states)
    print(f"Warm start state: {warmstart}")
    return warmstart

def main(m = None, p=20, N_bayes=200, init_initial_state = False, self_init_linegraph = False):
    assert m is not None
    edges = [(i, i + 1) for i in range(m)]
    weights = [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    initial_state = get_warm_start_state((edges, weights), n) if init_initial_state else None

    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n} nodes, p: {p} layers, N_bayes: {N_bayes} iterations")
    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights)

    QAOA.initial_state = initial_state
    print(f"Initial state: {initial_state}") if initial_state is not None else print("No initial state provided.")
    QAOA.self_init_linegraph = self_init_linegraph
    QAOA.build_qaoa_maxcut_circuit(add_measurements=False) # TODO check parameter (changed from True to False)

    minimum_energy = bayesian_optimisation(QAOA=QAOA, N_bayes=N_bayes, no_layers=p) if optimiser_bayesian else optimise_cobyla(QAOA=QAOA, no_layers=p, max_iter=N_bayes)


    print(minimum_energy)
    if not optimiser_bayesian:
        print(minimum_energy.fun)

if __name__ == "__main__":
    #for m in range(5, 15):
    results = []
    #results.append(main(m=3, p=20, N_bayes=10, init_initial_state=True))
    results.append(main(m=int(sys.argv[1]), p=20, N_bayes=200, init_initial_state=True))
    #results.append(main(m=3, p=20, N_bayes=10))

    print(f"results: {results}")
    #main(m=sys.argv[1], p=int(sys.argv[2]), N_bayes=int(sys.argv[3]))