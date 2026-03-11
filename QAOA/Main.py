from Circuit import QAOACircuit
from ParamOptimisation import bayesian_optimisation
import sys

#edges = [(0, 1), (1, 2)]  # , (2, 3), (3, 4), (4, 5), (5, 6)]  # ring
#weights = [1.0] * len(edges)
#set_of_nodes = {i for k in edges for i in k}
#n = len(set_of_nodes)
#print(set_of_nodes, n)
#p = 20

def main(m_ = None):
    assert sys.argv[1] is not None
    m = int(sys.argv[1]) if len(sys.argv) > 1 else m_
    assert m is not None
    edges = [(i, i + 1) for i in range(m)]
    weights = [1.0] * len(edges)
    p = 20
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(f"Generated line graph with {m} edges, {n} nodes, and {p} layers.") if m == n - 1 else None

    assert edges is not None and weights is not None and set_of_nodes is not None and n is not None and p is not None
    print(f"Edges: {edges}, weights: {weights}, set of nodes: {set_of_nodes}, n: {n}, p: {p}")
    QAOA = QAOACircuit(n=n, p=p, edges=edges, weights=weights)
    QAOA.build_qaoa_maxcut_circuit(add_measurements=True)

    minimum_energy = bayesian_optimisation(QAOA=QAOA, N_bayes=200, no_layers=p)

    print(minimum_energy)

if __name__ == "__main__":
    for m in range(5, 15):
        main(m)