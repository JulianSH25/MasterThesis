from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer

import numpy as np

def bind_circuit_parameters(
    qc: QuantumCircuit,
    gammas,
    betas,
    gamma_values,
    beta_values,
):
    """
    This method binds numeric values to a parameterized QAOA circuit.

    :param qc: Qiskit QuantumCircuit object
    :param gammas: ParameterVector (or list of Parameter) for the cost-layer angles γ[0..p-1] (original placeholders/previous assignments)
    :param betas: ParameterVector (or list of Parameter) for the mixer-layer angles β[0..p-1] (original placeholders/previous assignments)
    :param gamma_values: list of float parameter values for the edge interaction gates
    :param beta_values: list of float parameter values for the individual node X rotation gates
    :return: Qiskit QuantumCircuit object (i.e. parameterized version of the passed QAOA circuit (:param qc), to be used in place of the passed circuit)
    """
    if len(gamma_values) != len(gammas) or len(beta_values) != len(betas):
        raise ValueError(
            f"Length mismatch: len(gamma_values)={len(gamma_values)} vs {len(gammas)}, "
            f"len(beta_values)={len(beta_values)} vs {len(betas)}"
        )

    bind_map = {gammas[i]: float(gamma_values[i]) for i in range(len(gammas))}
    bind_map.update({betas[i]: float(beta_values[i]) for i in range(len(betas))})

    return qc.assign_parameters(bind_map, inplace=False)

def qaoa_maxcut_circuit(n, edges, num_layers, weights=None, add_measurements=True):
    """
    :param n: the size of the circuit
    :param edges: the list of edges
    :param num_layers: the number of iterations p (i.e. rounds of alternating application of Cost and Mixer Unitaries); directly relates to gate complexity, being (n + #edges)*num_layers
    :param weights: list of weights for each edge. If not provided we assume unweighted, i.e. equal weights of 1.0 for all edges
    :param add_measurements: Per default measurements are added to the circuit at the end of the circuit. This can be overridden by providing a boolean "False" for this parameter
    :return: returns the quantum circuit and the used paramter vectors TODO consider adding parameters as another argument
    """
    """
    Sources:
    Farhi et al. QAOA for MaxCut
    Qiskit Documentation (https://quantum.cloud.ibm.com/docs/de/api/qiskit/qiskit.circuit.library.RXGate)
    """
    assert n == len({i for k in edges for i in k})
    if weights is None:
        weights = [1.0] * len(edges)
    assert len(weights) == len(edges)

    # Placeholder parameter vectors (angles)
    gammas = ParameterVector("γ", num_layers)
    betas  = ParameterVector("β", num_layers)

    qc = QuantumCircuit(n, n if add_measurements else 0)

    # Initialise in equal superposition
    # TODO add warm start
    qc.h(range(n))

    for layer in range(num_layers):
        gamma = gammas[layer]#
        beta  = betas[layer]

        # add Cost Hamiltonian for all edges, taking into account their respective weights
        for (j, k), w in zip(edges, weights):
            qc.rzz(-gamma * w, j, k) # z_j z_k, i.e. z interaction term between qubtis j and k

        # add Mixer Hamiltionian for all nodes
        qc.rx(2 * beta, range(n))

    if add_measurements:
        qc.measure(range(n), range(n))

    return qc, gammas, betas

def set_random_params(p: int, seed: int | None = None):
    """Rather pointless method to generate random parameters for the QAOA circuit. Mainly used for initial testing"""
    rng = np.random.default_rng(seed)
    gamma_values = rng.uniform(0.0, 2*np.pi, size=p)
    beta_values = rng.uniform(0.0, np.pi, size=p)

    return gamma_values, beta_values

def run_circuit(
        qc: QuantumCircuit,
        backend: str = "qasm_simulator",
        shots: int = 1024,
        seed: int | None = None
):
    """Runs the circuit on the specified backend and returns the results. A quantum circuit needs to be passed. All other parameters are optional."""
    backend = Aer.get_backend(backend)
    tqc = transpile(qc, backend=backend, optimization_level=1)

    run_args = {"shots": shots}
    if seed is not None:
        run_args["seed_simulator"] = int(seed)

    result = backend.run(tqc, **run_args).result()
    return result.get_counts()


# Example usage:
if __name__ == "__main__":
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]  # ring
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(set_of_nodes, n)
    p = 2

    qc, gammas, betas = qaoa_maxcut_circuit(n, edges, p)
    print(qc.draw("text"))

    gamma_values, beta_values = set_random_params(p)
    print(gamma_values, beta_values)
    qc_bounded = bind_circuit_parameters(qc, gammas, betas, gamma_values, beta_values)
    results = run_circuit(qc_bounded, backend="qasm_simulator", shots=1024)

    sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_results[:10])