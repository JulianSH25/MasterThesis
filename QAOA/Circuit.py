from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

from StatePrep import prepare_line_singlet_circuit
from utils import set_random_params
import numpy as np

use_measurements = False

class QAOACircuit(QuantumCircuit):
    def __init__(self, n, p, edges, weights):
        """
        Initialises QAOA circuit metadata and backend configuration.

        :param n: number of nodes (qubits)
        :param p: circuit depth (number of alternating layers)
        :param edges: list of graph edges as qubit index pairs
        :param weights: edge weights aligned with edges
        """
        self.n = n # number of nodes
        self.p = p # Circuit depth; number of layers
        self.edges = edges
        self.weights = weights
        self.gammas: np.ndarray[float] = None
        self.betas: np.ndarray[float] = None
        self.qc: QuantumCircuit = None
        self.qc_no_params: QuantumCircuit = None # Auxiliary variable that is used to update the quantum circuit parameters
        self.backend = Aer.get_backend('qasm_simulator')
        self.initial_state = None
        self.self_init_linegraph = False

    def bind_circuit_parameters(
        self,
        gamma_values,
        beta_values,
    ) -> None:
        """
        This method binds numeric values to a parameterized QAOA circuit.

        # made class variable: :param qc: Qiskit QuantumCircuit object
        # made class variable: :param gammas: ParameterVector (or list of Parameter) for the cost-layer angles γ[0..p-1] (original placeholders/previous assignments)
        # made class variable: :param betas: ParameterVector (or list of Parameter) for the mixer-layer angles β[0..p-1] (original placeholders/previous assignments)
        :param gamma_values: list of float parameter values for the edge interaction gates
        :param beta_values: list of float parameter values for the individual node X rotation gates
        :return: Qiskit QuantumCircuit object (i.e. parameterized version of the passed QAOA circuit (:param qc), to be used in place of the passed circuit)
        """
        if len(gamma_values) != len(self.gammas) or len(beta_values) != len(self.betas):
            raise ValueError(
                f"Length mismatch: len(gamma_values)={len(gamma_values)} vs {len(self.gammas)}, "
                f"len(beta_values)={len(beta_values)} vs {len(self.betas)}"
            )

        bind_map = {self.gammas[i]: float(gamma_values[i]) for i in range(len(self.gammas))}
        bind_map.update({self.betas[i]: float(beta_values[i]) for i in range(len(self.betas))})

        #self.qc_no_params = self.qc.copy()
        self.qc = self.qc_no_params.assign_parameters(bind_map, inplace=False)

    @staticmethod
    def qaoa_compute_energy(product_states, edges, weights=None, params = (1, 1, 1)):
        """
        This method computes the Hamiltonian expectation from two-qubit edge marginals.

        :param product_states: mapping of edge tuples (i, j) to 4x4 reduced density matrices
        :param edges: edge list used to evaluate the Hamiltonian
        :param weights: optional edge weights; if None, weights default to 1 per edge
        :param params: tuple (a, b, c) with coefficients for XX, YY, and ZZ terms
        :return: complex energy expectation value for the full edge Hamiltonian
        """
        # H_map = np.zeros((len(edges), len(edges)), dtype=complex)
        weights = weights if weights is not None else np.ones(len(edges))

        a, b, c = params

        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        energy = 0.0
        for (i, j), w in zip(edges, weights):
            H = (1/(1 + a+b+c) * w *
                 (np.kron(I, I) - a * np.kron(X, X) - b * np.kron(Y, Y) - c * np.kron(Z, Z)))
            #p = np.kron(product_states[i], product_states[j])
            l = product_states[(i, j)]
            energy += np.trace(H @ l)

        return energy

    def build_qaoa_maxcut_circuit(self, add_measurements=True):
        """
        # made class variable: :param n: the size of the circuit
        # made class variable: :param edges: the list of edges
        # made class variable: :param num_layers: the number of iterations p (i.e. rounds of alternating application of Cost and Mixer Unitaries); directly relates to gate complexity, being (n + #edges)*num_layers
        # made class variable: :param weights: list of weights for each edge. If not provided we assume unweighted, i.e. equal weights of 1.0 for all edges
        :param add_measurements: Per default measurements are added to the circuit at the end of the circuit. This can be overridden by providing a boolean "False" for this parameter
        :return: returns the quantum circuit and the used paramter vectors TODO consider adding parameters as another argument
        """
        """
        Sources:
        Farhi et al. QAOA for MaxCut
        Qiskit Documentation (https://quantum.cloud.ibm.com/docs/de/api/qiskit/qiskit.circuit.library.RXGate)
        """
        assert self.edges is not None and self.n is not None and self.p is not None
        assert self.n == len({i for k in self.edges for i in k})
        if self.weights is None:
            self.weights = [1.0] * len(self.edges)
        assert len(self.weights) == len(self.edges)

        # Placeholder parameter vectors (angles)
        self.gammas = ParameterVector("γ", self.p)
        self.betas  = ParameterVector("β", self.p)

        self.qc = QuantumCircuit(self.n, self.n if add_measurements else 0)

        # Initialise in equal superposition
        # TODO add warm start
        #qc.h(range(n))
        if self.initial_state is not None:
            _initial_state = np.asarray(self.initial_state, dtype=complex)
            assert _initial_state.shape == (2**self.n,)
            assert _initial_state.ndim == 1
            norm = np.linalg.norm(_initial_state)
            assert norm > 0
            _initial_state /= norm
            self.qc.initialize(_initial_state, range(self.n))
            print("Initial state injected as warm start")
        elif self.self_init_linegraph:
            prepare_line_singlet_circuit(self.qc, self.n)
            print("Line graph state preparation: Singlet injection")
        else:
            self.qc.h(range(self.n)) # Default: equal superposition
            print("Default QAOA state preparation: Equal superposition")

        for layer in range(self.p):
            gamma = self.gammas[layer]#
            beta  = self.betas[layer]

            # add Cost Hamiltonian for all edges, taking into account their respective weights
            for (j, k), w in zip(self.edges, self.weights):
                a, b, c = 1, 1, 1 # TODO receive from actual parameter dictionary, not static
                w = w/ (1 + a + b + c)
                self.qc.rzz(-2 * gamma * w , j, k) # z_j z_k, i.e. z interaction term between qubtis j and k
                self.qc.rxx(-2 * gamma * w, j, k) #TODO add parameter settings that decide whether the hamiltonian is quantum or classical
                self.qc.ryy(-2 * gamma * w, j, k) #TODO same as above
                # The factor 2 accomodates for qiskits default weighting of /2 for .rzz, .rxx, .ryy

            # add Mixer Hamiltionian for all nodes
            self.qc.rx(2 * beta, range(self.n))

        if add_measurements:
            self.qc.measure(range(self.n), range(self.n))

        self.qc_no_params = self.qc.copy()

        return self.qc, self.gammas, self.betas # TODO Consider removing since all returned params are now class variables

    def run_circuit(
            self,
            shots: int = 1024,
            seed: int | None = None,
            return_statevector: bool = False
    ):
        # TODO rewrite for clean code; seperate classical result return from quantum result return (currently via param :return_statevector)
        """Runs the circuit on the specified backend and returns the results. A quantum circuit needs to be passed. All other parameters are optional."""
        assert self.qc is not None
        assert self.gammas is not None and self.betas is not None
        if return_statevector:
            qc_bound = self.qc.remove_final_measurements(inplace=False)
            return Statevector.from_instruction(qc_bound)

        tqc = transpile(self.qc, backend=self.backend, optimization_level=1)

        run_args = {"shots": shots}
        if seed is not None:
            run_args["seed_simulator"] = int(seed)

        result = self.backend.run(tqc, **run_args).result()

        product_states: dict[tuple[int, int], np.ndarray] = {}
        total_energy = 0
        for (i, j), w in zip(self.edges, self.weights):
            res = self.two_qubit_marginal(psi=result, n=self.n, i=0, j=1)
            product_states[(i, j)] = res
        total_energy = QAOA.qaoa_compute_energy(product_states, edges, weights).real

        return result.get_counts(), total_energy

    @staticmethod
    def two_qubit_marginal(psi, n, i, j):
        """
        This method returns the reduced density matrix of qubits i and j.

        :param psi: state representation accepted by DensityMatrix
        :param n: total number of qubits
        :param i: first qubit index
        :param j: second qubit index
        :return: 4x4 numpy array of the two-qubit reduced density matrix
        """
        rho = DensityMatrix(psi)
        trace_out = [q for q in range(n) if q not in (i, j)]
        return partial_trace(rho, trace_out).data   # returns 4x4 np.array


# Example usage:
if __name__ == "__main__":
    """QAOA circuit test execution (not SDP)"""
    use_measurements = False
    edges = [(0,1), (1,2)] #, (2, 3), (3, 4), (4, 5), (5, 6)]  # ring
    weights = [1.0] * len(edges)
    set_of_nodes = {i for k in edges for i in k}
    n = len(set_of_nodes)
    print(set_of_nodes, n)
    p = 20

    QAOA = QAOACircuit(n, p, edges, weights)

    counts_higher_energy, counts_lower_energy = 0, 0
    def benchmark():
        gamma_values, beta_values = set_random_params(p)
        print(gamma_values, beta_values)


        def test(qc, gammas, betas):
            QAOA.bind_circuit_parameters(gamma_values=gamma_values, beta_values=beta_values)
            assert QAOA.qc.num_parameters == 0
            results, _ = QAOA.run_circuit(shots=1024, return_statevector=not use_measurements)

            if not use_measurements:
                product_states: dict[tuple[int, int], np.ndarray] = {}
                total_energy = 0
                for (i, j), w in zip(edges, weights):
                    p = QAOA.two_qubit_marginal(results, n, 0, 1)
                    product_states[(i, j)] = p
                total_energy = QAOA.qaoa_compute_energy(product_states, edges, weights).real
                print(f"Energy: {total_energy}")
                return total_energy
            else:
                sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
                print(sorted_results[:10])
                return None

        QAOA.self_init_linegraph = True
        qc_, gammas_, betas_ = QAOA.build_qaoa_maxcut_circuit()
        print(qc_.draw("text"))
        """Test on equal superposition:"""
        energy_injected = test(qc_, gammas_, betas_)

        QAOA.self_init_linegraph = False
        qc_, gammas_, betas_ = QAOA.build_qaoa_maxcut_circuit()
        print(qc_.draw("text"))
        """Test on random initial state:"""
        energy_equal_superpos = test(qc_, gammas_, betas_)

        print(f"Energy injected: {energy_injected}")
        print(f"Energy equal superposition: {energy_equal_superpos}")
        print(f"Singlet injection yields higher energy: {energy_injected > energy_equal_superpos}")
        return energy_injected > energy_equal_superpos

    for _ in range(1):
        higher_energy = benchmark()
        if higher_energy:
            counts_higher_energy += 1
        else:
            counts_lower_energy += 1

    print(f"Counts higher energy: {counts_higher_energy} vs. {counts_lower_energy} lower energy")
    print(f"Ratio: {counts_higher_energy / (counts_higher_energy + counts_lower_energy)}")