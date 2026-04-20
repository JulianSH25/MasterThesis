from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, SparsePauliOp

from StatePrep import prepare_line_singlet_circuit
from utils import set_random_params, get_benchmark_params
import numpy as np
from collections import defaultdict

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
        self.warm_start_correlations = None
        self.self_init_linegraph = False
        self.params = None
        self.cost_operator: SparsePauliOp | None = None
        benchm_params = get_benchmark_params()
        self.start_index = benchm_params['start_index_singlet']
        self.debug = benchm_params['debug']
        self.initial_ws_energy = None

    def _pauli_label_for_edge(self, i: int, j: int, pauli: str) -> str:
        """
        Builds a full-length Pauli label for a 2-local term on qubits i and j.

        Qiskit Pauli labels are big-endian strings where the right-most
        character corresponds to qubit 0.
        """
        label = ["I"] * self.n
        label[self.n - 1 - i] = pauli
        label[self.n - 1 - j] = pauli
        return "".join(label)

    def build_cost_operator(self) -> None:
        """
        Precomputes the weighted QAOA cost Hamiltonian as a SparsePauliOp.

        This operator is reused across all objective evaluations and avoids
        repeated per-edge density-matrix/partial-trace work.
        """
        assert self.params is not None
        assert self.edges is not None
        assert self.weights is not None

        a, b, c = self.params
        norm = 1.0 / float(1 + a + b + c)

        coeffs: dict[str, complex] = defaultdict(complex)
        identity = "I" * self.n

        for (i, j), w in zip(self.edges, self.weights):
            w_norm = norm * float(w)
            coeffs[identity] += w_norm
            if a != 0:
                coeffs[self._pauli_label_for_edge(i, j, "X")] += -w_norm * float(a)
            if b != 0:
                coeffs[self._pauli_label_for_edge(i, j, "Y")] += -w_norm * float(b)
            if c != 0:
                coeffs[self._pauli_label_for_edge(i, j, "Z")] += -w_norm * float(c)

        pauli_terms = [(label, coeff) for label, coeff in coeffs.items() if abs(coeff) > 0]
        self.cost_operator = SparsePauliOp.from_list(pauli_terms)

    def compute_energy_from_statevector(self, statevec: Statevector) -> float:
        """
        Compute cost expectation in a single operator expectation call.
        """
        if self.cost_operator is None:
            self.build_cost_operator()
        value = statevec.expectation_value(self.cost_operator)
        return float(np.real(value))

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
    def qaoa_compute_energy(product_states, edges, weights=None, params = None):
        """
        This method computes the Hamiltonian expectation from two-qubit edge marginals.

        :param product_states: mapping of edge tuples (i, j) to 4x4 reduced density matrices
        :param edges: edge list used to evaluate the Hamiltonian
        :param weights: optional edge weights; if None, weights default to 1 per edge
        :param params: tuple (a, b, c) with coefficients for XX, YY, and ZZ terms
        :return: complex energy expectation value for the full edge Hamiltonian
        """
        # H_map = np.zeros((len(edges), len(edges)), dtype=complex)
        assert params is not None
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

    def apply_warm_start_correlations(self):
        for (i, j) in self.edges:
            c_ij = self.warm_start_correlations[(i, j)]

            c_ij = float(np.clip(c_ij, -3.0, 3.0))
            x = np.pi * (c_ij - 3.0) / 6.0

            print(f"Applying warm start correlation {c_ij} on edge ({i}, {j}) with rotation angle {x:.4f} radians")

            self.qc.rxx(-2*x, i, j)
            self.qc.ryy(-2*x, i, j)
            self.qc.rzz(-2*x, i, j)

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
        assert self.edges is not None and self.n is not None and self.p is not None and self.params is not None
        assert self.n == len({i for k in self.edges for i in k})
        if self.weights is None:
            self.weights = [1.0] * len(self.edges)
        assert len(self.weights) == len(self.edges)

        # Placeholder parameter vectors (angles)
        self.gammas = ParameterVector("γ", self.p)
        self.betas  = ParameterVector("β", self.p)

        self.qc = QuantumCircuit(self.n, self.n if add_measurements else 0)

        # Initialise in equal superposition
        #qc.h(range(n))
        if self.initial_state is not None:
            _initial_state = np.asarray(self.initial_state, dtype=complex)
            assert _initial_state.shape == (2**self.n,)
            assert _initial_state.ndim == 1
            # obsolete?
            norm = np.linalg.norm(_initial_state)
            assert norm > 0
            _initial_state /= norm
            self.qc.initialize(_initial_state, range(self.n))
            print("Initial state injected as warm start")
            E_initial = self.compute_energy_from_statevector(Statevector(_initial_state))
            self.initial_ws_energy = E_initial
            print(f"Initial state energy for vector-normalised SDP solution used as warm start: {E_initial}")

            # TODO: add warm start correlations here
            # TODO WARNING: REMOVE WHEN COMPARING AGAINST 010101... INITIAL STATE!
            if self.warm_start_correlations is not None:
                self.apply_warm_start_correlations()
                print("Warm start correlations applied in the form of weak entanglement")
        elif self.self_init_linegraph:
            assert self.start_index is not None and isinstance(self.start_index, int) and self.start_index in {0, 1}
            print("Line graph state preparation: Singlet injection")
            print(f"Singlets induced on ODD parity edges") if self.start_index == 0 else print("Singlets induced on EVEN parity edges")
            prepare_line_singlet_circuit(self.qc, self.n, start_index=self.start_index)
            #self.initial_ws_energy = self.compute_energy_from_statevector(Statevector(self.qc.draw(output='statevector')))
        else:
            self.qc.h(range(self.n)) # Default: equal superposition
            print("Default QAOA state preparation: Equal superposition")

        for layer in range(self.p):
            gamma = self.gammas[layer]#
            beta  = self.betas[layer]

            # add Cost Hamiltonian for all edges, taking into account their respective weights
            a, b, c = self.params
            for (j, k), w in zip(self.edges, self.weights):
                w = w/ (1 + a + b + c)
                self.qc.rxx(-2 * gamma * w * a, j, k)
                self.qc.ryy(-2 * gamma * w * b, j, k)
                self.qc.rzz(-2 * gamma * w * c, j, k)  # z_j z_k, i.e. z interaction term between qubtis j and k
                # The factor 2 accomodates for qiskits default weighting of /2 for .rzz, .rxx, .ryy

            # add Mixer Hamiltionian for all nodes
            self.qc.rx(2 * beta, range(self.n))
            self.qc.rz(2 * beta, range(self.n)) # NOTE experimental
            self.qc.ry(2 * beta, range(self.n)) # NOTE experimental
            print(f"Layer {layer}: Added Cost and Mixer unitaries with gamma={gamma} and beta={beta}")
            print(".rz and .ry mixer terms added in addition to classical .rx mixer") # TODO remove print if rz, ry not used!

        if add_measurements:
            self.qc.measure(range(self.n), range(self.n))

        self.qc_no_params = self.qc.copy()
        self.build_cost_operator()

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
        total_energy = QAOACircuit.qaoa_compute_energy(product_states, self.edges, self.weights, params=self.params).real

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
                total_energy = QAOA.qaoa_compute_energy(product_states, edges, weights, params=self.params).real
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