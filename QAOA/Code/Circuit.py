"""Build and evaluate the parameterised QAOA circuits used by benchmarks.

``Main.py`` instantiates :class:`QAOACircuit` after any SDP preprocessing. The
class owns circuit construction for standard QAOA and HamQAOA, applies the
chosen warm-start mode, and exposes exact statevector-energy evaluation to the
classical optimiser.
"""

from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, SparsePauliOp
from scipy.stats import contingency
from math import pi
import csv

from StatePrep import prepare_line_singlet_circuit
from Utils import set_random_params, get_benchmark_params
import numpy as np
import sys, uuid, time
import warnings
from pathlib import Path
from collections import defaultdict

use_measurements = False

class QAOACircuit(QuantumCircuit):
    """QAOA circuit plus the metadata needed to bind and evaluate it.

    Args:
        n: Number of graph vertices and circuit qubits.
        p: QAOA circuit depth.
        edges: Undirected graph edges indexed by their qubits.
        weights: Edge weights aligned with ``edges``.

    The construction respects the current benchmark configuration, including
    the circuit ansatz and optional SDP-derived state, correlations, or
    King-inspired rotation data.

    README:
    This class is responsible for Circuit building, execution, and result processing/evaluation. The main functionality is stretched over 2 marked 'sections':
    1) Circuit Building: This section includes methods for constructing the QAOA circuit based on the provided graph instance, parameters, and warm start configuration. It handles the creation of the quantum circuit with the appropriate gates and parameterization according to the specified QAOA variant (e.g., standard, HAMQAOA) and warm start mode.
    2) Circuit Execution and Result Processing: This section includes methods for running the quantum circuit on the specified backend, processing the measurement results or statevector outputs, and computing the energy of the resulting states with respect to the cost Hamiltonian. It provides functionality for evaluating the performance of the QAOA circuit based on the obtained results.
        Here at every rerun also the rebinding of the optimised QAOA parameters taskes place.
    AFTER: only some helper methods
    """
    def __init__(self, n, p, edges, weights):
        """
        Initialise QAOA circuit metadata and backend configuration.

        Args:
            n: Number of nodes and qubits.
            p: Circuit depth.
            edges: Graph edges as qubit-index pairs.
            weights: Edge weights aligned with ``edges``.
        """
        self.uid = str(uuid.uuid4())[:8]
        self.n = n # number of nodes
        self.p = p # Circuit depth; number of layers
        self.edges = edges
        self.weights = weights
        benchm_params = get_benchmark_params()
        raw_circuit_type = str(benchm_params.get('circuit_type') or 'standard').lower()
        # HOG selects the graph source, not a different circuit ansatz.
        # The corresponding QAOA circuit is still the standard ansatz.
        self.circuit_type = 'standard' if raw_circuit_type == 'hog' else raw_circuit_type
        self.no_param_types = None
        self.qaoa_parameters: list[np.ndarray[float]] = [] # e.g. [gammas, betas] -> this implementation enables to have more parameters than just fixed gamma and beta
        self.param_ranges: list[tuple] | tuple | None = None
        self.qc: QuantumCircuit = None
        self.qc_no_params: QuantumCircuit = None # Auxiliary variable that is used to update the quantum circuit parameters
        self.backend = Aer.get_backend('qasm_simulator')
        self.initial_state = None
        self.classical_WS_cut = None
        self.WS_ENERGY = None
        self.warm_start_correlations = None
        self.warm_start_king_data = None
        self.self_init_linegraph = False
        self.params = None
        self.cost_operator: SparsePauliOp | None = None
        self.start_index = benchm_params.get('start_index_singlet')
        self.warm_start_flag = bool(benchm_params.get("warm_start") or False)
        self.warm_start_mode = str(benchm_params.get("warm_start_mode") or "standard").lower()
        self.warm_start_corr_strength = float(benchm_params.get("warm_start_corr_strength") or 1.0)
        self.warm_start_corr_repeats = int(benchm_params.get("warm_start_corr_repeats") or 1)
        self.debug = bool(benchm_params.get('debug') or False)
        self.log_qc_svg = bool(benchm_params.get('save_circuit_svg') or False)
        self.debug_path = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
        self.result_name_suffix = str(benchm_params.get("result_name_suffix") or "")
        if self.result_name_suffix and not all(
            char.isascii() and (char.isalnum() or char in "._-")
            for char in self.result_name_suffix
        ):
            raise ValueError(
                "result_name_suffix may contain only letters, numbers, dots, underscores, and hyphens."
            )
        if self.debug_path and self.result_name_suffix and not Path(self.debug_path).name.endswith(self.result_name_suffix):
            self.debug_path = f"{self.debug_path}{self.result_name_suffix}"
        self.debug_run_counter = 0
        self.debug_previous_result: float | None = None
        self.parameter_log_path: Path | None = None # NOTE for debugging only
        self.initial_ws_energy = None
        raw_lasserre_level = benchm_params.get("lasserre_level")
        self.lasserre_level = int(raw_lasserre_level) if raw_lasserre_level is not None else None

        print(f"Solving for uuid: {self.uid}")

        circuit_type = self.circuit_type
        if circuit_type == 'standard':
            self.no_param_types = 2
            self.param_ranges = [(0, 2*pi), (0, pi)]
        elif circuit_type == 'hamqaoa':
            self.no_param_types = 4
            self.param_ranges = [(-pi, pi) for _ in range(self.no_param_types)]
        elif benchm_params.get('no_param_types') is not None:
            self.no_param_types = int(benchm_params['no_param_types'])

    """SECTION 1: Circuit Building"""
    def build_qaoa_maxcut_circuit(self, add_measurements=True):
        """Construct the configured parameterised QAOA circuit.

        The method prepares the selected circuit input, adds ``p`` ansatz layers,
        optionally appends measurements, and saves an unbound copy for later
        parameter binding.

        Args:
            add_measurements: Append computational-basis measurements when true.

        Returns:
            The built Qiskit circuit and its parameter-vector groups.

        Raises:
            ValueError: If the ansatz or number of parameter groups is invalid.
        """
        assert self.edges is not None and self.n is not None and self.p is not None and self.params is not None
        assert self.n == len({i for k in self.edges for i in k})
        if self.weights is None:
            self.weights = [1.0] * len(self.edges)
        assert len(self.weights) == len(self.edges)

        # Placeholder parameter vectors (angles)
        no_param_types = int(self.no_param_types) if self.no_param_types is not None else 2
        if no_param_types < 2:
            raise ValueError(f"no_param_types must be >= 2 for this circuit, got {no_param_types}")

        param_names = ["γ", "β"] + [f"θ{k + 3}" for k in range(no_param_types - 2)]
        self.qaoa_parameters = [ParameterVector(name, self.p) for name in param_names]

        self.qc = QuantumCircuit(self.n, self.n if add_measurements else 0)

        # Initialise using configured warm-start mode
        if self.self_init_linegraph or self.warm_start_flag:
            self.apply_warm_start()
        else:
            self.qc.h(range(self.n))
            print("No warm start: Default initial state is equal superposition")

        if self.params is None:
            self.params = [1, 1, 1]
        a, b, c = self.params
        king_strength, king_repeats = self._resolve_correlation_settings(self.warm_start_mode)
        if self.circuit_type == 'standard':
            print("Building standard QAOA circuit with 2 parameter types (γ and β)")
            for layer in range(self.p):
                gamma = self.qaoa_parameters[0][layer]
                beta = self.qaoa_parameters[1][layer]

                # add Cost Hamiltonian for all edges, taking into account their respective weights
                for (j, k), w in zip(self.edges, self.weights):
                    scale = 0.5 if self.lasserre_level == 2 else 1.0 / float(1 + a + b + c)
                    w = w * scale
                    # NOTE multiplying all parameters by 2 since qiskit's rxx, ryy, rzz gates apply a rotation of theta/2 for an input angle theta; i.e. we undo the default 1/2 division to allow full parameter range!
                    self.qc.rxx(2*gamma, j, k)
                    self.qc.ryy(2*gamma, j, k)
                    self.qc.rzz(2*gamma, j, k)
                    # BUG this is a temporary change to test for possible bugs.
                    # NOTE the above temporary notation does NOT accomodate for weighted instances
                    """self.qc.rxx(-2 * gamma * w * a, j, k)
                    self.qc.ryy(-2 * gamma * w * b, j, k)
                    self.qc.rzz(-2 * gamma * w * c, j, k) """ # z_j z_k, i.e. z interaction term between qubtis j and k
                    # The factor 2 accomodates for qiskits default weighting of /2 for .rzz, .rxx, .ryy

                if self.warm_start_mode in {"amplified_king", "entangled_king"}:
                    self._apply_king_warm_start_layer(
                        strength=king_strength,
                        repeats=king_repeats,
                    )

                # add Mixer Hamiltionian for all nodes
                self.qc.rx(2 * beta, range(self.n))
                self.qc.rz(2 * beta, range(self.n)) # NOTE experimental
                self.qc.ry(2 * beta, range(self.n)) # NOTE experimental
                print(f"Layer {layer}: Added Cost and Mixer unitaries with gamma={gamma} and beta={beta}") if self.debug else None
                print(".rz and .ry mixer terms added in addition to classical .rx mixer") # TODO remove print if rz, ry not used!
        elif self.circuit_type == 'hamqaoa':
            for layer in range(self.p):
                a, b, c, d = self.qaoa_parameters[0][layer], self.qaoa_parameters[1][layer], self.qaoa_parameters[2][layer], self.qaoa_parameters[3][layer]
                for (j, k), w in zip(self.edges, self.weights):
                    self.qc.rzz(2*a, j, k) # Gate 1 of HAMQAOA layer: ZZ interaction term with parameter a

                if self.warm_start_mode in {"amplified_king", "entangled_king"}:
                    self._apply_king_warm_start_layer(
                        strength=king_strength,
                        repeats=king_repeats,
                    )

                self.qc.rx(2*b, range(self.n)) # Gate 2 of HAMQAOA layer: X mixer term with parameter b
                self.qc.rz(2*c, range(self.n)) # Gate 3 of HAMQAOA layer: Z mixer term with parameter c

                cut_values = self.classical_WS_cut
                if cut_values is None:
                    warnings.warn(
                        "No classical warm-start cut provided for HAMQAOA; using random cut values for fourth HAMQAOA gate. "
                        "This may lead to suboptimal performance and may be unintentional if a classical warm start was intended."
                    )
                    cut_values = np.random.choice([-1, 1], size=self.n)
                    if self.debug:
                        print("No classical warm-start cut; using random cut for fourth HAMQAOA gate.")

                for i in range(self.n):
                    print(f"Applying fourth gate of HAMQAOA layer {layer} on qubit {i} with parameter {d} and classical warm start cut value {cut_values[i]}") if self.debug else None
                    self.qc.rz(cut_values[i] * 2*d, i) # Gate 4 of HAMQAOA layer: Z rotation with parameter d and classical warm-start cut value as rotation direction
        else:
            raise ValueError(f"Unsupported circuit type: {self.circuit_type} in build_qaoa_maxcut_circuit()")

        if add_measurements:
            self.qc.measure(range(self.n), range(self.n))

        self.qc_no_params = self.qc.copy()
        self.build_cost_operator()

        if self.debug:
            self._print_circuit(name_addition=f"initial_({self.debug_run_counter})", print_to_log=True)

        return self.qc, self.qaoa_parameters

    def _apply_warm_start_correlations_with_strength(self, strength: float = 1.0, repeats: int = 1) -> None:
        # Helper for apply_warm_start()
        """
        @param strength: scaling factor for the correlation-based gate angles, allowing for amplification or attenuation of the influence of the warm start correlations on the circuit; a strength > 1 amplifies the correlations, while a strength < 1 attenuates them.
        @param repeats: number of times to apply the correlation-based gates, allowing for iterative strengthening of the influence of the warm start correlations on the circuit; more repeats can enhance the effect of the correlations but may also increase circuit depth and noise sensitivity.
        """
        # This method applies correlation-based gates to the quantum circuit based on the provided warm start correlations, with a specified strength and number of repetitions to control the influence of the correlations on the circuit.
        repeats = max(1, int(repeats))
        for _ in range(repeats):
            for (i, j) in self.edges:
                c_ij = self.warm_start_correlations[(i, j)]
                c_ij = float(np.clip(c_ij, -3.0, 3.0))
                """CORE STEP ⤴: the weighting parameter c_ij is the correlation extracted from the SDP solution for edge (i, j), which is then transformed into a rotation angle x for the entangling gates. 
                The transformation maps the correlation range [-3, 3] to a rotation range of approximately [-pi, pi], with the strength parameter allowing for amplification or attenuation of the correlations as needed."""
                # i.e. the 'likelihood/confidence of the SDP cutting the edge (i,j)=(j,i)'

                x = strength * np.pi * (c_ij - 3.0) / 6.0

                print(f"Applying warm start correlation {c_ij} on edge ({i}, {j}) with rotation angle {x:.4f} radians") if self.debug else None

                # Applying correlation weighted gates
                self.qc.rxx(-2*x, i, j)
                self.qc.ryy(-2*x, i, j)
                self.qc.rzz(-2*x, i, j)

    @staticmethod
    def _restore_json_complex(value):
        """Recursively restore complex values encoded in cached JSON data.

        Args:
            value: JSON-safe cache value.

        Returns:
            Equivalent nested data with complex numbers restored.
        """
        if isinstance(value, dict) and set(value.keys()) == {"real", "imag"}:
            return complex(value["real"], value["imag"])
        if isinstance(value, list):
            return [QAOACircuit._restore_json_complex(item) for item in value]
        return value

    @staticmethod
    def _edge_dict_value(mapping: dict, edge: tuple[int, int], name: str):
        """Read edge data while accepting either orientation and cache key format.

        Args:
            mapping: Cached edge-keyed data.
            edge: Undirected edge to retrieve.
            name: Field name included in error messages.

        Returns:
            Stored data for ``edge``.

        Raises:
            RuntimeError: If no equivalent edge key is present.
        """
        i, j = edge
        for key in (edge, (j, i), str(edge), str((j, i)), f"{i},{j}", f"{j},{i}"):
            if key in mapping:
                return mapping[key]
        raise RuntimeError(f"Missing King {name} for edge {edge}")

    def _apply_king_product_state(self) -> None:
        """Initialise each qubit from cached Level-2 product-state vectors.

        Raises:
            RuntimeError: If required King data is absent or malformed.
        """
        if self.warm_start_king_data is None:
            raise RuntimeError("King warm start requires warm_start_king_data")

        product_state_vectors = self.warm_start_king_data.get("product_state_vectors")
        if product_state_vectors is None:
            raise RuntimeError("King warm start data does not contain product_state_vectors")
        if len(product_state_vectors) != self.n:
            raise RuntimeError(
                f"Expected {self.n} King product state vectors, got {len(product_state_vectors)}"
            )

        for qubit, vector in enumerate(product_state_vectors):
            state = np.asarray(self._restore_json_complex(vector), dtype=complex)
            if state.shape != (2,):
                raise RuntimeError(f"King product state for qubit {qubit} has shape {state.shape}, expected (2,)")
            norm = np.linalg.norm(state)
            if norm <= 0:
                raise RuntimeError(f"King product state for qubit {qubit} has zero norm")
            self.qc.initialize(state / norm, [qubit])

    def _apply_king_warm_start_layer(self, strength: float = 1.0, repeats: int = 1) -> None:
        """Append the fixed Level-2 King-inspired two-qubit rotation layer.

        Args:
            strength: Multiplier applied to each cached edge rotation angle.
            repeats: Number of times to repeat the full edge layer.

        Raises:
            RuntimeError: If cached rotation data is incomplete or invalid.
        """
        if self.warm_start_king_data is None:
            raise RuntimeError("King warm-start rotations require warm_start_king_data")

        theta_dict = self.warm_start_king_data.get("theta_dict")
        epsilon_dict = self.warm_start_king_data.get("epsilon_dict")
        n_vectors = self.warm_start_king_data.get("n_vectors")
        if theta_dict is None or epsilon_dict is None:
            raise RuntimeError("King warm-start data requires theta_dict and epsilon_dict")

        if n_vectors is None and self.warm_start_king_data.get("P_matrices") is not None:
            n_vectors = []
            for qubit, matrix in enumerate(self.warm_start_king_data["P_matrices"]):
                matrix = np.asarray(self._restore_json_complex(matrix), dtype=complex)
                if matrix.shape != (2, 2):
                    raise RuntimeError(f"King P_matrix for qubit {qubit} has shape {matrix.shape}, expected (2, 2)")
                n_vectors.append(
                    np.array(
                        [
                            float(np.real(matrix[0, 1])),
                            float(-np.imag(matrix[0, 1])),
                            float(np.real(matrix[0, 0])),
                        ]
                    )
                )

        if n_vectors is None and self.warm_start_king_data.get("product_state_vectors") is not None:
            n_vectors = []
            for qubit, state in enumerate(self.warm_start_king_data["product_state_vectors"]):
                state = np.asarray(self._restore_json_complex(state), dtype=complex)
                if state.shape != (2,):
                    raise RuntimeError(f"King product state for qubit {qubit} has shape {state.shape}, expected (2,)")
                norm = np.linalg.norm(state)
                if norm <= 0:
                    raise RuntimeError(f"King product state for qubit {qubit} has zero norm")
                state = state / norm
                alpha, beta = state
                bloch = np.array(
                    [
                        2.0 * np.real(np.conj(alpha) * beta),
                        2.0 * np.imag(np.conj(alpha) * beta),
                        abs(alpha) ** 2 - abs(beta) ** 2,
                    ],
                    dtype=float,
                )
                bloch_norm = np.linalg.norm(bloch)
                if bloch_norm <= 1e-10:
                    raise RuntimeError(f"King product state for qubit {qubit} has near-zero Bloch vector")
                bloch = bloch / bloch_norm
                reference = np.array([1.0, 0.0, 0.0]) if abs(bloch[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = np.cross(bloch, reference)
                axis_norm = np.linalg.norm(axis)
                if axis_norm <= 1e-10:
                    reference = np.array([0.0, 0.0, 1.0])
                    axis = np.cross(bloch, reference)
                    axis_norm = np.linalg.norm(axis)
                n_vectors.append(axis / axis_norm)

        if n_vectors is None:
            raise RuntimeError("King warm-start data requires n_vectors, P_matrices, or product_state_vectors")
        if len(n_vectors) != self.n:
            raise RuntimeError(f"Expected {self.n} King axis vectors, got {len(n_vectors)}")

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I4 = np.eye(4, dtype=complex)

        repeats = max(1, int(repeats))
        for _ in range(repeats):
            for i, j in self.edges:
                theta_ij = float(self._edge_dict_value(theta_dict, (i, j), "theta"))
                epsilon_ij = float(self._edge_dict_value(epsilon_dict, (i, j), "epsilon"))

                n_i = np.asarray(n_vectors[i], dtype=float)
                n_j = np.asarray(n_vectors[j], dtype=float)
                if n_i.shape != (3,) or n_j.shape != (3,):
                    raise RuntimeError(f"King axis vectors for edge {(i, j)} must have shape (3,)")

                P_i = n_i[0] * X + n_i[1] * Y + n_i[2] * Z
                P_j = n_j[0] * X + n_j[1] * Y + n_j[2] * Z

                # For qargs [i, j], Qiskit interprets the left tensor factor as qubit j.
                generator = np.kron(P_j, P_i)
                angle = strength * epsilon_ij * theta_ij
                gate = np.cos(angle) * I4 + 1j * np.sin(angle) * generator

                self.qc.unitary(gate, [i, j], label="King")

    def apply_warm_start(self) -> None:
        """Prepare the circuit input according to the selected warm-start mode.

        Statevector-only modes initialise the SDP state directly. Correlation
        modes may add an entangling layer, while King Entangled begins from the
        equal-superposition state and inserts its fixed rotations inside each
        QAOA layer.

        Raises:
            RuntimeError: If the selected mode lacks required cached data.
            ValueError: If incompatible initialisation modes are combined.
        """
        warm_mode = self._resolve_warm_start_mode() # Validate and resolve the warm start mode to determine how to apply the warm start to the circuit
        # 'standard' mode uses the initial state directly, 'amplified' mode applies correlation-based gates on top of the initial state, and 'entangled' mode starts from an equal superposition and applies correlation-based gates to induce entanglement.
        corr_strength, corr_repeats = self._resolve_correlation_settings(warm_mode) # Determine the strength and number of repetitions for applying correlation-based gates based on the warm start mode and configuration parameters

        ###
        if self.initial_state is not None and self.self_init_linegraph: # XXX Sanity check
            raise ValueError(
                "Cannot use both an initial statevector and self-initializing line graph (singlets ws) warm start simultaneously"
            )
        if warm_mode in {"amplified", "entangled"} and self.warm_start_correlations is None: # XXX Sanity check for correlation-based warm start modes
            raise RuntimeError("warm_start_correlations are required for amplified/entangled warm start modes")
        if warm_mode in {"amplified_king", "entangled_king"} and self.warm_start_king_data is None:
            raise RuntimeError("warm_start_king_data is required for amplified_king/entangled_king warm start modes")
        ###

        if warm_mode == "entangled_king":
            self.qc.h(range(self.n))
            print("Entangled King warm start: equal superposition; King rotations applied in each QAOA layer")
            self._set_initial_ws_energy("entangled King warm start")
            return
        if warm_mode == "entangled": # In 'entangled' mode, we start from an equal superposition state and apply correlation-based gates to induce entanglement, without directly using the provided initial statevector as the initial state for the circuit
            # Use equal superposition as the base state for the entangled warm start, then apply correlations to induce entanglement
            self.qc.h(range(self.n)) # STEP 1: Initialise circuit as equal superposition
            print("Entangled warm start: equal superposition + correlations")
            self._apply_warm_start_correlations_with_strength( # STEP 2: apply warm start as correlations, i.e. gates applied between any two qubits based on their corresponding warm start solution
                strength=corr_strength,
                repeats=corr_repeats,
            )
            self._set_initial_ws_energy("entangled warm start")
            return
        elif self.initial_state is not None and warm_mode in {"standard", "amplified", "amplified_king"}: # In 'standard' and 'amplified' modes, we use the provided initial statevector directly as the initial state for the circuit, and optionally apply correlation-based gates on top of it in 'amplified' mode to enhance the influence of the warm start
            # Otherwise, if an initial statevector is provided, use it directly as the initial state for the circuit
            _initial_state = np.asarray(self.initial_state, dtype=complex)
            assert _initial_state.shape == (2**self.n,) # XXX Sanity check
            assert _initial_state.ndim == 1 # XXX Sanity check
            # Normalize the initial statevector to ensure it represents a valid quantum state:
            norm = np.linalg.norm(_initial_state)
            assert norm > 0 # XXX Sanity check to avoid division by zero
            _initial_state /= norm
            # Inject the normalized initial state into the quantum circuit as the warm start:
            self.qc.initialize(_initial_state, range(self.n))
            print("Initial state injected as warm start")

            if self.warm_start_correlations is not None: # If in 'amplified' mode and correlations are provided, apply correlation-based gates on top of the initial state
                assert warm_mode in {"amplified"} # XXX Sanity check to ensure correlation-based gates are only applied in 'amplified' mode when an initial state is provided
                # Amplify the provided warm start vector by applying correlations as weak entangling gates, which can help to escape local minima and provide a stronger initial signal for the optimization, while still preserving the core structure of the provided state.
                self._apply_warm_start_correlations_with_strength(
                    strength=corr_strength,
                    repeats=corr_repeats,
                )
                print("Warm start correlations applied in the form of weak entanglement")
            self._set_initial_ws_energy("warm start")
            return
        elif self.self_init_linegraph:
            # Apply singlet injections for line graph warm start
            assert self.start_index is not None and isinstance(self.start_index, int) and self.start_index in {0, 1}
            print("Line graph state preparation: Singlet injection")
            print(f"Singlets induced on ODD parity edges") if self.start_index == 0 else print("Singlets induced on EVEN parity edges")
            prepare_line_singlet_circuit(self.qc, self.n, start_index=self.start_index)
            return
        else:
            raise RuntimeError("No valid warm start configuration found; cannot apply warm start to circuit. Check and confirm parameter settings. Warm start flag: {self.warm_start_flag}, warm start mode: {warm_mode}, initial state provided: {self.initial_state}, self_init_linegraph: {self.self_init_linegraph}")

    def build_cost_operator(self) -> None:
        """
        Precompute the weighted QMC Hamiltonian as a ``SparsePauliOp``.

        This operator is reused across all objective evaluations and avoids
        repeated per-edge density-matrix/partial-trace work. It also records
        the active Level-1 or Level-2 normalisation convention.
        """
        if self.params is None:
            self.params = [1, 1, 1]
        assert self.edges is not None
        assert self.weights is not None

        a, b, c = self.params
        norm = float(0.5) if self.lasserre_level == 2 else 1.0 / float(1 + a + b + c)

        coeffs: dict[str, complex] = defaultdict(complex)
        identity = "I" * self.n

        for (i, j), w in zip(self.edges, self.weights):
            w_norm = norm * float(w)
            coeffs[identity] += w_norm
            if a != 0:
                coeffs[self._pauli_label_for_edge(i, j, "X", self.n)] += -w_norm * float(a)
            if b != 0:
                coeffs[self._pauli_label_for_edge(i, j, "Y", self.n)] += -w_norm * float(b)
            if c != 0:
                coeffs[self._pauli_label_for_edge(i, j, "Z", self.n)] += -w_norm * float(c)

        pauli_terms = [(label, coeff) for label, coeff in coeffs.items() if abs(coeff) > 0]
        self.cost_operator = SparsePauliOp.from_list(pauli_terms)

    """SECTION 2: Circuit Execution and Result Processing"""

    def run_circuit(
            self,
            shots: int = 1024,
            seed: int | None = None,
            return_statevector: bool = False
    ):
        """Execute the currently bound circuit by exact or sampled simulation.

        Args:
            shots: Number of samples for measurement-based simulation.
            seed: Optional simulator seed for sampled execution.
            return_statevector: Return an exact statevector instead of sampled
                counts and their estimated energy.

        Returns:
            An exact statevector, or ``(counts, energy)`` in sampled mode.
        """
        assert self.qc is not None
        assert self.qaoa_parameters is not None and len(self.qaoa_parameters) > 0
        self.debug_run_counter += 1
        if return_statevector: # NOTE this is the Quantum Max Cut case
            qc_bound = self.qc.remove_final_measurements(inplace=False)
            self._print_circuit(qc_bound, name_addition=f"run_{self.debug_run_counter}") if self.debug else None
            return Statevector.from_instruction(qc_bound)
        else:
            tqc = transpile(self.qc, backend=self.backend, optimization_level=1)
            self._print_circuit(tqc, name_addition=f"run_{self.debug_run_counter}") if self.debug else None

            run_args = {"shots": shots}
            if seed is not None:
                run_args["seed_simulator"] = int(seed)

            result = self.backend.run(tqc, **run_args).result()

            product_states: dict[tuple[int, int], np.ndarray] = {}
            total_energy = 0
            for (i, j), w in zip(self.edges, self.weights):
                res = self.two_qubit_marginal(psi=result, n=self.n, i=0, j=1)
                product_states[(i, j)] = res
            total_energy = QAOACircuit.qaoa_compute_energy(product_states, self.edges, self.weights, params=self.params, lasserre_level=self.lasserre_level).real
            self.debug_previous_result = total_energy

            return result.get_counts(), total_energy

    def bind_circuit_parameters(self,parameters: list[np.ndarray]) -> None:
        """
        Bind numeric values to the parameterised QAOA circuit.

        Args:
            parameters: One numerical vector per circuit parameter group, with
                exactly ``p`` values in each group.

        Raises:
            ValueError: If parameter groups have wrong dimensions or values.
        """
        assert self.qaoa_parameters is not None and len(self.qaoa_parameters) > 0

        if len(parameters) != len(self.qaoa_parameters):
            raise ValueError(
                f"Length mismatch: len(parameters)={len(parameters)} vs len(self.qaoa_parameters)={len(self.qaoa_parameters)}"
            )

        bind_map = {}
        normalised_parameters: list[list[float]] = []

        for idx, (parameter_vector, parameter_values) in enumerate(zip(self.qaoa_parameters, parameters)):
            if len(parameter_values) != len(parameter_vector):
                raise ValueError(
                    f"Length mismatch in parameter group {idx}: len(values)={len(parameter_values)} vs len(vector)={len(parameter_vector)}"
                )

            if isinstance(self.param_ranges, list):
                if len(self.param_ranges) == 0:
                    raise ValueError("self.param_ranges must not be empty when provided as a list")
                bounds = self.param_ranges[idx] if idx < len(self.param_ranges) else self.param_ranges[-1]
            else:
                bounds = self.param_ranges
            LB, UB = bounds
            range_width = UB - LB
            if range_width <= 0:
                raise ValueError(f"Invalid parameter range ({LB}, {UB}) for group {idx}")

            normalised_values: list[float] = []
            for value in parameter_values:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite parameter value {value} in group {idx}")
                value = ((value - LB) % range_width) + LB
                normalised_values.append(value)

            bind_map.update({parameter_vector[i]: normalised_values[i] for i in range(len(parameter_vector))})
            normalised_parameters.append(normalised_values)

        #self.qc_no_params = self.qc.copy()
        self.qc = self.qc_no_params.assign_parameters(bind_map, inplace=False)
        self._append_parameter_log_row(parameters=normalised_parameters)




    """Static methods/helpers that do not require class access:"""

    # 1st option to compute energy from two-qubit marginals; this is the more general method that can be used for both statevector and measurement result inputs, as long as the appropriate two-qubit marginals are provided in the form of reduced density matrices for each edge.
    @staticmethod
    def qaoa_compute_energy(product_states, edges, weights=None, params = None, lasserre_level: int | None = None) -> float | Any:
        """
        Evaluate the QMC Hamiltonian from one two-qubit marginal per edge.

        Args:
            product_states: Mapping from edges to ``4 x 4`` density matrices.
            edges: Graph edges used for the Hamiltonian.
            weights: Optional edge weights, defaulting to one.
            params: ``(a, b, c)`` selectors for ``XX``, ``YY``, and ``ZZ``.
            lasserre_level: Normalisation convention used by the comparison.

        Returns:
            Complex expectation value of the full edge Hamiltonian.
        """
        # H_map = np.zeros((len(edges), len(edges)), dtype=complex)
        if params is None:
            params = [1, 1, 1]
        weights = weights if weights is not None else np.ones(len(edges))

        a, b, c = params

        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

        energy = 0.0
        for (i, j), w in zip(edges, weights):
            scale = 0.5 if lasserre_level == 2 else 1.0 / float(1 + a + b + c)

            H = scale * w * (
                np.kron(I, I)
                - a * np.kron(X, X)
                - b * np.kron(Y, Y)
                - c * np.kron(Z, Z)
            )
            #p = np.kron(product_states[i], product_states[j])
            l = product_states[(i, j)]
            energy += np.trace(H @ l)

        return energy
    
    # 2nd option to compute energy directly from statevector
    def compute_energy_from_statevector(self, statevec: Statevector) -> float:
        """Evaluate the cached QMC operator on an exact statevector.

        Args:
            statevec: Exact Qiskit statevector to evaluate.

        Returns:
            Real QMC energy expectation.
        """
        if self.cost_operator is None:
            self.build_cost_operator()
        value = statevec.expectation_value(self.cost_operator)
        return float(np.real(value))

    @staticmethod
    def two_qubit_marginal(psi, n, i, j):
        """
        Return the reduced density matrix on a selected qubit pair.

        Args:
            psi: State representation accepted by ``DensityMatrix``.
            n: Total number of qubits.
            i: First retained qubit index.
            j: Second retained qubit index.

        Returns:
            ``4 x 4`` two-qubit reduced density matrix.
        """
        rho = DensityMatrix(psi)
        trace_out = [q for q in range(n) if q not in (i, j)]
        return partial_trace(rho, trace_out).data   # returns 4x4 np.array
    
    @staticmethod
    def _pauli_label_for_edge(i: int, j: int, pauli: str, n: int) -> str:
        """
        Build a Qiskit big-endian Pauli label for one two-qubit term.

        Args:
            i: First qubit index.
            j: Second qubit index.
            pauli: Single-qubit label ``X``, ``Y``, or ``Z``.
            n: Total number of qubits.

        Returns:
            Length-``n`` big-endian Pauli label, where its rightmost character
            corresponds to qubit zero.
        """
        label = ["I"] * n
        label[n - 1 - i] = pauli
        label[n - 1 - j] = pauli
        return "".join(label)
    
    """Utility methods with class access:"""
    def _set_initial_ws_energy(self, label: str) -> None:
        """Evaluate and retain the energy immediately after warm-state preparation.

        Args:
            label: Human-readable preparation label used in diagnostics.
        """
        # NOTE Helper method 
        # Compute and store the initial energy of the warm-start statevector for later comparison [applicable only in modes 'standard' and 'amplified' where this vector is used as initial qc state]
        statevec = Statevector.from_instruction(self.qc)
        self.initial_ws_energy = self.compute_energy_from_statevector(statevec)
        print(f"Initial state energy for {label}: {self.initial_ws_energy}") if self.debug else None

    def _resolve_warm_start_mode(self) -> str:
        """Validate the configured warm-start mode.

        Returns:
            Canonical warm-start mode name.

        Raises:
            ValueError: If the configuration selects an unknown mode.
        """
        # NOTE Helper method [getter] with validation
        warm_mode = self.warm_start_mode
        """
        Available warm start modes (so far; last updated 05.05.2026):
        - 'standard': use the provided initial statevector directly as the warm start without additional correlation-based gates.
        - 'amplified': apply correlation-based gates ON TOP OF the initial state (i.e. an extension of 'standard' mode) with angles directly derived from the correlations, effectively amplifying the influence of the warm start on the initial state.
        - 'entangled': start from an equal superposition state and apply correlation-based gates to induce entanglement
        - 'entangled_king': prepare the GP/GW product state and apply the King Algorithm-17 rotation layer
        - 'amplified_king': use the standard warm-start state and add fixed King Algorithm-17 rotation layers inside each QAOA layer"""
        if warm_mode not in {"standard", "amplified", "entangled", "entangled_king", "amplified_king"}:
            raise ValueError(f"Unsupported warm_start_mode: {warm_mode}")
        return warm_mode

    def _resolve_correlation_settings(self, warm_mode: str) -> tuple[float, int]:
        """Select correlation-layer strength and repetition count for a mode.

        Args:
            warm_mode: Already validated warm-start mode.

        Returns:
            ``(strength, repeats)`` for the requested correlation layer.
        """
        strength = self.warm_start_corr_strength # This parameter controls how strongly the warm start correlations influence the initial state. A value of 0 means no influence (i.e., no correlation-based gates applied), while a value of 1 means full influence (i.e., gates applied with angles directly derived from the correlations). Values greater than 1 can be used to amplify the effect of the correlations, while values between 0 and 1 can be used to attenuate it, allowing for fine-tuning of the warm start's impact on the optimization landscape.
        repeats = max(1, self.warm_start_corr_repeats) # This parameter determines how many times the correlation-based gates are applied. Repeating the application of these gates can amplify their effect on the initial state, potentially helping to escape local minima and providing a stronger initial signal for the optimization.
        if warm_mode == "amplified" and strength == 1.0 and repeats == 1:
            strength = 2.0 # Default amplification for 'amplified' mode
        return strength, repeats
    
    def _print_circuit(self, circuit = None, name_addition = "", print_to_log = False) -> None:
        """Render a circuit for debug output and optionally append it to a log.

        Args:
            circuit: Qiskit circuit to render; defaults to the active circuit.
            name_addition: Unique suffix for the debug artefact filename.
            print_to_log: Whether to print the text rendering to stdout.
        """
        # TODO move and rename
        # NOTE This method is used for logging and debugging purposes to visualize the quantum circuit. It can print the circuit to the console and/or save it as an SVG file depending on the configuration 
        assert self.debug
        try:
            circuit = circuit if circuit is not None else self.qc
            #print("Quantum circuit build:")
            #print(circuit.draw()) if print_to_log else print("Circuit drawing skipped in console output due to print_to_log=False; Saving to svg file instead.")
            if self.log_qc_svg and self.debug_path:
                debug_path = Path(self.debug_path)
                debug_path.mkdir(parents=True, exist_ok=True)
                fig = circuit.draw(output="mpl", fold=1000)
                fig.savefig(debug_path / f"{self.n}_{self.p}_{len(self.edges)}_{str(uuid.uuid4())[:8]}_{name_addition}_circuit{self.result_name_suffix}.svg", bbox_inches="tight")
            else:
                #print(f"Circuit SVG saving skipped due to log_qc_svg=False or debug_path not set. [log_qc_svg={self.log_qc_svg}, debug_path={'set' if self.debug_path else 'not set'}]")
                pass
        except Exception as e:
            print(f"Logging of cirquit failed with exception: {e}")
            pass
    
    def _append_parameter_log_row(self, parameters: list[list[float]]) -> None:
        """Append the current normalised parameter groups to the debug CSV.

        Args:
            parameters: Bound QAOA parameter values grouped by gate type.
        """
        # This method appends a row of parameter values to a CSV log file for debugging purposes. It includes the current parameter values, the run counter, the previous result, and a timestamp. The log file is created in the specified debug path and is named based on the circuit configuration and a unique identifier.
        if not self.debug:
            return

        if not self.debug_path:
            print("Parameter logging skipped because debug_path is not set.")
            return

        debug_dir = Path(self.debug_path)
        debug_dir.mkdir(parents=True, exist_ok=True)

        if self.parameter_log_path is None:
            self.parameter_log_path = debug_dir / (
                f"{self.n}_{self.p}_{len(self.edges)}_{self.uid}_parameter_trace{self.result_name_suffix}.csv"
            )

        fieldnames = ["bind_index", "run_counter_at_bind", "previous_result", "timestamp"]
        for group_index, param_group in enumerate(parameters):
            if group_index == 0:
                prefix = "gamma"
            elif group_index == 1:
                prefix = "beta"
            else:
                prefix = f"theta{group_index + 1}"
            fieldnames.extend([f"{prefix}_{i + 1}" for i in range(len(param_group))])

        next_run_counter = self.debug_run_counter + 1
        row = {
            "bind_index": next_run_counter,
            "run_counter_at_bind": self.debug_run_counter,
            "previous_result": self.debug_previous_result,
            "timestamp": time.time(),
        }
        for group_index, param_group in enumerate(parameters):
            if group_index == 0:
                prefix = "gamma"
            elif group_index == 1:
                prefix = "beta"
            else:
                prefix = f"theta{group_index + 1}"
            row.update({f"{prefix}_{i + 1}": float(param_group[i]) for i in range(len(param_group))})

        write_header = not self.parameter_log_path.exists()
        with self.parameter_log_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)




# Example usage (deprecated, not for use; run QAOA > Code(i.e. this directory) > Main.py instead):
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
        """Run the deprecated interactive circuit benchmark example."""
        gamma_values, beta_values = set_random_params(p, range=(0, 2*np.pi)), set_random_params(p, range=(0, np.pi))
        print(gamma_values, beta_values)


        def test(qc):
            """Evaluate the deprecated example circuit at sampled angles.

            Args:
                qc: Retained legacy argument; the closure evaluates ``QAOA``.

            Returns:
                Energy in statevector mode, otherwise ``None``.
            """
            QAOA.bind_circuit_parameters(parameters=[gamma_values, beta_values])
            assert QAOA.qc.num_parameters == 0
            results, _ = QAOA.run_circuit(shots=1024, return_statevector=not use_measurements)

            if not use_measurements:
                product_states: dict[tuple[int, int], np.ndarray] = {}
                total_energy = 0
                for (i, j), w in zip(edges, weights):
                    p = QAOA.two_qubit_marginal(results, n, 0, 1)
                    product_states[(i, j)] = p
                total_energy = QAOA.qaoa_compute_energy(product_states, edges, weights, params=self.params, lasserre_level=self.lasserre_level).real
                print(f"Energy: {total_energy}")
                return total_energy
            else:
                sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
                print(sorted_results[:10])
                return None

        QAOA.self_init_linegraph = True
        qc_, _ = QAOA.build_qaoa_maxcut_circuit()
        print(qc_.draw("text"))
        """Test on equal superposition:"""
        energy_injected = test(qc_)

        QAOA.self_init_linegraph = False
        qc_, _ = QAOA.build_qaoa_maxcut_circuit()
        print(qc_.draw("text"))
        """Test on random initial state:"""
        energy_equal_superpos = test(qc_)

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
