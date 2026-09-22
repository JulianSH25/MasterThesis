"""Prepare specialised initial states used by optional QAOA experiments.

The main QAOA pipeline normally starts from an equal superposition or an SDP
warm-start state.  This module holds the separate line-graph singlet initialiser.
"""

from numpy.f2py.auxfuncs import throw_error
from qiskit import QuantumCircuit


def prepare_line_singlet_circuit(
    qc: QuantumCircuit,
    n: int,
    start_index: int = 1,
    leftover_mode: str = "plus",
) -> None:
    """
    Prepare singlets on every second edge of a line graph.

    Qubits represent nodes, so an edge (i, i+1) is the qubit pair (i, i+1).
    This prepares singlets on a disjoint matching:
      - start_index=0 -> (0,1), (2,3), (4,5), ...
      - start_index=1 -> (1,2), (3,4), (5,6), ...

    Leftover qubits are initialised according to ``leftover_mode``.  The
    currently supported choice is the equal-superposition state ``|+>``.

    Args:
        qc: Circuit modified in place.
        n: Number of graph vertices and qubits.
        start_index: Matching offset, selecting either even or odd line edges.
        leftover_mode: Initialisation applied to unmatched qubits.

    Raises:
        ValueError: If the offset or leftover-state mode is unsupported.
    """
    if start_index not in (0, 1):
        raise ValueError("start_index must be 0 or 1")

    if leftover_mode not in ("plus", "zero", "one"):
        raise ValueError("leftover_mode must be 'plus', 'zero', or 'one'")

    # Track which qubits are already used in singlets
    used = [False] * n

    # Prepare singlets on every second edge
    for i in range(start_index, n - 1, 2):
        # Prepare singlet state: |psi^-> = (|01> - |10>) / sqrt(2) on qubits (i, i+1)
        qc.x(i + 1)
        qc.h(i)
        qc.cx(i, i + 1)
        qc.z(i)

        used[i] = True
        used[i + 1] = True

    # Initialize leftover qubit
    for q in range(n):
        if not used[q]:
            if leftover_mode == "plus":
                qc.h(q)
            else:
                throw_error(f"Unimplemented leftover_mode: {leftover_mode}. Currently available: 'plus' (|+>).")
