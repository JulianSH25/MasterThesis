Benchmark configuration parameter reference
===========================================

Purpose
-------

This file documents the JSON parameters used by the benchmark configuration
files.

It is intended as a general-purpose reference for all benchmark configs, not only
for exact solving. Each parameter entry states:

    - expected JSON type if the parameter is used
    - known discrete options where the option set is small
    - example values where the value is open-ended
    - what the parameter controls

JSON note:
    Use null for parameters that are intentionally not used in a given benchmark
    mode. JSON uses null, not Python None.

Suggested file location:
    QAOA/Code/run_configurations/benchmark_config_parameters.txt


General benchmark control
-------------------------

optimiser
    Expected type if used: string
    Supported/known options: "exact", "adam"
    Meaning:
        Selects the optimisation/backend mode.
        "adam" uses QAOA parameter optimisation with Adam.
        "exact" uses the exact solver/eigensolver path.

optimiser_use_heuristic
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables or disables an additional heuristic optimiser step.
        This is relevant for heuristic/QAOA parameter optimisation modes, not for
        a pure exact eigensolver run.

heuristic_optimiser_iterations
    Expected type if used: integer
    Example value: 2
    Meaning:
        Number of iterations used by the heuristic optimiser.

heuristic_optimiser_sampleSize
    Expected type if used: integer
    Example value: 1000
    Meaning:
        Number of parameter samples or candidate points used by the heuristic
        optimiser.


Circuit / ansatz settings
-------------------------

circuit_type
    Expected type if used: string
    Supported/known options: "HOG"
    Meaning:
        Selects the circuit/problem family.
        For "HOG", graph instances are loaded from a House of Graphs
        adjacency-list file and n is interpreted as a graph index.

parameter_vector
    Expected type if used: array of numbers
    Example value: [1, 1, 1]
    Meaning:
        Explicit QAOA parameter vector.
        Only relevant for modes that evaluate or initialise parameterised QAOA
        circuits.

iterations_list
    Expected type if used: array of integers
    Example value: [50, 100]
    Meaning:
        List of optimiser iteration counts to benchmark.
        Relevant for iterative optimisation modes such as Adam.
        Not conceptually relevant for exact solving.

learning_rate_adam
    Expected type if used: number
    Example value: 0.05
    Meaning:
        Learning rate used by the Adam optimiser.

depth_list
    Expected type if used: array of integers
    Example value: [1, 2, 3, 4, 5]
    Meaning:
        List of QAOA circuit depths p to benchmark.
        Relevant for QAOA circuit evaluation/optimisation.
        Not conceptually relevant for exact solving.


Graph / instance settings
-------------------------

n_start
    Expected type if used: integer
    Example value: 2
    Meaning:
        Start value for graph size or graph index, depending on the benchmark
        mode.
        For generated graph families, this usually denotes the smallest graph
        size.
        For HOG configs, the launcher may overwrite this and use graph indices
        from the adjacency-list file instead.

n_end
    Expected type if used: integer
    Example value: 10
    Meaning:
        End value for graph size or graph index, depending on the benchmark mode.
        For generated graph families, this usually denotes the largest graph
        size.
        For HOG configs, the launcher may overwrite this and use graph indices
        from the adjacency-list file instead.

graph_generation_type
    Expected type if used: string
    Supported/known options: "line", "cycle", "complete"
    Meaning:
        Selects the generated graph family.
        This is relevant when graphs are generated procedurally.
        It is not needed when graph instances are loaded from
        relative_graph_adjList_path.

weighted
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Controls whether graph instances should be treated as weighted.

relative_graph_adjList_path
    Expected type if used: string
    Example value:
        "HOG_graphs/triangleFree_connected_3regular_4to15vertices_13instances_AdjacencyList.txt"
    Meaning:
        Path to a House of Graphs adjacency-list file, relative to QAOA/Code/.
        Required for circuit_type="HOG".
        In HOG mode, the launcher counts the graphs in this file and loops over
        valid graph indices.


Singlet injection settings
--------------------------

singlet_injection
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables or disables singlet injection in circuit construction.
        Only relevant for circuit modes that explicitly support singlet
        injection.

start_index_singlet
    Expected type if used: integer
    Example value: 0
    Meaning:
        Start index used for singlet injection.
        Only relevant if singlet_injection is enabled.


Warm-start / SDP settings
-------------------------

warm_start
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables or disables warm-start initialisation.

warm_start_mode
    Expected type if used: string
    Supported/known options: "standard"
    Meaning:
        Selects the warm-start strategy.
        Known option set depends on the implemented Python code.

lasserre_level
    Expected type if used: integer
    Example value: 2
    Meaning:
        SDP/Lasserre hierarchy level used for SDP-based relaxations,
        warm-starts, or comparisons.

initial_solver_level_M
    Expected type if used: integer
    Example value: 2
    Meaning:
        Initial SDP/Lasserre solver level.

sdp_seed
    Expected type if used: integer
    Example value: 89
    Meaning:
        Random seed used for SDP rounding, warm-start sampling, or other
        seed-dependent SDP-related preprocessing.
        The launcher can override this via SDP_SEED_OVERRIDE for repeated runs.

warm_start_corr_strength
    Expected type if used: number
    Example value: 2.0
    Meaning:
        Strength parameter for correlation-based warm-start construction.

warm_start_corr_repeats
    Expected type if used: integer
    Example value: 2
    Meaning:
        Number of repeated correlation-based warm-start attempts.

warm_start_correlations
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables or disables correlation-based warm-start logic.

compare_with_010101
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables comparison with a fixed 010101... reference state or assignment.

init_QAOAparams_close_to_zero
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Initialises QAOA parameters close to zero.

use_correlations_as_initial_params
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Uses correlation information to initialise QAOA parameters.


Runtime / launcher settings
---------------------------

time_limit
    Expected type if used: integer
    Example value: 10800
    Meaning:
        Per-job timeout in seconds.
        A value greater than zero enables timeout handling if timeout/gtimeout is
        available.

failed_instance_termination_thrsh
    Expected type if used: integer
    Example value: 10
    Meaning:
        Number of consecutive timed-out jobs after which the launcher stops
        spawning further jobs.

easiest_first
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Controls launch order.
        If true, easier/smaller jobs are launched first according to the
        launcher's internal score (heuristically).
        If false, harder/larger jobs are launched first.

cpu_util_threshold
    Expected type if used: integer or float
    Example value: 90
    Meaning:
        CPU utilisation threshold used by the launcher when CPU-based launch
        limiting is enabled.

debug
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables additional debug output in Python code where supported.

save_circuit_svg
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        Enables saving circuit diagrams as SVG if the selected mode constructs
        circuits and supports SVG output.

rerun_exclude_finished_instances
    Expected type if used: boolean
    Supported options: true, false
    Meaning:
        If true, the launcher skips run keys listed in completed_runs.txt.

optimiser_debug_iterations
    Expected type if used: integer
    Example value: 1
    Meaning:
        Debug iteration limit for optimiser internals.

num_repeats
    Expected type if used: integer
    Example value: 100
    Meaning:
        Number of repeated runs per benchmark setting.
        Useful for stochastic algorithms, repeated seeds, timing analysis, or
        reproducibility checks.


Exact eigensolver configuration notes
-------------------------------------

For optimiser="exact", the benchmark uses an exact solver/eigensolver path
instead of QAOA parameter optimisation.

The following parameters are usually relevant for exact HOG benchmarks:

    optimiser
    circuit_type
    time_limit
    failed_instance_termination_thrsh
    weighted
    easiest_first
    debug
    save_circuit_svg
    rerun_exclude_finished_instances
    num_repeats
    relative_graph_adjList_path

The following parameter groups are usually not needed for exact solving and can
therefore be set to null for clarity:

    QAOA optimiser parameters:
        optimiser_use_heuristic
        heuristic_optimiser_iterations
        heuristic_optimiser_sampleSize
        parameter_vector
        iterations_list
        learning_rate_adam
        depth_list
        optimiser_debug_iterations

    Singlet-injection parameters:
        singlet_injection
        start_index_singlet

    Warm-start / SDP parameters:
        warm_start
        warm_start_mode
        lasserre_level
        initial_solver_level_M
        sdp_seed
        warm_start_corr_strength
        warm_start_corr_repeats
        warm_start_correlations
        compare_with_010101
        init_QAOAparams_close_to_zero
        use_correlations_as_initial_params

    Generated-graph range parameters for HOG:
        n_start
        n_end
        graph_generation_type

For HOG exact solving, n_start and n_end are not needed if the launcher counts
all graphs in relative_graph_adjList_path and automatically loops over graph
indices.


Run command

-----------

Placeholder command:

    sudo nohup zsh qaoa_benchmarks.sh <path-to-config-json> > Results/logs/launcher_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

Example command:

    sudo nohup zsh qaoa_benchmarks.sh Code/run_configurations/benchmark_config_exact_hog.json > Results/logs/launcher_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

Run in sequence:

    Run first: sudo -v

    (run with nohup as well to be able to close the connection)

    pid=12345
    next_config="Code/run_configurations/benchmark_config_next.json"

    nohup zsh -c '
    pid="$1"
    next_config="$2"

    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done

    echo "Previous launcher PID $pid finished at $(date +'\''%Y-%m-%d %H:%M:%S'\'')"
    echo "Starting next config: $next_config"

    sudo nohup zsh qaoa_benchmarks.sh "$next_config" > Results/logs/launcher_$(date +'\''%Y%m%d_%H%M%S'\'').log 2>&1 &
    ' zsh "$pid" "$next_config" > Results/logs/waiter_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

Notes:
    - Run this from the QAOA/ directory.
    - <path-to-config-json> must be explicit. The launcher intentionally has no
      default config file to avoid accidental runs with the wrong parameters.
    - The launcher writes its own stdout/stderr to Results/logs/launcher_<timestamp>.log.
    - Each benchmark job writes separate logs under Results/logs/<optimiser>/<run_tag>/.

