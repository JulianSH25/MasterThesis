# MasterThesis

This repository contains the implementation used for the thesis on SDP-based warm-starts for QAOA applied to Quantum Max-Cut. It includes the SDP relaxations and their rounding procedures, the HamQAOA implementation, several warm-start modes, the benchmark configurations, and the scripts used to process the resulting data.

## Repository structure

### `SDP/`

The `SDP/` directory contains the code for generating SDP solutions, including the rounding of the relaxed solution. `Main.py` is the entry point for a standalone SDP computation. During a warm-started QAOA run, it is reached indirectly from `QAOA/Code/Main.py` through `QAOA/Code/WarmStart.py`.

`SDP_solver.py` formulates the Lasserre Level-1 relaxation and coordinates the Level-2 formulation. The Level-2-specific constraints and the implementation of the components of King's Algorithm 17 are kept in `Lasserre_level_2.py` for readability. `Rounding.py` contains the Goemans--Williamson-style / Gharibian--Parekh rounding procedure. `Utilities.py` provides the shared indexing and matrix helpers used by the SDP code. `benchmark_pipeline.py` contains the pieces required for benchmarking, while `Gurobi_exact_solver.py` is used for exact reference results where applicable.

### `QAOA/Code/`

The QAOA pipeline is located in `QAOA/Code/`. `Main.py` is the main Python entry point for an individual benchmark run. It receives one graph instance and one benchmark configuration, obtains an SDP warm start when requested, builds the circuit, runs the chosen classical optimiser, and writes the result fields used later for the analysis.

The remaining pipeline files are split by task:

- `Circuit.py` implements the HamQAOA circuit and its Qiskit statevector evaluation.
- `ParamOptimisation.py` contains the classical optimisation procedures, including Adam and the initial-point heuristic.
- `WarmStart.py` connects the SDP and QAOA pipelines, whereas `WarmStart_Helpers.py` handles cache creation, loading, and configuration-dependent warm-start preparation (auxiliary tasks).
- `StatePrep.py` prepares the circuit input states and the additional correlation or King-inspired gates required by the different warm-start modes.
- `InstanceGenerator.py` and `GraphCharacteristics.py` load or classify the graph instances used in the benchmarks.
- `Utils.py`, `benchmark_utils.py`, and `compute_run_key.py` contain shared configuration, result, and benchmark-identity helpers.

Files with a leading underscore, for example `__compare_final_qaoa_results.py`, are postprocessing scripts for CSV aggregation, result analysis, and figure generation. They do not belong to the QAOA execution pipeline. They are included so that the reported results and figures can be reproduced, but they are not needed to start a benchmark run.

### Configurations, launchers, and results

Benchmark configurations are JSON files stored in `QAOA/Code/run_configurations/`. They specify the graph data set, SDP and QAOA settings, warm-start mode, optimiser settings, seeds, cache locations, and output suffix. The exact meaning of the configuration parameters and the values used for the thesis benchmarks are described in the thesis appendix.

To create a new benchmark configuration, copy the closest existing JSON file in `QAOA/Code/run_configurations/`, adjust the required settings, and give it a distinct `result_name_suffix`. This suffix becomes part of the result and log directory names. The shell launchers snapshot the supplied JSON configuration before a run starts, so the configuration used for a result remains available in `QAOA/Results/config_snapshots/`.

`QAOA/qaoa_benchmarks.sh` is the local launcher. It receives one configuration file, prepares the individual benchmark jobs, sets `BENCHMARK_CONFIG_FILE`, and calls `QAOA/Code/Main.py` for every required instance. The `qaoa_benchmarks_dsri*.sh` files provide the corresponding launchers used on the DACS HPC cluster. The `QAOA/slurm_FINAL_BENCHMARKS_*/` directories contain the concrete Slurm submission files used for the final benchmark experiments; each of them points to the corresponding JSON configuration.

`QAOA/Results/` contains configuration snapshots, execution logs, result CSVs, exact-result data, and persistent warm-start caches. In particular, `QAOA/Results/logs/ADAM/FINAL/` contains the final Adam benchmark results used in the thesis analysis.

## Running a benchmark

The provided environment specification is `QAOA/environment_dsri.yml`. It contains the main numerical and optimisation dependencies. The QAOA circuit code additionally requires Qiskit. SCS is the default open-source SDP solver; running configurations that use MOSEK or Gurobi additionally require a local installation and, where relevant, a valid licence.

After creating and activating a suitable environment, a local benchmark can be launched from the `QAOA/` directory with:

```bash
zsh qaoa_benchmarks.sh Code/run_configurations/<path-to-configuration>.json
```

The launcher passes the selected JSON file to the Python pipeline through the `BENCHMARK_CONFIG_FILE` environment variable. `QAOA/Code/Main.py` should therefore normally be started through a launcher rather than called directly.

On DACS HPC, the corresponding Slurm file can be submitted from the `QAOA/` directory with:

```bash
sbatch slurm_FINAL_BENCHMARKS_<MODE>/<setting>/<data-set>/<job-file>.slurm
```

The final benchmark Slurm files are intentionally concrete rather than generic templates. For a new experiment, first create the JSON configuration, then create or copy a matching Slurm file that points to it.

Otherwhise the main entry point is QAOA/Code/Main.py:

### Running a Single Warm-Started Configuration

`QAOA/Code/Main.py` can be called directly for a small warm-started run. These examples solve the required SDP relaxation for the complete graph with two vertices and then run one depth-1 QAOA optimisation step.

**L1M1**

```bash
cd QAOA

BENCHMARK_CONFIG_FILE="$PWD/Code/run_configurations/FINAL_BENCHMARKS_STANDARD/lr05_noheuristic/complete_2to12/benchm_config_complete_2to12_lr05_noheuristic_L1M1.json" \
QAOA_JOBLIB_N_JOBS=1 \
conda run -n MasterThesis python Code/Main.py 1 1 2 2 Results/logs/adam/example_l1m1
```

**L2M2**

```bash
cd QAOA

BENCHMARK_CONFIG_FILE="$PWD/Code/run_configurations/FINAL_BENCHMARKS_STANDARD/lr05_noheuristic/complete_2to12/benchm_config_complete_2to12_lr05_noheuristic_L2M2.json" \
QAOA_JOBLIB_N_JOBS=1 \
conda run -n MasterThesis python Code/Main.py 1 1 2 2 Results/logs/adam/example_l2m2
```

The arguments following `Code/Main.py` are:

1. `1`: Number of Adam optimiser updates.
2. `1`: QAOA circuit depth \(p\).
3. `2`: First graph size to process. For the complete-graph configuration, this selects \(K_2\).
4. `2`: Last graph size to process. Using the same value processes only \(K_2\).
5. `Results/logs/adam/example_l1m1`: Output-file prefix. The program appends `.csv`.

`BENCHMARK_CONFIG_FILE` selects the JSON configuration and therefore the graph family, SDP level, warm-start mode, learning rate, and other benchmark settings. `QAOA_JOBLIB_N_JOBS=1` limits local parallelism to one worker, which is useful for a small test run. The L2M2 example becomes substantially more expensive for larger graphs because it solves the Level-2 SDP relaxation.

## Processing results

The scripts beginning with `__` in `QAOA/Code/` process the stored result CSVs and warm-start-cache summaries. The main scripts used for the thesis figures are:

- `__compare_sdp_results.py` for the Level-1 and Level-2 SDP comparison;
- `__compare_qaoa_only_heuristics.py` for QAOA-only learning-rate and heuristic comparisons;
- `__compare_final_qaoa_results.py` for final comparisons between QAOA configurations and warm-start modes;
- `__summarize_warm_start_caches.py` for creating a CSV summary of the cached SDP results.

The output directories used by these scripts are chosen through their command-line arguments. This keeps the raw benchmark CSVs separate from the generated figures and numerical summaries.

## Runtime

A conda environment yaml has been added to the repo: current_environment_backup/environment-cluster-exact.yml
Additionally I have included a requirements.txt file with all versions used throughout the final benchmark. All final benchmarks were conducted on the DACS HPC cluster and I recommend doing so if needed. Especially for Lasserre 2 warm start based runs, RAM is a major bottleneck.



### EXPERIMENTS ###

You will find the results of all experiments in MasterThesis/QAOA/Results/logs/adam. The QAOA/Results/logs/adam/FINAL directory contains the sorted by warm start type benchmark results while the raw and unsorted results can be found in QAOA/Results/logs/ADAM/raw_unorganised_logs_and_result_csvs_for_FINAL_BENCHMARK. Check the dedicated ReadMe in the ´´adam´´ directory for more information. The warm start modes are all given in the thesis, being 'Standard' for initial statevector only warm start as well as no warm start QAOA, 'Amplified' corr. to the Level-1 information warm start, and 'KINGAMPLIFIED' and 'KINGENTANGLED' corr. to the two respective Level-2 information warm starts. The respective subdirectories of interest for the ``FINAL`` subdirectory are those named ``FINAL_BENCHMARKS_'warmstartmode'`` and ``FINAL_BENCHMARKS_'warmstartmode'_BestPerformers`` respectively for the last experiment with higher layer depth etc. as explained in the thesis. The other directories merely contain analysis files and plots, though most of the comparing csv files have been excluded due to their size (going into GB territory). I will however gladly provide them on request. This is also the case for some combined files in the above mentioned subdirectories which the plotting/analysis code then uses, those are also excluded. Going down the hierarchy in the above mentioned subdirectories, you will find the data sorted by learning rate and parameter selection heuristic type (most that are not warm start mode standard are 'noheuristic' with 0-vector QAOA parameter initialisation; not to be confused with the initial statevector used as warm start). These directories are called ``lr[xxx]_[heuristic_type]``. Then again inside those you will find the type of graph instance, e.g. ``bipartite_454`` where the number identifies the size of the dataset, of which however not all instances where used due to computational limits (limited the graph size), the exact number of instances benchmarked of a given graph type can be found in the thesis. These directories then contain the csv files that where produced during benchmarking, containing all the relevant data that was also aggregated for the plots. These csv files are called ``qaoa_results_adam_[timestamp]_pid[process_identifier]_FINAL_BENCHMARKS_[warmstartmode]_[graphclass]_lr[learning_rate]_[heuristic_type]_[hierarchy_level].csv``. There are also those with and additional `_clipped.csv` that contain only those instances where the QAOA could improve upon the energy of the roundedn (+ rotated for L2) warm start solution. (I realise now the naming is unfortunate as it suggests the oppossite). Those were however mainly used for earlier analysis purposes and are not reported in the thesis.

Please find the explanation for the columns in those csv's below:

## Result CSV columns

The following describes the column order used in the current final-benchmark result CSVs. Fields that do not apply to a configuration are left blank. In particular, the `algorithm17_*` fields are only populated for Level-2 runs, and the `*_010101` fields are only populated when `compare_with_010101=true`.

All energy fields labelled `normalized` use the Level-1 energy convention. For Level-2 runs, this divides the native Level-2 value by two.

1. `run_id` — Unique identifier for the individual benchmark process.
2. `n` — Number of vertices/qubits in the graph instance.
3. `m` — Number of graph edges.
4. `p` — HamQAOA circuit depth, i.e. number of layers.
5. `precision/iterations` — Configured maximum number of optimiser iterations; for Adam, the requested number of Adam updates.
6. `singlet_injection` — Whether the legacy line-graph singlet-state initialisation was used.
7. `warm_start` — Whether an SDP-derived warm start was enabled.
8. `parameter_vector` — Hamiltonian parameter vector used to define the Quantum Max-Cut interaction.
9. `optimal_result` — Exact optimum recovered from the miscellaneous exact-results table. This can be blank when an analytic exact-result helper was used instead.
10. `sdp_objective_value_step1` — Raw objective value of the solved SDP relaxation.
11. `sdp_objective_value_king_normalized_step1` — SDP relaxation objective after conversion to the common Level-1 energy convention.
12. `algorithm17_actual_energy` — Actual energy of the final Level-2 entangled state constructed through the King-inspired rotation procedure.
13. `algorithm17_lower_bound_energy` — Analytical lower bound for the Level-2 rotation construction; populated only where applicable, in particular for triangle-free instances.
14. `algorithm17_analytic_F_value` — Value of the analytical edge-energy expression \(F\) used in the Level-2 rotation construction.
15. `algorithm17_F_bound_applicable` — Whether the analytical \(F\)-based lower bound applies to this instance.
16. `algorithm17_beta_mode` — Method used to choose \(\beta^\star\), for example `fixed` or `analytic_instance`.
17. `algorithm17_beta_star` — Final value of \(\beta^\star\) used to construct the Level-2 rotation angles.
18. `algorithm17_beta_optimisation_time_seconds` — Time required to choose the Level-2 \(\beta^\star\) value.
19. `initial_ws_energy_prodStates_step2` — Energy of the GP/GW-rounded product state before any Level-2 entangling rotations.
20. `initial_sdp_statevector_energy` — Energy of the SDP-derived statevector before the QAOA circuit is applied.
21. `initial_qaoa_input_energy_normalized` — Energy of the state supplied to the parameterised QAOA layers.
22. `adam_initial_point_energy_normalized` — Energy evaluated at Adam update \(0\), using the selected initial parameter vector.
23. `adam_energy_history_normalized_json` — JSON list of stored energies throughout Adam optimisation. Index \(0\) represents the initial state or initial point; later entries correspond to completed Adam updates.
24. `adam_updates_completed` — Number of Adam updates actually completed.
25. `initial_sdp_statevec_ratio` — Approximation ratio of the SDP-derived statevector before QAOA, relative to the exact optimum.
26. `initial_ws_energy_010101` — Initial energy of the alternating computational-basis reference state \(\ket{0101\ldots}\).
27. `QAOA_improvement_over_SDP_statevectorEnergy` — Final QAOA energy minus the initial SDP-statevector energy, in the native energy convention.
28. `QAOA_improvement_over_SDP_prodStatesEnergy` — Final QAOA energy minus the rounded product-state energy, in the native energy convention.
29. `result` — Final QAOA energy after optimisation, normalised to the common Level-1 convention.
30. `result_010101` — Final QAOA energy obtained from the alternating \(\ket{0101\ldots}\) reference state.
31. `approx_ratio` — Final QAOA approximation ratio, \(E_{\mathrm{QAOA}}/E_{\mathrm{opt}}\).
32. `approx_ratio_010101` — Approximation ratio of the alternating-state reference run.
33. `diff. approx. ratio` — `approx_ratio - approx_ratio_010101`.
34. `sdp ws greater` — Whether the SDP-derived warm start achieved at least the approximation ratio of the alternating-state reference run.
35. `duration_seconds` — Wall-clock duration of the QAOA run for this instance.
36. `duration_seconds_excluding_warm_start_cache_io` — Runtime adjusted to exclude cache loading and saving overhead; when a cache was reused, it includes the cached warm-start computation time instead.
37. `warm_start_cache_used` — Whether a compatible persistent warm-start cache entry was reused.
38. `warm_start_compute_time_seconds` — Time used to compute the warm-start data.
39. `warm_start_load_time_seconds` — Time used to load the warm-start cache entry.
40. `warm_start_save_time_seconds` — Time used to save the warm-start cache entry.
41. `warm_start_effective_time_seconds` — Effective warm-start preparation time attributed to the run.
42. `warm_start_cache_path` — Path of the cache entry used or created for this instance.
43. `full duration_seconds` — Total wall-clock time since the benchmark process started, including setup and all preceding work in that process.
44. `finished_at` — Local timestamp at which the instance run finished.
45. `processor` — Processor architecture or model reported by the execution host.
46. `hostname` — Name of the host that ran the instance.
47. `total_ram_gb` — Total installed RAM reported by the execution host, in GB.
48. `physical_cores` — Number of physical CPU cores reported by the execution host.
49. `logical_cores` — Number of logical CPU cores reported by the execution host.
50. `python_version` — Python version used for the run.
51. `peak_ram_mb` — Peak resident-memory usage recorded for the process, in MB.
52. `Instance_is_triangle_free` — Whether the graph contains no triangles.
53. `Instance_is_3_regular` — Whether every vertex has degree three.
54. `Instance_is_bipartite` — Whether the graph is bipartite.
55. `Instance_is_regular` — Whether every vertex has the same degree.
56. `Regular_degree` — Shared vertex degree when the graph is regular.
57. `Instance_is_claw_free` — Whether the graph is claw-free.
58. `Instance_is_twin_free` — Whether the graph is twin-free.
59. `Instance_is_planar` — Whether the graph is planar.
60. `Instance_is_eulerian` — Whether every vertex has even degree.
61. `optimiser` — Classical optimiser used for the QAOA parameters, e.g. `adam`.
62. `optimiser_use_heuristic` — Whether the initial parameter-selection heuristic was enabled.
63. `circuit_type` — Circuit ansatz used, e.g. `hamqaoa`.
64. `learning_rate_adam` — Adam learning rate.
65. `warm_start_mode` — Selected warm-start mode, e.g. `standard`, `amplified`, `amplified_king`, or `entangled_king`.
66. `lasserre_level` — Level of the downstream rounding pipeline.
67. `initial_solver_level_M` — Level of the SDP relaxation that was solved.
68. `sdp_seed` — Seed used for SDP rounding.
69. `sdp_solver_mode` — SDP solver backend, e.g. `scs` or `mosek`.
70. `sdp_scs_eps` — Requested SCS solution tolerance.
71. `sdp_scs_max_iters` — Maximum number of SCS iterations.
72. `compare_with_010101` — Whether the alternating computational-basis reference run was executed.
73. `init_QAOAparams_close_to_zero` — Whether randomly sampled QAOA parameters were restricted to values near zero.
74. `use_correlations_as_initial_params` — Whether extracted SDP correlations were used to initialise QAOA parameters.
75. `graph_generation_type` — Source/type of graph generation, e.g. `hog`.
76. `weighted` — Whether the graph was treated as weighted.
77. `easiest_first` — Whether the benchmark launcher prioritised simpler instances first.
78. `debug` — Whether additional debugging output was enabled.
79. `save_circuit_svg` — Whether circuit diagrams were written as SVG files.
80. `rerun_exclude_finished_instances` — Whether the launcher skipped instances already recorded as finished.
81. `warm_start_cache_producer_only` — Whether the job only created or verified warm-start cache entries without executing QAOA.
82. `qaoa_only_from_existing_warm_start_cache` — Whether the launcher required an existing warm-start cache and did not schedule missing SDP-cache producer jobs.
83. `num_repeats` — Number of configured benchmark repetitions.
84. `relative_graph_adjList_path` — Project-relative path of the graph adjacency-list source.
85. `persistent_warm_start_cache` — Whether warm-start data was retained in the persistent cache.
86. `warm_start_cache_dir` — Directory used for persistent warm-start cache entries.
87. `warm_start_cache_max_file_mb` — Maximum permitted size of one cache file, in MB.
88. `warm_start_cache_max_total_gb` — Maximum total cache size, in GB.
89. `sdp_max_parallel` — Maximum number of concurrently scheduled SDP tasks.
90. `qaoa_max_parallel` — Maximum number of concurrently scheduled QAOA tasks.
91. `sdp_threads` — CPU-thread budget per SDP task.
92. `qaoa_threads` — CPU-thread budget per QAOA task.
93. `sdp_memory_limit_total_gb` — Total memory limit applied by the launcher to concurrent SDP tasks, in GB.
94. `sdp_memory_limit_single_gb` — Per-SDP-task memory limit applied by the launcher, in GB.
95. `sdp_memory_poll_seconds` — Interval at which the launcher polls SDP memory usage.
96. `sdp_max_retries` — Maximum number of retries for an SDP task that exceeds the configured resource limit.
97. `queue_poll_interval_seconds` — Interval at which the launcher polls its task queue.
98. `max_graph_vertices` — Largest allowed graph size, measured in vertices.
99. `qaoa_seed` — Base seed configured for QAOA parameter sampling.
100. `initialise_standard_warm_start_with_zero_angles` — Whether supported statevector-based warm-start modes use an all-zero Adam initial point.
101. `result_name_suffix` — Suffix used to identify the benchmark family in output file names.
102. `benchmark_repeat_idx` — Index of this benchmark repetition.
103. `qaoa_seed_used` — Effective QAOA random seed used for this row.
104. `hog_graph_index` — Index of the graph within the HOG adjacency-list source; blank for non-HOG graph sources.
105. `edges` — Serialised list of graph edges.
106. `weights` — Serialised list of edge weights, aligned with `edges`.

The schema is generated from the benchmark configuration. A CSV produced from a configuration containing additional scalar settings may therefore contain a few extra configuration columns.