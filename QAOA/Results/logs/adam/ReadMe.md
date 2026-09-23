FINAL contains all benchmark results and analysis files along with the respective plots, that are partially used in the thesis, in a sorted by benchmark manner.

raw_unorganised_logs_and_result_csvs_for_FINAL_BENCHMARK on the other hand contains all the same benchmark data (without any analysis files) and additionally the log files, but in an unsorted manner. Each csv and folder maps to the final subdirectories in ´´FINAL´´via the timestamp and pid number in the respective names; Originally these folders and csv files where written directly into the ´´adam´´ directory, but I sorted it to make it less cluttered.


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