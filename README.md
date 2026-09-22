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

## Processing results

The scripts beginning with `__` in `QAOA/Code/` process the stored result CSVs and warm-start-cache summaries. The main scripts used for the thesis figures are:

- `__compare_sdp_results.py` for the Level-1 and Level-2 SDP comparison;
- `__compare_qaoa_only_heuristics.py` for QAOA-only learning-rate and heuristic comparisons;
- `__compare_final_qaoa_results.py` for final comparisons between QAOA configurations and warm-start modes;
- `__summarize_warm_start_caches.py` for creating a CSV summary of the cached SDP results.

The output directories used by these scripts are chosen through their command-line arguments. This keeps the raw benchmark CSVs separate from the generated figures and numerical summaries.

## Runtime

A conda environment yaml has been added to the repo: current_environment_backup/MasterThesis.yml
Additionally I have included a requirements.txt file with all versions used throughout the final benchmark. All final benchmarks were conducted on the DACS HPC cluster and I recommend doing so if needed. Especially for Lasserre 2 warm start based runs, RAM is a major bottleneck.
