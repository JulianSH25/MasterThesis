#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

# Flip these switches before running the generator.
GENERATE_QAOA_FROM_CACHE = True
GENERATE_SDP_CACHE = False
GENERATE_EXACT = False

# Submit generated QAOA jobs in Slurm's held state. Release them once their
# warm-start caches are ready; SDP-cache and exact jobs are never held here.
HOLD_QAOA_JOBS_ON_SUBMIT = True

# Cluster environment used by the generated Slurm files.
CONDA_ENV_NAME = "masterthesis"
CONDA_ACTIVATE_SCRIPT = "/home/i6408800/miniforge3/bin/activate"

# Resource knobs copied into the generated JSON and Slurm files.
# Each generated QAOA job requests 16 CPUs and runs 16 one-thread workers internally.
QAOA_FROM_CACHE_CPUS_PER_TASK = 16
QAOA_FROM_CACHE_MAX_PARALLEL = 16
QAOA_FROM_CACHE_SLURM_MEM = "125G"
QAOA_FROM_CACHE_TIME = "7-00:00:00"
# Set to one node or a comma-separated allowed node list. Leave as None to
# let Slurm choose, e.g. "dacsgpu0001.fse-cslab.nl,dacsgpu0002.fse-cslab.nl".
QAOA_FROM_CACHE_NODELIST: str | None =  None

SDP_CACHE_CPUS_PER_TASK = 32
SDP_CACHE_MAX_PARALLEL = 32
SDP_CACHE_SLURM_MEM = "250G"
SDP_CACHE_TIME = "7-00:00:00"
SDP_CACHE_NODELIST: str | None = None 
SDP_CACHE_MEMORY_LIMIT_TOTAL_GB = 250
SDP_CACHE_MEMORY_LIMIT_SINGLE_GB = 100
SDP_CACHE_MAX_RETRIES = 3

EXACT_CPUS_PER_TASK = 32
EXACT_MAX_PARALLEL = 32
EXACT_SLURM_MEM = "250G"
EXACT_TIME = "7-00:00:00"
EXACT_NODELIST: str | None = None  # e.g. "dacsgpu0002.fse-cslab.nl"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "QAOA").is_dir() else SCRIPT_DIR.parent
QAOA_ROOT = REPO_ROOT / "QAOA"
EXP3_QAOA = QAOA_ROOT / "Code/run_configurations/Exp3/QAOA"
EXP3_EXACT = QAOA_ROOT / "Code/run_configurations/Exp3/exact"

EXP_NAME = "FINAL_BENCHMARKS_STANDARD"
DEPTH_LIST = [1, 2, 3, 4, 5]
ITERATIONS_LIST = [75]
QAOA_SEED = 42
SDP_SEED = 42
QAOA_WARM_START_MODE = "standard"
SDP_CACHE_WARM_START_MODE = "amplified"
ALGORITHM17_BETA_MODE = "analytic_instance"
# Graphs with more vertices are omitted. Set to None to disable the limit;
# individual DATASETS entries may override this with "max_graph_vertices".
MAX_GRAPH_VERTICES: int | None = 10

# Optional label appended to QAOA result artifacts. Leave empty for the
# historical names, or use e.g. "_Exp5_subexp1" to identify a subexperiment.
RESULT_NAME_SUFFIX = "_FINAL_BENCHMARKS_STANDARD"

# When enabled, standard warm-start QAOA runs begin Adam at zero for every
# rotation parameter, rather than using the heuristic or a random initial point.
INITIALISE_STANDARD_WARM_START_WITH_ZERO_ANGLES = True

EXP5_TEMPLATE_SET = {
    "warm_template": "benchm_config_176instances_L2M2.json",
    "l1_template": "benchm_config_176instances_L1M1.json",
    "qaoa_template": "benchm_config_QAOA_Only_176instances.json",
    "exact_template": "benchm_config_exact_176instances.json",
}

DATASETS = {
    "tf_176": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/triangleFree_connected_4to8vertices_176instances_AdjacencyList.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
    "complete_2to12": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/complete_graphs_2to12vertices_11instances_AdjacencyList.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
    "cycle_3to12": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/cycle_graphs_3to12vertices_10instances_AdjacencyList.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
    "path_2to12": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/path_graphs_2to12vertices_11instances_AdjacencyList.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
    "bipartite_454": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/list_454_graphs_adjacency_listhog_v04-12_bipartite.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": True,
    },
    "planar_clawfree_193": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/list_193_graphs_adjacency_listhog_v04-12_planar_clawFree.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
    "regular_388": {
        **EXP5_TEMPLATE_SET,
        "relative_graph_adjList_path": "HOG_graphs/Exp5/list_388_graphs_adjacency_list_hog_v04-12_regular.txt",
        "n_start": None,
        "n_end": None,
        "easiest_first": False,
    },
}

LEARNING_RATES = [
    ("lr05", 0.05),
    ("lr01", 0.01),
    ("lr005", 0.005),
    ("lr001", 0.001),
]

HEURISTICS = [
    ("noheuristic", False, None, None),
    ("h1_s10", True, 1, 10),
    ("h1_s25", True, 1, 25),
    ("h1_s50", True, 1, 50),
]

QAOA_FAMILIES = ["QAOA_only"]
SDP_CACHE_FAMILIES = ["L1M1", "L2M2", "L2M1"]

SLURM_PROFILES = {
    "qaoa_from_cache": {
        "runner": "qaoa_benchmarks_dsri_queuedJobs.sh",
        "time": QAOA_FROM_CACHE_TIME,
        "cpus": QAOA_FROM_CACHE_CPUS_PER_TASK,
        "mem": QAOA_FROM_CACHE_SLURM_MEM,
        "nodelist": QAOA_FROM_CACHE_NODELIST,
    },
    "sdp_cache": {
        "runner": "qaoa_benchmarks_dsri_queuedJobs.sh",
        "time": SDP_CACHE_TIME,
        "cpus": SDP_CACHE_CPUS_PER_TASK,
        "mem": SDP_CACHE_SLURM_MEM,
        "nodelist": SDP_CACHE_NODELIST,
    },
    "exact": {
        "runner": "qaoa_benchmarks_dsri_sdpqueue.sh",
        "time": EXACT_TIME,
        "cpus": EXACT_CPUS_PER_TASK,
        "mem": EXACT_SLURM_MEM,
        "nodelist": EXACT_NODELIST,
    },
}

LR_ORDER = {name: i for i, (name, _) in enumerate(LEARNING_RATES)}
HEUR_ORDER = {name: i for i, (name, *_rest) in enumerate(HEURISTICS)}
FAMILY_ORDER = {name: i for i, name in enumerate(QAOA_FAMILIES)}
MODE_ORDER = {"qaoa_from_cache": 0, "sdp_cache": 1, "exact": 2}
DATASET_ORDER = {name: i for i, name in enumerate(DATASETS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Exp4 JSON configs plus matching .slurm and submit .sh files. "
            "Use the GENERATE_* switches at the top of this file to choose modes."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated configs/slurm files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_qaoa_template(name: str) -> dict:
    return load_json(EXP3_QAOA / name)


def load_exact_template(name: str) -> dict:
    return load_json(EXP3_EXACT / name)


def write_json(path: Path, data: dict, force: bool) -> str:
    if path.exists() and not force:
        return "skipped"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return "updated" if existed else "created"


def write_text(path: Path, text: str, force: bool = True) -> str:
    if path.exists() and not force:
        return "skipped"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if path.suffix == ".sh":
        path.chmod(0o755)
    return "updated" if existed else "created"


def normalise_common(config: dict, dataset_base: dict, lr: float, use_heuristic: bool,
                     heuristic_iterations: int | None, heuristic_sample_size: int | None) -> None:
    config["depth_list"] = DEPTH_LIST
    config["iterations_list"] = ITERATIONS_LIST
    config["learning_rate_adam"] = lr
    config["optimiser_use_heuristic"] = use_heuristic
    config["heuristic_optimiser_iterations"] = heuristic_iterations
    config["heuristic_optimiser_sampleSize"] = heuristic_sample_size
    config["n_start"] = dataset_base.get("n_start")
    config["n_end"] = dataset_base.get("n_end")
    config["max_graph_vertices"] = dataset_base.get("max_graph_vertices", MAX_GRAPH_VERTICES)
    config["relative_graph_adjList_path"] = dataset_base["relative_graph_adjList_path"]
    config["easiest_first"] = dataset_base.get("easiest_first", False)
    config["num_repeats"] = 1
    config["sdp_threads"] = 1
    config["qaoa_threads"] = 1
    config["queue_poll_interval_seconds"] = 5
    config["completed_result_csv_paths"] = []
    config["qaoa_seed"] = QAOA_SEED
    config["initialise_standard_warm_start_with_zero_angles"] = True
    config["result_name_suffix"] = RESULT_NAME_SUFFIX


def make_warm_config(base: dict, dataset_base: dict, lr: float, use_heuristic: bool,
                     heuristic_iterations: int | None, heuristic_sample_size: int | None,
                     family: str) -> dict:
    config = deepcopy(base)
    normalise_common(config, dataset_base, lr, use_heuristic, heuristic_iterations, heuristic_sample_size)
    config["optimiser"] = "adam"
    config["warm_start"] = True
    config["warm_start_mode"] = QAOA_WARM_START_MODE
    config["initialise_standard_warm_start_with_zero_angles"] = (
        INITIALISE_STANDARD_WARM_START_WITH_ZERO_ANGLES
    )
    config["sdp_seed"] = SDP_SEED
    config["persistent_warm_start_cache"] = True
    config["warm_start_cache_producer_only"] = False
    config["qaoa_only_from_existing_warm_start_cache"] = True
    config["sdp_max_parallel"] = 0
    config["qaoa_max_parallel"] = QAOA_FROM_CACHE_MAX_PARALLEL
    config["sdp_memory_limit_total_gb"] = 0
    config["sdp_memory_limit_single_gb"] = 0
    config["sdp_max_retries"] = 0
    config["rerun_exclude_finished_instances"] = False
    if family == "L1M1":
        config["lasserre_level"] = 1
        config["initial_solver_level_M"] = 1
    elif family == "L2M1":
        config["lasserre_level"] = 2
        config["initial_solver_level_M"] = 1
    elif family == "L2M2":
        config["lasserre_level"] = 2
        config["initial_solver_level_M"] = 2
    else:
        raise ValueError(f"Unknown warm-start family: {family}")
    config["algorithm17_beta_mode"] = (
        ALGORITHM17_BETA_MODE if config["lasserre_level"] == 2 else "fixed"
    )
    return config


def make_qaoa_only_config(base: dict, dataset_base: dict, lr: float, use_heuristic: bool,
                          heuristic_iterations: int | None, heuristic_sample_size: int | None) -> dict:
    config = deepcopy(base)
    normalise_common(config, dataset_base, lr, use_heuristic, heuristic_iterations, heuristic_sample_size)
    config["optimiser"] = "adam"
    config["warm_start"] = False
    config["warm_start_mode"] = None
    config["lasserre_level"] = None
    config["initial_solver_level_M"] = None
    config["algorithm17_beta_mode"] = None
    config["sdp_seed"] = None
    config["persistent_warm_start_cache"] = False
    config["warm_start_cache_producer_only"] = False
    config["qaoa_only_from_existing_warm_start_cache"] = False
    config["sdp_max_parallel"] = 0
    config["qaoa_max_parallel"] = QAOA_FROM_CACHE_MAX_PARALLEL
    config["sdp_memory_limit_total_gb"] = 0
    config["sdp_memory_limit_single_gb"] = 0
    config["sdp_max_retries"] = 0
    return config


def make_sdp_cache_config(warm_template: dict, l1_template: dict, dataset_base: dict, family: str) -> dict:
    base = l1_template if family == "L1M1" else warm_template
    config = make_warm_config(base, dataset_base, lr=0.05, use_heuristic=False,
                              heuristic_iterations=None, heuristic_sample_size=None,
                              family=family)
    config["iterations_list"] = [ITERATIONS_LIST[-1]]
    config["depth_list"] = [DEPTH_LIST[0]]
    config["learning_rate_adam"] = None
    config["optimiser_use_heuristic"] = None
    config["heuristic_optimiser_iterations"] = None
    config["heuristic_optimiser_sampleSize"] = None
    config["warm_start_cache_producer_only"] = True
    config["warm_start_mode"] = SDP_CACHE_WARM_START_MODE
    config["initialise_standard_warm_start_with_zero_angles"] = False
    config["qaoa_only_from_existing_warm_start_cache"] = False
    config["persistent_warm_start_cache"] = True
    config["sdp_max_parallel"] = SDP_CACHE_MAX_PARALLEL
    config["qaoa_max_parallel"] = 0
    config["sdp_memory_limit_total_gb"] = SDP_CACHE_MEMORY_LIMIT_TOTAL_GB
    config["sdp_memory_limit_single_gb"] = SDP_CACHE_MEMORY_LIMIT_SINGLE_GB
    config["sdp_max_retries"] = SDP_CACHE_MAX_RETRIES
    config["rerun_exclude_finished_instances"] = True
    return config


def make_exact_config(base: dict, dataset_base: dict) -> dict:
    config = deepcopy(base)
    config["optimiser"] = "exact"
    config["optimiser_use_heuristic"] = None
    config["heuristic_optimiser_iterations"] = None
    config["heuristic_optimiser_sampleSize"] = None
    config["parameter_vector"] = None
    config["iterations_list"] = None
    config["learning_rate_adam"] = None
    config["depth_list"] = None
    config["n_start"] = dataset_base.get("n_start")
    config["n_end"] = dataset_base.get("n_end")
    config["max_graph_vertices"] = dataset_base.get("max_graph_vertices", MAX_GRAPH_VERTICES)
    config["relative_graph_adjList_path"] = dataset_base["relative_graph_adjList_path"]
    config["easiest_first"] = dataset_base.get("easiest_first", False)
    config["warm_start"] = False
    config["warm_start_mode"] = None
    config["lasserre_level"] = None
    config["initial_solver_level_M"] = None
    config["algorithm17_beta_mode"] = None
    config["sdp_seed"] = None
    config["sdp_solver_mode"] = None
    config["sdp_scs_eps"] = None
    config["sdp_scs_max_iters"] = None
    config["result_name_suffix"] = RESULT_NAME_SUFFIX
    config["warm_start_corr_strength"] = None
    config["warm_start_corr_repeats"] = None
    config["use_correlations_as_initial_params"] = None
    config["rerun_exclude_finished_instances"] = False
    config["warm_start_cache_producer_only"] = False
    config["qaoa_only_from_existing_warm_start_cache"] = False
    config["persistent_warm_start_cache"] = False
    config["warm_start_cache_dir"] = None
    config["sdp_max_parallel"] = 0
    config["qaoa_max_parallel"] = EXACT_MAX_PARALLEL
    config["sdp_threads"] = 1
    config["qaoa_threads"] = 1
    config["sdp_memory_limit_total_gb"] = 0
    config["sdp_memory_limit_single_gb"] = 0
    config["sdp_memory_poll_seconds"] = 5
    config["sdp_max_retries"] = 0
    config["queue_poll_interval_seconds"] = 5
    config.pop("completed_result_csv_paths", None)
    return config


def make_slurm(job_name: str, config_path: Path, exp_name: str, profile: dict) -> str:
    rel_config = config_path.relative_to(QAOA_ROOT)
    runner = profile["runner"]
    node_directive = (
        f"#SBATCH --nodelist={profile['nodelist']}\n"
        if profile.get("nodelist")
        else ""
    )
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --partition=research
{node_directive}#SBATCH --time={profile['time']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={profile['cpus']}
#SBATCH --mem={profile['mem']}

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p Results/slurm

if [ -f "{CONDA_ACTIVATE_SCRIPT}" ]; then
  source "{CONDA_ACTIVATE_SCRIPT}" "{CONDA_ENV_NAME}"
elif [ -f "$HOME/miniforge3/bin/activate" ]; then
  source "$HOME/miniforge3/bin/activate" "{CONDA_ENV_NAME}"
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate" "{CONDA_ENV_NAME}"
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
  source "$HOME/anaconda3/bin/activate" "{CONDA_ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "{CONDA_ENV_NAME}"
else
  echo "Error: could not find conda activation script for environment {CONDA_ENV_NAME}." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

zsh {runner} {rel_config}
"""


def family_from_stem(stem: str) -> str:
    for family in sorted(QAOA_FAMILIES, key=len, reverse=True):
        if stem.endswith(f"_{family}"):
            return family
    return stem.rsplit("_", 1)[-1]


def submit_sort_key(path: Path, slurm_root: Path) -> tuple:
    rel_parts = path.relative_to(slurm_root).parts
    mode = rel_parts[0] if rel_parts else ""
    if mode == "sdp_cache":
        dataset = rel_parts[1] if len(rel_parts) > 1 else ""
        family = family_from_stem(path.stem)
        return (MODE_ORDER[mode], 0, DATASET_ORDER.get(dataset, 999), FAMILY_ORDER.get(family, 999), path.name)
    if mode == "exact":
        dataset = rel_parts[1] if len(rel_parts) > 1 else ""
        return (MODE_ORDER[mode], 0, DATASET_ORDER.get(dataset, 999), 0, path.name)

    variation = rel_parts[0] if rel_parts else ""
    dataset = rel_parts[1] if len(rel_parts) > 1 else ""
    family = family_from_stem(path.stem)
    lr, heuristic = variation.split("_", 1) if "_" in variation else (variation, "")
    return (
        MODE_ORDER["qaoa_from_cache"],
        HEUR_ORDER.get(heuristic, 999),
        LR_ORDER.get(lr, 999),
        DATASET_ORDER.get(dataset, 999),
        FAMILY_ORDER.get(family, 999),
        path.name,
    )


def make_submit_script(slurm_files: list[Path], slurm_root: Path) -> str:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for path in sorted(slurm_files, key=lambda p: submit_sort_key(p, slurm_root)):
        top_level_dir = path.relative_to(slurm_root).parts[0]
        mode = top_level_dir if top_level_dir in {"sdp_cache", "exact"} else "qaoa_from_cache"
        hold_option = " --hold" if HOLD_QAOA_JOBS_ON_SUBMIT and mode == "qaoa_from_cache" else ""
        rel = path.relative_to(QAOA_ROOT)
        lines.append(f"sbatch{hold_option} {rel}")
    lines.append("")
    return "\n".join(lines)


def make_submit_slurm(exp_name: str, submit_script_name: str, job_name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --partition=research
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash slurm_{exp_name}/{submit_script_name}
"""


def add_count(counts: dict[str, int], key: str, action: str) -> None:
    counts[f"{key}_{action}"] = counts.get(f"{key}_{action}", 0) + 1


def generate_pair(config: dict, config_path: Path, slurm_path: Path, job_name: str,
                  mode: str, args: argparse.Namespace, counts: dict[str, int],
                  by_mode: dict[str, list[Path]]) -> None:
    config_action = write_json(config_path, config, args.force)
    add_count(counts, "configs", config_action)
    slurm_action = write_text(slurm_path, make_slurm(job_name, config_path, EXP_NAME, SLURM_PROFILES[mode]), args.force)
    add_count(counts, "slurm", slurm_action)
    by_mode[mode].append(slurm_path)


def main() -> None:
    args = parse_args()
    enabled_modes = [
        mode for mode, enabled in (
            ("qaoa_from_cache", GENERATE_QAOA_FROM_CACHE),
            ("sdp_cache", GENERATE_SDP_CACHE),
            ("exact", GENERATE_EXACT),
        ) if enabled
    ]
    if not enabled_modes:
        raise SystemExit("No generation mode enabled. Flip at least one GENERATE_* switch at the top of the file.")

    config_root = QAOA_ROOT / f"Code/run_configurations/{EXP_NAME}"
    slurm_root = QAOA_ROOT / f"slurm_{EXP_NAME}"
    counts: dict[str, int] = {}
    by_mode = {mode: [] for mode in enabled_modes}

    for dataset, dataset_base in DATASETS.items():
        warm_template = load_qaoa_template(dataset_base["warm_template"])
        l1_template = load_qaoa_template(dataset_base["l1_template"])
        qaoa_template = load_qaoa_template(dataset_base["qaoa_template"])

        if GENERATE_QAOA_FROM_CACHE:
            for lr_name, lr in LEARNING_RATES:
                for heuristic_name, use_heuristic, heuristic_iterations, heuristic_sample_size in HEURISTICS:
                    variation = f"{lr_name}_{heuristic_name}"
                    for family in QAOA_FAMILIES:
                        if family == "QAOA_only":
                            config = make_qaoa_only_config(
                                qaoa_template,
                                dataset_base,
                                lr,
                                use_heuristic,
                                heuristic_iterations,
                                heuristic_sample_size,
                            )
                        else:
                            base = l1_template if family == "L1M1" else warm_template
                            config = make_warm_config(
                                base,
                                dataset_base,
                                lr,
                                use_heuristic,
                                heuristic_iterations,
                                heuristic_sample_size,
                                family,
                            )

                        config_name = f"benchm_config_{dataset}_{variation}_{family}.json"
                        slurm_name = f"{EXP_NAME.lower()}_{dataset}_{variation}_{family}.slurm"
                        job_name = f"{EXP_NAME}_{dataset}_{variation}_{family}{RESULT_NAME_SUFFIX}"
                        generate_pair(
                            config,
                            config_root / variation / dataset / config_name,
                            slurm_root / variation / dataset / slurm_name,
                            job_name,
                            "qaoa_from_cache",
                            args,
                            counts,
                            by_mode,
                        )

        if GENERATE_SDP_CACHE:
            for family in SDP_CACHE_FAMILIES:
                config = make_sdp_cache_config(warm_template, l1_template, dataset_base, family)
                config_name = f"benchm_config_{dataset}_sdp_cache_{family}.json"
                slurm_name = f"{EXP_NAME.lower()}_sdp_cache_{dataset}_{family}.slurm"
                job_name = f"{EXP_NAME}_sdp_cache_{dataset}_{family}{RESULT_NAME_SUFFIX}"
                generate_pair(
                    config,
                    config_root / "sdp_cache" / dataset / config_name,
                    slurm_root / "sdp_cache" / dataset / slurm_name,
                    job_name,
                    "sdp_cache",
                    args,
                    counts,
                    by_mode,
                )

        if GENERATE_EXACT:
            exact_template = load_exact_template(dataset_base["exact_template"])
            config = make_exact_config(exact_template, dataset_base)
            config_name = f"benchm_config_{dataset}_exact.json"
            slurm_name = f"{EXP_NAME.lower()}_exact_{dataset}.slurm"
            job_name = f"{EXP_NAME}_exact_{dataset}{RESULT_NAME_SUFFIX}"
            generate_pair(
                config,
                config_root / "exact" / dataset / config_name,
                slurm_root / "exact" / dataset / slurm_name,
                job_name,
                "exact",
                args,
                counts,
                by_mode,
            )

    all_slurm_files = [path for files in by_mode.values() for path in files]
    for mode, slurm_files in by_mode.items():
        if not slurm_files:
            continue
        submit_name = f"submit_{EXP_NAME.lower()}_{mode}_all.sh"
        submit_slurm_name = f"submit_{EXP_NAME.lower()}_{mode}_all.slurm"
        write_text(slurm_root / submit_name, make_submit_script(slurm_files, slurm_root), force=True)
        write_text(
            slurm_root / submit_slurm_name,
            make_submit_slurm(EXP_NAME, submit_name, f"submit_{EXP_NAME}_{mode}"),
            force=True,
        )

    write_text(slurm_root / f"submit_{EXP_NAME.lower()}_all.sh", make_submit_script(all_slurm_files, slurm_root), force=True)
    write_text(
        slurm_root / f"submit_{EXP_NAME.lower()}_all.slurm",
        make_submit_slurm(EXP_NAME, f"submit_{EXP_NAME.lower()}_all.sh", f"submit_{EXP_NAME}_all"),
        force=True,
    )

    print("Enabled modes: " + ", ".join(enabled_modes))
    print(
        "Configs: "
        f"created={counts.get('configs_created', 0)}, "
        f"updated={counts.get('configs_updated', 0)}, "
        f"skipped={counts.get('configs_skipped', 0)}"
    )
    print(
        "Slurm: "
        f"created={counts.get('slurm_created', 0)}, "
        f"updated={counts.get('slurm_updated', 0)}, "
        f"skipped={counts.get('slurm_skipped', 0)}"
    )
    for mode in enabled_modes:
        print(f"{mode}: {len(by_mode[mode])} job files")
    print(f"Submit wrappers written to {slurm_root}")


if __name__ == "__main__":
    main()
