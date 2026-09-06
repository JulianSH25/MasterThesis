#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


QAOA_ROOT = Path(__file__).resolve().parent
SNAPSHOT_DIR = QAOA_ROOT / "Results/config_snapshots"
RESULT_DIR = QAOA_ROOT / "Results/logs/ADAM"

DATASET_HINTS = {
    "triangleFree_connected_4to8vertices_176instances": "tf_176",
    "3regular_connected_74instances": "reg3_74",
    "random_graphs_8to14vertices_78instances": "rand_78",
    "cache_size_scaling_cycles_4to20vertices_9instances": "stress",
    "stress_test": "stress",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Exp4 result CSVs into a mirrored folder layout.")
    parser.add_argument(
        "name_addition",
        help='Exact result_name_suffix to organise, e.g. "_Exp4_v2_standardZeroAngles".',
    )
    parser.add_argument("--apply", action="store_true", help="Actually copy files. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Overwrite files that already exist.")
    parser.add_argument("--no-logs", action="store_true", help="Do not copy the matching run log folder.")
    return parser.parse_args()


def run_tag(snapshot: Path) -> str:
    return snapshot.stem.removeprefix("benchmark_config_")


def dataset_slug(config: dict) -> str:
    path = str(config.get("relative_graph_adjList_path", ""))
    for hint, slug in DATASET_HINTS.items():
        if hint in path:
            return slug
    return "unknown_dataset"


def lr_slug(config: dict) -> str:
    value = config.get("learning_rate_adam")
    return f"lr{str(value).replace('0.', '0').replace('.', '')}"


def heuristic_slug(config: dict) -> str:
    if not config.get("optimiser_use_heuristic"):
        return "noheuristic"
    iterations = config.get("heuristic_optimiser_iterations")
    sample_size = config.get("heuristic_optimiser_sampleSize")
    return f"h{iterations}_s{sample_size}"


def family_slug(config: dict) -> str:
    if not config.get("warm_start"):
        return "QAOA_only"
    return f"L{config.get('lasserre_level')}M{config.get('initial_solver_level_M')}"


def result_csv_for(snapshot: Path, config: dict) -> Path:
    optimiser = str(config.get("optimiser", "adam")).lower()
    return RESULT_DIR / f"qaoa_results_{optimiser}_{run_tag(snapshot)}.csv"


def target_dir_for(config: dict) -> Path:
    subexperiment = str(config["result_name_suffix"]).lstrip("_")
    experiment = subexperiment.split("_", 1)[0]
    dataset = dataset_slug(config)
    variation = f"{lr_slug(config)}_{heuristic_slug(config)}"
    return RESULT_DIR / experiment / subexperiment / variation / dataset


def target_csv_for(source: Path, config: dict) -> Path:
    dataset = dataset_slug(config)
    variation = f"{lr_slug(config)}_{heuristic_slug(config)}"
    family = family_slug(config)
    return target_dir_for(config) / f"{source.stem}__{dataset}_{variation}_{family}{source.suffix}"


def log_dir_for(snapshot: Path) -> Path:
    return RESULT_DIR / run_tag(snapshot)


def copy_result(source: Path, target: Path, apply: bool, force: bool) -> str:
    if not source.exists():
        return "missing"
    if target.exists() and not force:
        return "exists"
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return "copied"


def copy_log_dir(source: Path, target: Path, apply: bool, force: bool) -> str:
    if not source.is_dir():
        return "missing"
    if target.exists() and not force:
        return "exists"
    if apply:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    return "copied"


def main() -> None:
    args = parse_args()
    name_addition = args.name_addition.strip()
    if not name_addition:
        raise ValueError("name_addition must not be empty.")
    if not name_addition.startswith("_"):
        name_addition = f"_{name_addition}"
    experiment = name_addition.lstrip("_").split("_", 1)[0]
    csv_counts = {"copied": 0, "exists": 0, "missing": 0}
    log_counts = {"copied": 0, "exists": 0, "missing": 0, "skipped": 0}

    snapshots = sorted(SNAPSHOT_DIR.glob("benchmark_config_*.json"))

    runs = []
    available_name_additions = set()
    invalid_snapshots = 0
    for snapshot in snapshots:
        try:
            config = json.loads(snapshot.read_text())
        except json.JSONDecodeError as exc:
            invalid_snapshots += 1
            print(f"invalid snapshot {snapshot.name}: {exc}")
            continue
        if config.get("result_name_suffix"):
            available_name_additions.add(config["result_name_suffix"])
        if (
            dataset_slug(config) != "unknown_dataset"
            and str(config.get("optimiser", "")).lower() == "adam"
            and not config.get("warm_start_cache_producer_only")
            and config.get("result_name_suffix") == name_addition
        ):
            runs.append((snapshot, config))

    print(
        f"{'DRY-RUN ' if not args.apply else ''}Organising {len(runs)} {experiment} QAOA snapshots "
        f"with result_name_suffix={name_addition!r}."
    )
    if invalid_snapshots:
        print(f"Skipped {invalid_snapshots} invalid config snapshot(s).")
    if not runs:
        print(f"Available name additions: {sorted(available_name_additions)}")

    for snapshot, config in runs:
        source = result_csv_for(snapshot, config)
        target = target_csv_for(source, config)
        status = copy_result(source, target, args.apply, args.force)
        csv_counts[status] += 1

        log_status = "skipped"
        if source.exists() and not args.no_logs:
            log_source = log_dir_for(snapshot)
            log_target = target.parent / log_source.name
            log_status = copy_log_dir(log_source, log_target, args.apply, args.force)
        log_counts[log_status] += 1

        print(f"{status:7} logs={log_status:7} {source.name} -> {target.relative_to(RESULT_DIR)}")

    print("\nCSV summary:", ", ".join(f"{key}={value}" for key, value in csv_counts.items()))
    print("Log summary:", ", ".join(f"{key}={value}" for key, value in log_counts.items()))
    if not args.apply:
        print("Run again with --apply to copy the files.")


if __name__ == "__main__":
    main()
