#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


GRAPH_INDEX_PATTERN = re.compile(r"hog_idx(\d+)_")


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_float = float(numerator)
        denominator_float = float(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_float == 0:
        return None
    return numerator_float / denominator_float


def _read_cache(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        result = json.loads(str(cache["warm_start_result_json"]))

    graph_index_match = GRAPH_INDEX_PATTERN.search(cache_path.name)
    graph_index = int(graph_index_match.group(1)) if graph_index_match else None
    sdp_objective = result.get("sdp_objective_value")
    rounded_energy = result.get("rounded_solution_energy")
    algorithm17_energy = result.get("actual_energy")
    analytic_f_value = result.get("analytic_F_value")
    lower_bound_energy = result.get("lower_bound_energy")

    return {
        "cache_file": str(cache_path),
        "cache_dataset": cache_path.parent.name,
        "graph_index": graph_index,
        "graph_hash": metadata.get("graph_hash"),
        "n_vertices": metadata.get("n_vertices"),
        "n_edges": metadata.get("n_edges"),
        "lasserre_level": metadata.get("lasserre_level"),
        "initial_solver_level_M": metadata.get("initial_solver_level_M"),
        "warm_start_mode": metadata.get("warm_start_mode"),
        "parameter_vector": json.dumps(metadata.get("parameter_vector")),
        "sdp_solver_mode": metadata.get("sdp_solver_mode"),
        "sdp_scs_eps": metadata.get("sdp_scs_eps"),
        "sdp_scs_max_iters": metadata.get("sdp_scs_max_iters"),
        "sdp_seed": metadata.get("sdp_seed_used", metadata.get("sdp_seed")),
        "gp_rounding_seed": result.get("gp_rounding_seed"),
        "algorithm17_seed": result.get("algorithm17_seed"),
        "sdp_objective_value": sdp_objective,
        "rounded_solution_energy": rounded_energy,
        "algorithm17_actual_energy": algorithm17_energy,
        "algorithm17_analytic_F_value": analytic_f_value,
        "algorithm17_F_bound_applicable": result.get("analytic_F_bound_applicable"),
        "algorithm17_lower_bound_energy": lower_bound_energy,
        "rounded_energy_over_sdp_objective": _ratio(rounded_energy, sdp_objective),
        "algorithm17_energy_over_sdp_objective": _ratio(algorithm17_energy, sdp_objective),
        "algorithm17_lower_bound_over_sdp_objective": _ratio(lower_bound_energy, sdp_objective),
        "warm_start_compute_time_seconds": metadata.get("warm_start_compute_time_seconds"),
        "cache_created_at": metadata.get("created_at"),
    }


def _safe_dataset_name(relative_graph_path: str) -> str:
    filename = Path(relative_graph_path).name
    filename = re.sub(r"\.[^.]*$", "", filename)
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return re.sub(r"_+", "_", filename).strip("_")


def _matches_config(row: dict[str, Any], config: dict[str, Any]) -> bool:
    expected_values = {
        "lasserre_level": config.get("lasserre_level"),
        "initial_solver_level_M": config.get("initial_solver_level_M"),
        "sdp_solver_mode": str(config.get("sdp_solver_mode") or "mosek").lower(),
        "sdp_scs_eps": config.get("sdp_scs_eps"),
        "sdp_scs_max_iters": config.get("sdp_scs_max_iters"),
    }
    if any(row[key] != value for key, value in expected_values.items()):
        return False

    graph_path = config.get("relative_graph_adjList_path")
    if graph_path and not str(row["cache_dataset"]).startswith(_safe_dataset_name(graph_path) + "__filehash_"):
        return False

    first_seed = config.get("sdp_seed")
    repeats = int(config.get("num_repeats") or 1)
    if first_seed is not None:
        valid_seeds = set(range(int(first_seed), int(first_seed) + repeats))
        if row["sdp_seed"] not in valid_seeds:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one CSV summary from warm-start cache files without running QAOA."
    )
    parser.add_argument(
        "cache_roots",
        nargs="+",
        type=Path,
        help="One or more cache files or directories to scan recursively.",
    )
    parser.add_argument("--output", required=True, type=Path, help="CSV file to write.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional benchmark config used to keep only caches belonging to that run.",
    )
    args = parser.parse_args()

    config = None
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

    cache_paths: set[Path] = set()
    for root in args.cache_roots:
        if root.is_file() and root.suffix == ".npz":
            cache_paths.add(root.resolve())
        elif root.is_dir():
            cache_paths.update(path.resolve() for path in root.rglob("*.npz"))
        else:
            raise FileNotFoundError(f"Cache path does not exist: {root}")

    rows = []
    invalid = []
    for cache_path in sorted(cache_paths):
        try:
            row = _read_cache(cache_path)
            if config is None or _matches_config(row, config):
                rows.append(row)
        except Exception as exc:
            invalid.append((cache_path, exc))

    rows.sort(
        key=lambda row: (
            str(row["cache_dataset"]),
            row["graph_index"] if row["graph_index"] is not None else -1,
            row["sdp_seed"] if row["sdp_seed"] is not None else -1,
            row["lasserre_level"] if row["lasserre_level"] is not None else -1,
            row["initial_solver_level_M"] if row["initial_solver_level_M"] is not None else -1,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if rows:
        with temporary_output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary_output.write_text("", encoding="utf-8")
    temporary_output.replace(args.output)

    print(f"Summarised {len(rows)} warm-start cache(s) in {args.output}")
    if invalid:
        print(f"Skipped {len(invalid)} invalid cache(s):")
        for cache_path, exc in invalid:
            print(f"  {cache_path}: {exc}")


if __name__ == "__main__":
    main()
