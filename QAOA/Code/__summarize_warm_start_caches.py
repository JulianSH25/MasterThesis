#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from Utils import graph_instance_hash, read_graphs_as_edge_lists


GRAPH_INDEX_PATTERN = re.compile(r"hog_idx(\d+)_")
DEFAULT_EXACT_RESULTS_PATH = (
    Path(__file__).resolve().parent / "optimal_results" / "optimal_results_misc.csv"
)


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
        "algorithm17_num_seeds": result.get("algorithm17_num_seeds", 1),
        "algorithm17_seed_start": result.get("algorithm17_seed_start"),
        "algorithm17_seeds_tried_json": json.dumps(
            result.get("algorithm17_seeds_tried", [])
        ),
        "algorithm17_candidate_energies_json": json.dumps(
            result.get("algorithm17_candidate_energies", [])
        ),
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


def _load_exact_result_index(exact_results_path: Path) -> dict[tuple[int, int, str], float]:
    if not exact_results_path.exists():
        return {}

    exact_results: dict[tuple[int, int, str], float] = {}
    with exact_results_path.open("r", newline="", encoding="utf-8") as exact_file:
        reader = csv.DictReader(exact_file)
        required = {"energy", "n", "m", "graph_hash"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Exact-results CSV {exact_results_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for line_number, exact_row in enumerate(reader, start=2):
            graph_hash = str(exact_row.get("graph_hash") or "").strip()
            if not graph_hash:
                continue
            try:
                key = (int(exact_row["n"]), int(exact_row["m"]), graph_hash)
                energy = float(exact_row["energy"])
            except (TypeError, ValueError) as exc:
                print(
                    f"Warning: skipping invalid exact result at "
                    f"{exact_results_path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            previous_energy = exact_results.get(key)
            if previous_energy is not None and not np.isclose(
                previous_energy, energy, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    "Conflicting exact energies for canonical graph identity "
                    f"n={key[0]}, m={key[1]}, hash={key[2]}"
                )
            exact_results[key] = energy

    return exact_results


def _config_graph_identities(config: dict[str, Any]) -> dict[int, tuple[int, int, str]]:
    graph_path = config.get("relative_graph_adjList_path")
    if not graph_path:
        raise ValueError(
            "The benchmark config has no relative_graph_adjList_path, so exact results "
            "cannot be matched safely."
        )
    if str(config.get("graph_generation_type") or "hog").lower() != "hog":
        raise ValueError("Safe exact-result matching currently requires indexed HOG input.")
    if config.get("weighted"):
        raise ValueError(
            "Safe exact-result matching cannot reconstruct weights for a weighted HOG run."
        )

    identities = {}
    for graph_index, edges in enumerate(read_graphs_as_edge_lists(str(graph_path))):
        weights = [1.0] * len(edges)
        vertices = {vertex for edge in edges for vertex in edge}
        n_vertices = len(vertices)
        identities[graph_index] = (
            n_vertices,
            len(edges),
            graph_instance_hash(edges, weights),
        )
    return identities


def _add_exact_result(
    row: dict[str, Any],
    graph_identities: dict[int, tuple[int, int, str]] | None,
    exact_results: dict[tuple[int, int, str], float],
    exact_results_path: Path,
) -> None:
    normalisation_factor = row.get("lasserre_level") or 1
    rounded_energy_normalized = _ratio(
        row.get("rounded_solution_energy"), normalisation_factor
    )
    algorithm17_energy_normalized = _ratio(
        row.get("algorithm17_actual_energy"), normalisation_factor
    )
    algorithm17_lower_bound_normalized = _ratio(
        row.get("algorithm17_lower_bound_energy"), normalisation_factor
    )
    row.update(
        {
            "exact_comparison_normalisation_factor": normalisation_factor,
            "rounded_solution_energy_normalized": rounded_energy_normalized,
            "algorithm17_actual_energy_normalized": algorithm17_energy_normalized,
            "algorithm17_lower_bound_energy_normalized": algorithm17_lower_bound_normalized,
            "canonical_graph_hash": None,
            "exact_result_match_status": "config_not_provided",
            "exact_result_source": None,
            "exact_optimal_energy": None,
            "rounded_energy_over_exact_optimum": None,
            "algorithm17_energy_over_exact_optimum": None,
            "algorithm17_lower_bound_over_exact_optimum": None,
        }
    )
    if graph_identities is None:
        return

    graph_index = row.get("graph_index")
    identity = graph_identities.get(graph_index)
    if identity is None:
        row["exact_result_match_status"] = "graph_index_not_found"
        return

    n_vertices, n_edges, canonical_hash = identity
    row["canonical_graph_hash"] = canonical_hash
    if row.get("n_vertices") != n_vertices or row.get("n_edges") != n_edges:
        row["exact_result_match_status"] = "graph_metadata_mismatch"
        return

    exact_energy = exact_results.get((n_vertices, n_edges, canonical_hash))
    if exact_energy is None:
        row["exact_result_match_status"] = "missing"
        return

    row.update(
        {
            "exact_result_match_status": "matched",
            "exact_result_source": str(exact_results_path),
            "exact_optimal_energy": exact_energy,
            "rounded_energy_over_exact_optimum": _ratio(
                rounded_energy_normalized, exact_energy
            ),
            "algorithm17_energy_over_exact_optimum": _ratio(
                algorithm17_energy_normalized, exact_energy
            ),
            "algorithm17_lower_bound_over_exact_optimum": _ratio(
                algorithm17_lower_bound_normalized, exact_energy
            ),
        }
    )


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
    parser.add_argument(
        "--exact-results",
        type=Path,
        default=DEFAULT_EXACT_RESULTS_PATH,
        help=(
            "Exact-results CSV containing n, m, and canonical graph_hash columns "
            f"(default: {DEFAULT_EXACT_RESULTS_PATH})."
        ),
    )
    args = parser.parse_args()

    config = None
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

    graph_identities = _config_graph_identities(config) if config is not None else None
    exact_results = _load_exact_result_index(args.exact_results)

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
                _add_exact_result(
                    row,
                    graph_identities,
                    exact_results,
                    args.exact_results,
                )
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
    if rows:
        matched_exact = sum(row["exact_result_match_status"] == "matched" for row in rows)
        print(
            f"Matched exact results for {matched_exact}/{len(rows)} cache(s) using "
            "canonical graph hash, n, and m."
        )
    if invalid:
        print(f"Skipped {len(invalid)} invalid cache(s):")
        for cache_path, exc in invalid:
            print(f"  {cache_path}: {exc}")


if __name__ == "__main__":
    main()
