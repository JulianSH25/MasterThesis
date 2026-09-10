#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import math
import re
from itertools import combinations
from pathlib import Path

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


TIMESTAMP_PATTERN = re.compile(r"_(\d{8}_\d{6})_pid")
LEARNING_RATE_PATTERN = re.compile(r"^lr(\d+)_")
FAMILY_ORDER = ["QAOA_only", "L1M1", "L2M1", "L2M2"]
FAMILY_LABELS = {
    "QAOA_only": "QAOA-only",
    "L1M1": "L1M1",
    "L2M1": "L2M1",
    "L2M2": "L2M2",
}
FAMILY_COLORS = {
    "QAOA_only": "#2CA02C",
    "L1M1": "#2878B5",
    "L2M1": "#8C62AA",
    "L2M2": "#D62728",
}
WARM_START_ORDER = [
    "QAOA_only",
    "standard",
    "amplified",
    "king_amplified",
    "king_entangled",
]
WARM_START_LABELS = {
    "QAOA_only": "QAOA-only",
    "standard": "Statevector-only",
    "amplified": "Amplified",
    "king_amplified": "Amplified King",
    "king_entangled": "King Entangled",
}
WARM_START_COLORS = {
    "QAOA_only": "#555555",
    "standard": "#2878B5",
    "amplified": "#E28E2C",
    "king_amplified": "#C43C39",
    "king_entangled": "#7A5195",
}
FAMILY_MARKERS = {
    "QAOA_only": "^",
    "L1M1": "o",
    "L2M1": "D",
    "L2M2": "s",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
LINESTYLES = [
    "-",
    (0, (5, 2)),
    (0, (5, 2, 1.5, 2)),
    (0, (1, 1.5)),
]
SUMMARY_ORDER = ("mean", "median", "range", "boxplot")
SUMMARY_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "range": "Minimum and maximum",
    "boxplot": "Distribution",
}
SUMMARY_SUFFIXES = {
    "mean": "",
    "median": "_median",
    "range": "_min_max",
    "boxplot": "_boxplots",
}


def parse_args() -> argparse.Namespace:
    qaoa_root = Path(__file__).resolve().parents[1]
    default_root = qaoa_root / "Results" / "logs" / "ADAM" / "FINAL"

    parser = argparse.ArgumentParser(
        description=(
            "Analyse final QAOA quality, optimisation trajectories, initial-state "
            "gains, graph-size trends, pairwise differences, and runtime. The input "
            "may be one organised experiment directory or a parent containing several."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Organised experiment or parent directory (default: {default_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/_final_qaoa_analysis)",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="GLOB",
        help="Experiment-name glob to include; repeat as needed (default: all).",
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Setting-directory glob such as lr005_h1_s50; repeat as needed "
            "(default: all settings, kept as separate plotted series)."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="GLOB",
        help="Dataset-directory glob to include; repeat as needed (default: all).",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=FAMILY_ORDER,
        default=[],
        help="QAOA family to include; repeat as needed (default: all available).",
    )
    parser.add_argument(
        "--clipping",
        choices=["both", "included", "unclipped"],
        default="both",
        help=(
            "Analyse rows with clipping retained, unclipped warm-start rows only, "
            "or both views (default: both)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    return parser.parse_args()


def matches_any(value: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def family_from_filename(path: Path) -> str | None:
    name = path.name
    if "_QAOA_only_clipped.csv" in name:
        return "QAOA_only"
    for family in ["L1M1", "L2M1", "L2M2"]:
        if f"_{family}_clipped.csv" in name:
            return family
    return None


def result_file_sort_key(path: Path) -> tuple[str, int]:
    match = TIMESTAMP_PATTERN.search(path.name)
    timestamp = match.group(1) if match else ""
    return timestamp, path.stat().st_mtime_ns


def contains_result_files(path: Path) -> bool:
    for setting_dir in path.iterdir():
        if not setting_dir.is_dir() or setting_dir.name.startswith("_"):
            continue
        for dataset_dir in setting_dir.iterdir():
            if dataset_dir.is_dir() and any(dataset_dir.glob("*_clipped.csv")):
                return True
    return False


def discover_experiment_dirs(root: Path) -> list[Path]:
    if contains_result_files(root):
        return [root]

    experiments = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and contains_result_files(path)
    ]
    return sorted(experiments)


def discover_latest_files(
    root: Path,
    experiment_patterns: list[str],
    setting_patterns: list[str],
    dataset_patterns: list[str],
    families: list[str],
) -> pd.DataFrame:
    candidates: dict[tuple[str, str, str, str], list[Path]] = {}
    experiment_dirs = discover_experiment_dirs(root)
    if not experiment_dirs:
        raise FileNotFoundError(f"No organised QAOA result directories found below {root}")

    for experiment_dir in experiment_dirs:
        experiment = experiment_dir.name
        if not matches_any(experiment, experiment_patterns):
            continue

        for setting_dir in sorted(experiment_dir.iterdir()):
            if not setting_dir.is_dir() or setting_dir.name.startswith("_"):
                continue
            setting = setting_dir.name
            if not matches_any(setting, setting_patterns):
                continue

            for dataset_dir in sorted(setting_dir.iterdir()):
                if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
                    continue
                dataset = dataset_dir.name
                if not matches_any(dataset, dataset_patterns):
                    continue

                for path in dataset_dir.glob("*_clipped.csv"):
                    family = family_from_filename(path)
                    if family is None or (families and family not in families):
                        continue
                    key = (experiment, setting, dataset, family)
                    candidates.setdefault(key, []).append(path)

    records: list[dict[str, object]] = []
    for (experiment, setting, dataset, family), paths in sorted(candidates.items()):
        latest = max(paths, key=result_file_sort_key)
        records.append(
            {
                "experiment": experiment,
                "setting": setting,
                "dataset": dataset,
                "family": family,
                "source_file": str(latest.resolve()),
                "candidate_files": len(paths),
            }
        )
        if len(paths) > 1:
            print(
                f"Warning: {experiment}/{setting}/{dataset}/{family} contains "
                f"{len(paths)} clipped CSVs; using {latest.name}."
            )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise FileNotFoundError("No result CSVs matched the requested filters.")
    return manifest


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def first_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        result = result.fillna(numeric_series(frame, column))
    return result


def parse_sequence(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                raw = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
    if not isinstance(raw, (list, tuple)):
        return []

    parsed: list[float] = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return parsed


def parse_edges_and_weights(edges_value: object, weights_value: object) -> str | None:
    try:
        edges = ast.literal_eval(str(edges_value)) if isinstance(edges_value, str) else edges_value
        weights = (
            ast.literal_eval(str(weights_value))
            if isinstance(weights_value, str)
            else weights_value
        )
    except (ValueError, SyntaxError):
        return None

    if not isinstance(edges, (list, tuple)):
        return None
    if not isinstance(weights, (list, tuple)) or len(weights) != len(edges):
        weights = [1.0] * len(edges)

    canonical: list[tuple[int, int, float]] = []
    try:
        for edge, weight in zip(edges, weights):
            u, v = int(edge[0]), int(edge[1])
            canonical.append((min(u, v), max(u, v), round(float(weight), 12)))
    except (TypeError, ValueError, IndexError):
        return None
    canonical.sort()
    return json.dumps(canonical, separators=(",", ":"))


def graph_signature(row: pd.Series) -> str:
    canonical = parse_edges_and_weights(row.get("edges"), row.get("weights"))
    n_value = row.get("n")
    m_value = row.get("m")
    if canonical is not None:
        return f"n={n_value}|m={m_value}|edges={canonical}"

    graph_index = row.get("hog_graph_index")
    if pd.notna(graph_index):
        return f"n={n_value}|m={m_value}|hog_graph_index={graph_index}"
    return f"n={n_value}|m={m_value}|source_row={row.get('source_row')}"


def value_is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(str(value).strip())


def infer_clipped(frame: pd.DataFrame) -> pd.Series:
    clipped = pd.Series(False, index=frame.index)
    for column in [
        "result_before_warm_start_clip",
        "approx_ratio_before_warm_start_clip",
    ]:
        if column in frame.columns:
            clipped |= frame[column].map(value_is_present)
    if "warm_start_clipped" in frame.columns:
        clipped |= frame["warm_start_clipped"].astype(str).str.lower().isin(
            ["1", "true", "yes"]
        )
    return clipped


def learning_rate_from_setting(setting: str) -> float:
    match = LEARNING_RATE_PATTERN.match(setting)
    if match is None:
        return math.nan
    token = match.group(1)
    return float(f"0.{token}")


def initialisation_from_setting(setting: str) -> str:
    match = LEARNING_RATE_PATTERN.match(setting)
    return setting[match.end():] if match is not None else setting


def format_initialisation_name(name: str) -> str:
    if name == "noheuristic":
        return "No heuristic"
    match = re.fullmatch(r"h(\d+)_s(\d+)", name)
    if match is not None:
        return f"Heuristic: {match.group(1)} short update(s), sample size {match.group(2)}"
    return name.replace("_", " ")


def warm_start_type(experiment: str, family: str) -> str:
    if family == "QAOA_only":
        return "QAOA_only"
    compact = experiment.lower().replace("_", "")
    if "kingentangled" in compact:
        return "king_entangled"
    if "kingamplified" in compact:
        return "king_amplified"
    if "amplified" in compact:
        return "amplified"
    if "standard" in compact:
        return "standard"
    return experiment


def assign_series_labels(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    experiment_count = results["experiment"].nunique()
    setting_count = results["setting"].nunique()
    family_count = results["family"].nunique()
    results["warm_start_type"] = [
        warm_start_type(experiment, family)
        for experiment, family in zip(results["experiment"], results["family"])
    ]
    results["warm_start_label"] = results["warm_start_type"].map(
        lambda value: WARM_START_LABELS.get(value, value.replace("_", " "))
    )
    if experiment_count == 1 and setting_count == 1:
        results["series"] = results["family"].map(FAMILY_LABELS)
    elif experiment_count == 1 and family_count == 1:
        results["series"] = results["setting"]
    elif experiment_count == 1:
        results["series"] = (
            results["setting"] + " | " + results["family"].map(FAMILY_LABELS)
        )
    else:
        family_suffix = np.where(
            results["family"].eq("QAOA_only"),
            "",
            " | " + results["family"].map(FAMILY_LABELS),
        )
        results["series"] = (
            results["warm_start_label"] + " | " + results["setting"] + family_suffix
        )
    return results


def load_results(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for record in manifest.to_dict("records"):
        path = Path(str(record["source_file"]))
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.copy()
        frame["source_row"] = np.arange(len(frame))
        for column in ["experiment", "setting", "dataset", "family", "source_file"]:
            frame[column] = record[column]

        optimum = first_numeric(frame, ["optimal_result", "optimal_value"])
        final_energy = first_numeric(frame, ["result", "final_energy"])
        final_ratio = first_numeric(frame, ["approx_ratio"])
        final_ratio = final_ratio.fillna(final_energy.div(optimum.replace(0, np.nan)))

        sdp_state_energy = first_numeric(
            frame,
            [
                "initial_sdp_statevector_energy",
                "algorithm17_actual_energy",
                "initial_ws_energy_prodStates_step2",
            ],
        )
        sdp_state_ratio = first_numeric(frame, ["initial_sdp_statevec_ratio"])
        sdp_state_ratio = sdp_state_ratio.fillna(
            sdp_state_energy.div(optimum.replace(0, np.nan))
        )

        circuit_initial_energy = first_numeric(
            frame,
            [
                "adam_initial_point_energy_normalized",
                "initial_qaoa_input_energy_normalized",
            ],
        )
        circuit_initial_ratio = circuit_initial_energy.div(optimum.replace(0, np.nan))

        frame["optimal_value_analysis"] = optimum
        frame["final_energy_analysis"] = final_energy
        frame["final_ratio"] = final_ratio
        frame["sdp_state_ratio"] = sdp_state_ratio
        frame["circuit_initial_ratio"] = circuit_initial_ratio
        frame["was_clipped"] = infer_clipped(frame)
        frame["qaoa_depth"] = numeric_series(frame, "p")
        frame["adam_iterations"] = first_numeric(
            frame, ["precision/iterations", "adam_updates_completed"]
        )
        frame["qaoa_seed_effective"] = first_numeric(
            frame, ["qaoa_seed_used", "qaoa_seed"]
        )
        frame["learning_rate"] = numeric_series(frame, "learning_rate_adam")
        frame["learning_rate"] = frame["learning_rate"].fillna(
            frame["setting"].map(learning_rate_from_setting)
        )
        frame["initialisation_setting"] = frame["setting"].map(
            initialisation_from_setting
        )
        frame["n_vertices"] = numeric_series(frame, "n")
        frame["n_edges"] = numeric_series(frame, "m")
        frame["duration_analysis"] = first_numeric(
            frame, ["duration_seconds", "full duration_seconds"]
        )
        frame["full_duration_analysis"] = first_numeric(
            frame, ["full duration_seconds", "duration_seconds"]
        )
        frame["updates_completed_analysis"] = numeric_series(
            frame, "adam_updates_completed"
        )
        frame["graph_signature"] = frame.apply(graph_signature, axis=1)
        frame["instance_key"] = frame["dataset"] + "|" + frame["graph_signature"]

        key_columns = [
            "instance_key",
            "qaoa_depth",
            "adam_iterations",
            "qaoa_seed_effective",
        ]
        frame["pair_key"] = frame[key_columns].astype(str).agg("|".join, axis=1)
        frame["series_id"] = (
            frame["experiment"]
            + "|"
            + frame["setting"]
            + "|"
            + frame["family"]
        )

        coverage_rows.append(
            {
                **record,
                "csv_rows": len(frame),
                "usable_final_ratio_rows": int(frame["final_ratio"].notna().sum()),
                "trajectory_rows": int(
                    frame.apply(has_trajectory_data, axis=1).sum()
                ),
                "clipped_rows": int(frame["was_clipped"].sum()),
            }
        )
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    results = results.dropna(subset=["final_ratio", "qaoa_depth"])
    if results.empty:
        raise ValueError("The selected CSVs contain no usable final QAOA ratios.")

    sort_columns = [
        column
        for column in ["finished_at", "source_file", "source_row"]
        if column in results.columns
    ]
    if sort_columns:
        results = results.sort_values(sort_columns)
    results = results.drop_duplicates(["series_id", "pair_key"], keep="last")
    results = assign_series_labels(results)
    return results, pd.DataFrame(coverage_rows)


def has_trajectory_data(row: pd.Series) -> bool:
    for column in [
        "adam_approx_ratio_history_json",
        "adam_energy_history_normalized_json",
        "adam_energy_history_json",
    ]:
        if column in row.index and parse_sequence(row.get(column)):
            return True
    return False


def extract_trajectory(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    initial_ratio_by_index: dict[int, float] = {}

    for row_index, row in results.iterrows():
        ratios = parse_sequence(row.get("adam_approx_ratio_history_json"))
        if not ratios:
            energies = parse_sequence(row.get("adam_energy_history_normalized_json"))
            if not energies:
                energies = parse_sequence(row.get("adam_energy_history_json"))
            optimum = row.get("optimal_value_analysis")
            if energies and pd.notna(optimum) and float(optimum) != 0:
                ratios = [energy / float(optimum) for energy in energies]

        if not ratios:
            continue
        initial_ratio_by_index[row_index] = ratios[0]
        for update, ratio in enumerate(ratios):
            records.append(
                {
                    "row_index": row_index,
                    "experiment": row["experiment"],
                    "setting": row["setting"],
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "series": row["series"],
                    "pair_key": row["pair_key"],
                    "qaoa_depth": row["qaoa_depth"],
                    "adam_update": update,
                    "approx_ratio": ratio,
                }
            )

    enriched = results.copy()
    history_initial = pd.Series(initial_ratio_by_index, dtype=float)
    enriched["circuit_initial_ratio"] = enriched["circuit_initial_ratio"].fillna(
        history_initial
    )
    enriched["gain_over_circuit_initial"] = (
        enriched["final_ratio"] - enriched["circuit_initial_ratio"]
    )
    enriched["relative_gain_over_circuit_initial"] = enriched[
        "gain_over_circuit_initial"
    ].div(enriched["circuit_initial_ratio"].replace(0, np.nan))
    enriched["gain_over_sdp_state"] = (
        enriched["final_ratio"] - enriched["sdp_state_ratio"]
    )
    enriched["relative_gain_over_sdp_state"] = enriched["gain_over_sdp_state"].div(
        enriched["sdp_state_ratio"].replace(0, np.nan)
    )
    return enriched, pd.DataFrame(records)


def aggregate_results(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = results.groupby(group_columns, observed=True, dropna=False)
    summary = grouped.agg(
        rows=("final_ratio", "size"),
        instances=("instance_key", "nunique"),
        mean_final_ratio=("final_ratio", "mean"),
        median_final_ratio=("final_ratio", "median"),
        std_final_ratio=("final_ratio", "std"),
        min_final_ratio=("final_ratio", "min"),
        max_final_ratio=("final_ratio", "max"),
        q25_final_ratio=("final_ratio", lambda values: values.quantile(0.25)),
        q75_final_ratio=("final_ratio", lambda values: values.quantile(0.75)),
        mean_circuit_initial_ratio=("circuit_initial_ratio", "mean"),
        median_circuit_initial_ratio=("circuit_initial_ratio", "median"),
        min_circuit_initial_ratio=("circuit_initial_ratio", "min"),
        max_circuit_initial_ratio=("circuit_initial_ratio", "max"),
        q25_circuit_initial_ratio=(
            "circuit_initial_ratio",
            lambda values: values.quantile(0.25),
        ),
        q75_circuit_initial_ratio=(
            "circuit_initial_ratio",
            lambda values: values.quantile(0.75),
        ),
        mean_sdp_state_ratio=("sdp_state_ratio", "mean"),
        median_sdp_state_ratio=("sdp_state_ratio", "median"),
        min_sdp_state_ratio=("sdp_state_ratio", "min"),
        max_sdp_state_ratio=("sdp_state_ratio", "max"),
        q25_sdp_state_ratio=("sdp_state_ratio", lambda values: values.quantile(0.25)),
        q75_sdp_state_ratio=("sdp_state_ratio", lambda values: values.quantile(0.75)),
        mean_gain_over_circuit_initial=("gain_over_circuit_initial", "mean"),
        median_gain_over_circuit_initial=("gain_over_circuit_initial", "median"),
        min_gain_over_circuit_initial=("gain_over_circuit_initial", "min"),
        max_gain_over_circuit_initial=("gain_over_circuit_initial", "max"),
        q25_gain_over_circuit_initial=(
            "gain_over_circuit_initial",
            lambda values: values.quantile(0.25),
        ),
        q75_gain_over_circuit_initial=(
            "gain_over_circuit_initial",
            lambda values: values.quantile(0.75),
        ),
        mean_relative_gain_over_circuit_initial=(
            "relative_gain_over_circuit_initial",
            "mean",
        ),
        mean_gain_over_sdp_state=("gain_over_sdp_state", "mean"),
        median_gain_over_sdp_state=("gain_over_sdp_state", "median"),
        min_gain_over_sdp_state=("gain_over_sdp_state", "min"),
        max_gain_over_sdp_state=("gain_over_sdp_state", "max"),
        q25_gain_over_sdp_state=(
            "gain_over_sdp_state",
            lambda values: values.quantile(0.25),
        ),
        q75_gain_over_sdp_state=(
            "gain_over_sdp_state",
            lambda values: values.quantile(0.75),
        ),
        mean_relative_gain_over_sdp_state=("relative_gain_over_sdp_state", "mean"),
        mean_duration_seconds=("duration_analysis", "mean"),
        mean_full_duration_seconds=("full_duration_analysis", "mean"),
        median_full_duration_seconds=("full_duration_analysis", "median"),
        min_full_duration_seconds=("full_duration_analysis", "min"),
        max_full_duration_seconds=("full_duration_analysis", "max"),
        q25_full_duration_seconds=(
            "full_duration_analysis",
            lambda values: values.quantile(0.25),
        ),
        q75_full_duration_seconds=(
            "full_duration_analysis",
            lambda values: values.quantile(0.75),
        ),
        mean_updates_completed=("updates_completed_analysis", "mean"),
    ).reset_index()

    improvement = grouped["gain_over_circuit_initial"].agg(
        improved=lambda values: int((values > 1e-12).sum()),
        tied=lambda values: int((values.abs() <= 1e-12).sum()),
        worse=lambda values: int((values < -1e-12).sum()),
    ).reset_index()
    summary = summary.merge(improvement, on=group_columns, how="left")
    denominator = summary[["improved", "tied", "worse"]].sum(axis=1).replace(0, np.nan)
    summary["improvement_rate_over_circuit_initial"] = (
        summary["improved"] / denominator
    )
    return summary


def aggregate_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    if trajectory.empty:
        return trajectory
    return (
        trajectory.groupby(
            [
                "experiment",
                "setting",
                "dataset",
                "family",
                "series",
                "qaoa_depth",
                "adam_update",
            ],
            observed=True,
        )["approx_ratio"]
        .agg(
            rows="size",
            mean_approx_ratio="mean",
            median_approx_ratio="median",
            std="std",
            min_approx_ratio="min",
            max_approx_ratio="max",
            q25_approx_ratio=lambda values: values.quantile(0.25),
            q75_approx_ratio=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def pairwise_scope(results: pd.DataFrame, dataset: str) -> list[dict[str, object]]:
    subset = results if dataset == "ALL_GRAPHS" else results.loc[results["dataset"].eq(dataset)]
    series_values = sorted(subset["series"].unique())
    rows: list[dict[str, object]] = []
    for left, right in combinations(series_values, 2):
        left_values = (
            subset.loc[subset["series"].eq(left), ["pair_key", "final_ratio"]]
            .drop_duplicates("pair_key", keep="last")
            .rename(columns={"final_ratio": "left_ratio"})
        )
        right_values = (
            subset.loc[subset["series"].eq(right), ["pair_key", "final_ratio"]]
            .drop_duplicates("pair_key", keep="last")
            .rename(columns={"final_ratio": "right_ratio"})
        )
        paired = left_values.merge(right_values, on="pair_key", how="inner")
        if paired.empty:
            continue
        delta = paired["right_ratio"] - paired["left_ratio"]
        rows.append(
            {
                "dataset": dataset,
                "left_series": left,
                "right_series": right,
                "matched_rows": len(delta),
                "mean_right_minus_left": delta.mean(),
                "median_right_minus_left": delta.median(),
                "right_wins": int((delta > 1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "right_losses": int((delta < -1e-12).sum()),
                "right_win_rate": float((delta > 1e-12).mean()),
            }
        )
    return rows


def build_pairwise_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = pairwise_scope(results, "ALL_GRAPHS")
    for dataset in sorted(results["dataset"].unique()):
        rows.extend(pairwise_scope(results, dataset))
    return pd.DataFrame(rows)


def format_dataset_name(name: str) -> str:
    replacements = {
        "bipartite_454": "Bipartite",
        "complete_2to12": "Complete",
        "cycle_3to12": "Cycle",
        "path_2to12": "Path",
        "planar_clawfree_193": "Planar claw-free",
        "regular_388": "Regular",
        "tf_176": "Triangle-free",
        "tf_829": "Triangle-free",
    }
    if name == "ALL_GRAPHS":
        return "All instances"
    return replacements.get(name, name.replace("_", " "))


def metric_summary_title(statistic: str, metric: str) -> str:
    if statistic == "boxplot":
        return f"{metric} distribution"
    if statistic == "range":
        return f"{metric}: minimum and maximum"
    return f"{SUMMARY_LABELS[statistic]} {metric.lower()}"


def plot_title(title: str, context: str) -> str:
    return f"{title} | {context}" if context else title


def series_order(results: pd.DataFrame) -> list[str]:
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    warm_start_rank = {
        mode: index for index, mode in enumerate(WARM_START_ORDER)
    }
    metadata = (
        results[["series", "warm_start_type", "setting", "family"]]
        .drop_duplicates()
        .assign(
            warm_start_rank=lambda frame: frame["warm_start_type"]
            .map(warm_start_rank)
            .fillna(99),
            family_rank=lambda frame: frame["family"].map(family_rank).fillna(99),
        )
    )
    if results["experiment"].nunique() == 1:
        metadata = metadata.sort_values(["family_rank", "setting", "series"])
    else:
        metadata = metadata.sort_values(
            ["warm_start_rank", "setting", "family_rank", "series"]
        )
    return metadata["series"].tolist()


def shade_color(
    base_color: str | tuple[float, ...],
    index: int,
    count: int,
) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(base_color), dtype=float)
    if count <= 1:
        return tuple(rgb)
    adjustment = float(np.linspace(0.32, -0.16, count)[index])
    if adjustment >= 0:
        shaded = rgb + (1.0 - rgb) * adjustment
    else:
        shaded = rgb * (1.0 + adjustment)
    return tuple(np.clip(shaded, 0.0, 1.0))


def build_styles(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    order = series_order(results)
    metadata = (
        results[["series", "warm_start_type", "setting", "family"]]
        .drop_duplicates()
        .set_index("series")
    )
    setting_styles = {
        setting: LINESTYLES[index % len(LINESTYLES)]
        for index, setting in enumerate(sorted(metadata["setting"].unique()))
    }
    single_experiment = results["experiment"].nunique() == 1
    colour_dimension = "family" if single_experiment else "warm_start_type"
    colour_groups = list(dict.fromkeys(metadata.loc[order, colour_dimension].tolist()))
    base_colors = FAMILY_COLORS if single_experiment else WARM_START_COLORS
    palette = plt.get_cmap("tab20", max(len(colour_groups), 1))
    series_colors: dict[str, tuple[float, float, float]] = {}
    for group_index, group in enumerate(colour_groups):
        group_series = [
            series for series in order if metadata.loc[series, colour_dimension] == group
        ]
        base_color = base_colors.get(group, palette(group_index))
        for shade_index, series in enumerate(group_series):
            series_colors[series] = shade_color(
                base_color, shade_index, len(group_series)
            )

    styles: dict[str, dict[str, object]] = {}
    for index, series in enumerate(order):
        row = metadata.loc[series]
        family = row["family"]
        styles[series] = {
            "color": series_colors[series],
            "marker": FAMILY_MARKERS.get(family, MARKERS[index % len(MARKERS)]),
            "linestyle": setting_styles[row["setting"]],
        }
    return styles


def draw_no_data(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def with_all_instances(frame: pd.DataFrame) -> pd.DataFrame:
    pooled = frame.copy()
    pooled["dataset"] = "ALL_GRAPHS"
    return pd.concat([pooled, frame], ignore_index=True)


def style_group_axis(axis: plt.Axes, x: np.ndarray, labels: list[str]) -> None:
    axis.set_xticks(x, labels)
    for index, (tick, label) in enumerate(zip(axis.get_xticklabels(), labels)):
        if label == "All instances":
            tick.set_fontweight("bold")
            if index + 1 < len(labels):
                separator = (float(x[index]) + float(x[index + 1])) / 2
                axis.axvline(
                    separator,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.6,
                    zorder=1,
                )


def grouped_boxplots(
    axis: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    groups: list[object],
    labels: list[str],
    value_column: str,
    order: list[str],
    styles: dict[str, dict[str, object]],
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = min(0.22, 0.76 / max(1, len(order)))
    for index, series in enumerate(order):
        offset = (index - (len(order) - 1) / 2) * width
        data = [
            frame.loc[
                frame[group_column].eq(group) & frame["series"].eq(series),
                value_column,
            ].dropna().to_numpy(dtype=float)
            for group in groups
        ]
        valid = [
            (position, values)
            for position, values in zip(x + offset, data)
            if len(values)
        ]
        if not valid:
            continue
        positions, values = zip(*valid)
        color = styles[series]["color"]
        boxplot = axis.boxplot(
            values,
            positions=positions,
            widths=width * 0.82,
            patch_artist=True,
            manage_ticks=False,
            whis=(0, 100),
            showfliers=False,
            showmeans=True,
            meanline=True,
            meanprops={"color": "black", "linestyle": ":", "linewidth": 1.3},
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.55)
        boxplot["boxes"][0].set_label(series)

    axis.plot([], [], color="black", linewidth=1.3, label="Median")
    axis.plot([], [], color="black", linestyle=":", linewidth=1.3, label="Mean")
    style_group_axis(axis, x, labels)


def categorical_boxplots(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    order: list[str],
    styles: dict[str, dict[str, object]],
) -> None:
    valid = [
        (
            series,
            frame.loc[frame["series"].eq(series), value_column]
            .dropna()
            .to_numpy(dtype=float),
        )
        for series in order
    ]
    valid = [(series, values) for series, values in valid if len(values)]
    if not valid:
        draw_no_data(axis, "No values available")
        return
    labels, values = zip(*valid)
    colors = [styles[label]["color"] for label in labels]
    boxplot = axis.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linestyle": ":", "linewidth": 1.3},
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.55)
    repeated_colors = [color for color in colors for _ in range(2)]
    for whisker, color in zip(boxplot["whiskers"], repeated_colors):
        whisker.set_color(color)
    for cap, color in zip(boxplot["caps"], repeated_colors):
        cap.set_color(color)
    axis.plot([], [], color="black", linewidth=1.3, label="Median")
    axis.plot([], [], color="black", linestyle=":", linewidth=1.3, label="Mean")
    axis.tick_params(axis="x", rotation=30)


def add_sdp_baselines(
    axis: plt.Axes,
    results: pd.DataFrame,
    depth: float | None = None,
    statistic: str = "mean",
) -> None:
    baseline_specs = [
        ("L1M1", "L1 SDP Result", "-"),
        ("L2M1", "L2M1 SDP Result", "-."),
        ("L2M2", "L2 SDP Result", ":"),
    ]
    subset = results
    if depth is not None:
        subset = subset.loc[subset["qaoa_depth"].eq(depth)]

    for family, label, linestyle in baseline_specs:
        values = subset.loc[subset["family"].eq(family), "sdp_state_ratio"].dropna()
        if values.empty:
            continue
        baseline = values.median() if statistic == "median" else values.mean()
        axis.axhline(
            baseline,
            color="black",
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )


def plot_overview(
    results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_path: Path,
    title_suffix: str,
    statistic: str,
    show: bool,
) -> None:
    order = series_order(results)
    styles = build_styles(results)
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    statistic_label = SUMMARY_LABELS[statistic]
    fig.suptitle(
        plot_title(
            f"Final QAOA result comparison: {statistic_label.lower()}",
            title_suffix,
        ),
        fontsize=18,
    )

    by_depth = summaries["by_depth"]
    depths = sorted(results["qaoa_depth"].dropna().unique())
    if statistic == "boxplot":
        grouped_boxplots(
            axes[0, 0],
            results,
            "qaoa_depth",
            depths,
            [str(int(value)) for value in depths],
            "final_ratio",
            order,
            styles,
        )
    else:
        for series in order:
            values = by_depth.loc[by_depth["series"].eq(series)].sort_values(
                "qaoa_depth"
            )
            if statistic == "range":
                axes[0, 0].fill_between(
                    values["qaoa_depth"],
                    values["min_final_ratio"],
                    values["max_final_ratio"],
                    color=styles[series]["color"],
                    alpha=0.16,
                )
                axes[0, 0].plot(
                    values["qaoa_depth"],
                    values["min_final_ratio"],
                    color=styles[series]["color"],
                    linewidth=1.2,
                )
                axes[0, 0].plot(
                    values["qaoa_depth"],
                    values["max_final_ratio"],
                    color=styles[series]["color"],
                    linewidth=1.2,
                    label=series,
                )
            else:
                axes[0, 0].plot(
                    values["qaoa_depth"],
                    values[f"{statistic}_final_ratio"],
                    linewidth=2,
                    markersize=6,
                    label=series,
                    **styles[series],
                )
        if statistic in {"mean", "median"}:
            add_sdp_baselines(axes[0, 0], results, statistic=statistic)
    axes[0, 0].set_title(
        f"{metric_summary_title(statistic, 'Final ratio')} by QAOA depth"
    )
    axes[0, 0].set_xlabel("QAOA depth p")
    axes[0, 0].set_ylabel("Final approximation ratio")
    if statistic != "boxplot":
        axes[0, 0].set_xticks(depths)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(alpha=0.25)

    datasets = ["ALL_GRAPHS", *sorted(results["dataset"].unique())]
    dataset_labels = [format_dataset_name(name) for name in datasets]
    dataset_results = with_all_instances(results)
    if statistic == "boxplot":
        grouped_boxplots(
            axes[0, 1],
            dataset_results,
            "dataset",
            datasets,
            dataset_labels,
            "final_ratio",
            order,
            styles,
        )
    else:
        dataset_summary = aggregate_results(dataset_results, ["dataset", "series"])
        x = np.arange(len(datasets))
        width = 0.82 / max(len(order), 1)
        for index, series in enumerate(order):
            values = dataset_summary.loc[dataset_summary["series"].eq(series)].set_index(
                "dataset"
            ).reindex(datasets)
            positions = x + (index - (len(order) - 1) / 2) * width
            if statistic == "range":
                lower = values["min_final_ratio"].to_numpy(dtype=float)
                upper = values["max_final_ratio"].to_numpy(dtype=float)
                axes[0, 1].bar(
                    positions,
                    upper - lower,
                    width,
                    bottom=lower,
                    color=styles[series]["color"],
                    edgecolor=styles[series]["color"],
                    alpha=0.45,
                    label=series,
                )
                axes[0, 1].scatter(
                    positions,
                    lower,
                    marker="_",
                    s=80,
                    color=styles[series]["color"],
                    zorder=3,
                )
                axes[0, 1].scatter(
                    positions,
                    upper,
                    marker="_",
                    s=80,
                    color=styles[series]["color"],
                    zorder=3,
                )
            else:
                axes[0, 1].bar(
                    positions,
                    values[f"{statistic}_final_ratio"],
                    width,
                    color=styles[series]["color"],
                    label=series,
                )
        style_group_axis(axes[0, 1], x, dataset_labels)
    axes[0, 1].set_title(
        f"{metric_summary_title(statistic, 'Final ratio')} by graph family"
    )
    axes[0, 1].set_xlabel("Graph family")
    axes[0, 1].set_ylabel("Final approximation ratio")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(axis="y", alpha=0.2)

    overall = summaries["overall"].set_index("series").reindex(order).reset_index()
    available_gain = results.dropna(subset=["gain_over_circuit_initial"])
    if available_gain.empty:
        draw_no_data(axes[1, 0], "No stored Adam update-0 values available")
    elif statistic == "boxplot":
        categorical_boxplots(
            axes[1, 0],
            available_gain,
            "gain_over_circuit_initial",
            order,
            styles,
        )
    else:
        gain_summary = overall.dropna(
            subset=[
                "min_gain_over_circuit_initial"
                if statistic == "range"
                else f"{statistic}_gain_over_circuit_initial"
            ]
        )
        if statistic == "range":
            lower = gain_summary["min_gain_over_circuit_initial"].to_numpy(dtype=float)
            upper = gain_summary["max_gain_over_circuit_initial"].to_numpy(dtype=float)
            axes[1, 0].bar(
                gain_summary["series"],
                upper - lower,
                bottom=lower,
                color=[styles[series]["color"] for series in gain_summary["series"]],
                alpha=0.45,
            )
            positions = np.arange(len(gain_summary))
            axes[1, 0].scatter(positions, lower, marker="_", s=90, color="black", zorder=3)
            axes[1, 0].scatter(positions, upper, marker="_", s=90, color="black", zorder=3)
        else:
            axes[1, 0].bar(
                gain_summary["series"],
                gain_summary[f"{statistic}_gain_over_circuit_initial"],
                color=[styles[series]["color"] for series in gain_summary["series"]],
            )
        axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title(
        metric_summary_title(statistic, "Final gain over Adam update 0")
    )
    axes[1, 0].set_xlabel("Series")
    axes[1, 0].set_ylabel("Approximation-ratio gain")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(axis="y", alpha=0.25)

    runtime = results.dropna(subset=["full_duration_analysis"])
    if runtime.empty:
        draw_no_data(axes[1, 1], "No runtime values available")
    elif statistic in {"range", "boxplot"}:
        if statistic == "boxplot":
            categorical_boxplots(
                axes[1, 1], runtime, "full_duration_analysis", order, styles
            )
        else:
            runtime_summary = overall.dropna(subset=["min_full_duration_seconds"])
            lower = runtime_summary["min_full_duration_seconds"].to_numpy(dtype=float)
            upper = runtime_summary["max_full_duration_seconds"].to_numpy(dtype=float)
            axes[1, 1].bar(
                runtime_summary["series"],
                upper - lower,
                bottom=lower,
                color=[styles[series]["color"] for series in runtime_summary["series"]],
                alpha=0.45,
            )
            positions = np.arange(len(runtime_summary))
            axes[1, 1].scatter(positions, lower, marker="_", s=90, color="black", zorder=3)
            axes[1, 1].scatter(positions, upper, marker="_", s=90, color="black", zorder=3)
            axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].set_title(
            f"{metric_summary_title(statistic, 'Runtime')} per benchmark row"
        )
        axes[1, 1].set_xlabel("Series")
        axes[1, 1].set_ylabel("Full duration (seconds)")
        axes[1, 1].grid(axis="y", alpha=0.25)
    else:
        runtime_summary = overall.dropna(
            subset=[f"{statistic}_full_duration_seconds", f"{statistic}_final_ratio"]
        )
        for _, row in runtime_summary.iterrows():
            style = styles[row["series"]]
            axes[1, 1].scatter(
                row[f"{statistic}_full_duration_seconds"],
                row[f"{statistic}_final_ratio"],
                s=85,
                color=style["color"],
                marker=style["marker"],
            )
            axes[1, 1].annotate(
                row["series"],
                (
                    row[f"{statistic}_full_duration_seconds"],
                    row[f"{statistic}_final_ratio"],
                ),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1, 1].set_title("Quality-runtime trade-off")
        axes[1, 1].set_xlabel(f"{statistic_label} full duration (seconds)")
        axes[1, 1].set_ylabel(f"{statistic_label} final approximation ratio")
        axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1, 1].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.95),
            fontsize=9,
            handlelength=3.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_by_depth(
    results: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    statistic: str,
    show: bool,
) -> None:
    if trajectory.empty:
        return
    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported trajectory statistic: {statistic}")
    order = series_order(results)
    styles = build_styles(results)
    aggregated = (
        trajectory.groupby(["series", "qaoa_depth", "adam_update"], observed=True)[
            "approx_ratio"
        ]
        .agg(
            mean_approx_ratio="mean",
            median_approx_ratio="median",
            rows="size",
        )
        .reset_index()
    )
    statistic_label = statistic.title()
    value_column = f"{statistic}_approx_ratio"
    depths = sorted(aggregated["qaoa_depth"].dropna().unique())
    columns = min(3, len(depths))
    rows = math.ceil(len(depths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.3 * columns, 4.4 * rows), squeeze=False)
    fig.suptitle(
        plot_title(
            f"{statistic_label} stored Adam trajectories by QAOA depth",
            title_suffix,
        ),
        fontsize=18,
    )

    for axis, depth in zip(axes.flat, depths):
        depth_data = aggregated.loc[aggregated["qaoa_depth"].eq(depth)]
        for series in order:
            values = depth_data.loc[depth_data["series"].eq(series)].sort_values("adam_update")
            if values.empty:
                continue
            axis.plot(
                values["adam_update"],
                values[value_column],
                linewidth=2,
                markersize=3.5,
                markevery=max(1, len(values) // 12),
                label=series,
                **styles[series],
            )
        add_sdp_baselines(axis, results, depth, statistic)
        axis.set_title(f"QAOA depth p={int(depth)}")
        axis.set_xlabel("Completed Adam updates (0 = initial point)")
        axis.set_ylabel(f"{statistic_label} approximation ratio")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(depths) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.95),
            fontsize=9,
            handlelength=3.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5, w_pad=1.8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_size_trends(
    results: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_path: Path,
    title_suffix: str,
    statistic: str,
    show: bool,
) -> None:
    order = series_order(results)
    styles = build_styles(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    statistic_label = SUMMARY_LABELS[statistic]
    fig.suptitle(
        plot_title(
            f"Final QAOA quality by graph size: {statistic_label.lower()}",
            title_suffix,
        ),
        fontsize=18,
    )

    for axis, table_name, x_column, title, label in [
        (axes[0], "by_vertices", "n_vertices", "Trend by number of vertices", "Vertices"),
        (axes[1], "by_edges", "n_edges", "Trend by number of edges", "Edges"),
    ]:
        table = summaries[table_name]
        groups = sorted(results[x_column].dropna().unique())
        if statistic == "boxplot":
            grouped_boxplots(
                axis,
                results,
                x_column,
                groups,
                [str(int(value)) for value in groups],
                "final_ratio",
                order,
                styles,
            )
        else:
            for series in order:
                values = table.loc[table["series"].eq(series)].sort_values(x_column)
                if values.empty:
                    continue
                if statistic == "range":
                    axis.fill_between(
                        values[x_column],
                        values["min_final_ratio"],
                        values["max_final_ratio"],
                        color=styles[series]["color"],
                        alpha=0.16,
                    )
                    axis.plot(
                        values[x_column],
                        values["min_final_ratio"],
                        color=styles[series]["color"],
                        linewidth=1.2,
                    )
                    axis.plot(
                        values[x_column],
                        values["max_final_ratio"],
                        color=styles[series]["color"],
                        linewidth=1.2,
                        label=series,
                    )
                else:
                    axis.plot(
                        values[x_column],
                        values[f"{statistic}_final_ratio"],
                        linewidth=2,
                        markersize=5,
                        label=series,
                        **styles[series],
                    )
        axis.set_title(f"{title}: {statistic_label.lower()}")
        axis.set_xlabel(label)
        axis.set_ylabel("Final approximation ratio")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.90),
            fontsize=9,
            handlelength=3.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.82 if statistic == "boxplot" else 0.88))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_learning_rate_comparisons(
    results: pd.DataFrame,
    output_dir: Path,
    title_suffix: str,
    show: bool,
) -> None:
    available = results.dropna(subset=["learning_rate", "qaoa_depth"]).copy()
    if available.empty or available["learning_rate"].nunique() < 2:
        print("Learning-rate plots skipped: fewer than two learning rates are available.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment in sorted(available["experiment"].unique()):
        experiment_results = available.loc[
            available["experiment"].eq(experiment)
        ].copy()
        experiment_dir = output_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)

        experiment_families = sorted(
            experiment_results["family"].unique(),
            key=lambda family: (
                FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER),
                family,
            ),
        )
        for family in experiment_families:
            family_results = experiment_results.loc[
                experiment_results["family"].eq(family)
            ].copy()
            family_dir = experiment_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)

            for initialisation in sorted(
                family_results["initialisation_setting"].unique()
            ):
                subset = family_results.loc[
                    family_results["initialisation_setting"].eq(initialisation)
                ].copy()
                if subset["learning_rate"].nunique() < 2:
                    continue

                grouped = (
                    subset.groupby(
                        ["qaoa_depth", "learning_rate"],
                        observed=True,
                    )
                    .agg(
                        rows=("final_ratio", "size"),
                        instances=("instance_key", "nunique"),
                        mean_final_ratio=("final_ratio", "mean"),
                        median_final_ratio=("final_ratio", "median"),
                        std=("final_ratio", "std"),
                        min_final_ratio=("final_ratio", "min"),
                        max_final_ratio=("final_ratio", "max"),
                        q25_final_ratio=(
                            "final_ratio",
                            lambda values: values.quantile(0.25),
                        ),
                        q75_final_ratio=(
                            "final_ratio",
                            lambda values: values.quantile(0.75),
                        ),
                    )
                    .reset_index()
                )
                grouped.insert(0, "initialisation_setting", initialisation)
                grouped.insert(0, "family", family)
                grouped.insert(0, "experiment", experiment)
                grouped.to_csv(
                    family_dir / f"learning_rate_comparison_{initialisation}.csv",
                    index=False,
                )

                depths = sorted(grouped["qaoa_depth"].unique())
                columns = min(3, len(depths))
                rows = math.ceil(len(depths) / columns)
                color = FAMILY_COLORS.get(family, "#333333")
                for statistic in SUMMARY_ORDER:
                    fig, axes = plt.subplots(
                        rows,
                        columns,
                        figsize=(6.3 * columns, 4.4 * rows),
                        squeeze=False,
                    )
                    learning_rate_title = (
                        "Learning-rate comparison by QAOA depth"
                        f": {SUMMARY_LABELS[statistic].lower()}"
                        f" | {experiment} | {FAMILY_LABELS.get(family, family)}"
                        f" | {format_initialisation_name(initialisation)}"
                    )
                    fig.suptitle(
                        plot_title(learning_rate_title, title_suffix),
                        fontsize=17,
                    )

                    for axis, depth in zip(axes.flat, depths):
                        values = grouped.loc[
                            grouped["qaoa_depth"].eq(depth)
                        ].sort_values("learning_rate")
                        rates = sorted(values["learning_rate"].unique())
                        if statistic == "boxplot":
                            depth_rows = subset.loc[subset["qaoa_depth"].eq(depth)]
                            data = [
                                depth_rows.loc[
                                    depth_rows["learning_rate"].eq(rate),
                                    "final_ratio",
                                ].dropna().to_numpy(dtype=float)
                                for rate in rates
                            ]
                            boxplot = axis.boxplot(
                                data,
                                tick_labels=[f"{rate:g}" for rate in rates],
                                patch_artist=True,
                                whis=(0, 100),
                                showfliers=False,
                                showmeans=True,
                                meanline=True,
                                meanprops={
                                    "color": "black",
                                    "linestyle": ":",
                                    "linewidth": 1.3,
                                },
                                medianprops={"color": "black", "linewidth": 1.3},
                            )
                            for box in boxplot["boxes"]:
                                box.set_facecolor(color)
                                box.set_edgecolor(color)
                                box.set_alpha(0.55)
                            axis.plot([], [], color="black", linewidth=1.3, label="Median")
                            axis.plot(
                                [],
                                [],
                                color="black",
                                linestyle=":",
                                linewidth=1.3,
                                label="Mean",
                            )
                        elif statistic == "range":
                            axis.fill_between(
                                values["learning_rate"],
                                values["min_final_ratio"],
                                values["max_final_ratio"],
                                color=color,
                                alpha=0.2,
                            )
                            axis.plot(values["learning_rate"], values["min_final_ratio"], color=color)
                            axis.plot(
                                values["learning_rate"],
                                values["max_final_ratio"],
                                color=color,
                                label=FAMILY_LABELS.get(family, family),
                            )
                        else:
                            axis.plot(
                                values["learning_rate"],
                                values[f"{statistic}_final_ratio"],
                                color=color,
                                marker="o",
                                linestyle="-",
                                linewidth=2,
                                markersize=6,
                                label=FAMILY_LABELS.get(family, family),
                            )
                        if statistic != "boxplot" and rates and min(rates) > 0:
                            axis.set_xscale("log")
                            axis.set_xticks(rates)
                            axis.set_xticklabels([f"{rate:g}" for rate in rates])
                        axis.set_title(f"QAOA depth p={int(depth)}")
                        axis.set_xlabel("Adam learning rate")
                        axis.set_ylabel("Final approximation ratio")
                        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                        axis.grid(alpha=0.25)

                    for axis in axes.flat[len(depths):]:
                        axis.set_visible(False)

                    handles, labels = axes.flat[0].get_legend_handles_labels()
                    if handles:
                        fig.legend(
                            handles,
                            labels,
                            loc="upper center",
                            ncol=min(4, len(handles)),
                            bbox_to_anchor=(0.5, 0.94),
                            fontsize=9,
                            handlelength=3.8,
                            handletextpad=0.6,
                            columnspacing=1.2,
                        )
                    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=2.5, w_pad=1.8)
                    suffix = SUMMARY_SUFFIXES[statistic]
                    fig.savefig(
                        family_dir
                        / f"learning_rate_comparison_{initialisation}{suffix}.png",
                        dpi=200,
                        bbox_inches="tight",
                    )
                    if show:
                        plt.show()
                    plt.close(fig)


def plot_dataset_detail(
    dataset: str,
    results: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    show: bool,
) -> None:
    subset = results.loc[results["dataset"].eq(dataset)].copy()
    if subset.empty:
        return
    trajectory_subset = trajectory.loc[trajectory["dataset"].eq(dataset)].copy()
    order = series_order(subset)
    styles = build_styles(subset)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    fig.suptitle(
        plot_title(
            f"Final QAOA analysis: {format_dataset_name(dataset)}",
            title_suffix,
        ),
        fontsize=18,
    )

    by_depth = aggregate_results(subset, ["series", "qaoa_depth"])
    for series in order:
        values = by_depth.loc[by_depth["series"].eq(series)].sort_values("qaoa_depth")
        axes[0, 0].plot(
            values["qaoa_depth"],
            values["mean_final_ratio"],
            linewidth=2,
            markersize=6,
            label=series,
            **styles[series],
        )
    axes[0, 0].set_title("Final ratio by QAOA depth")
    axes[0, 0].set_xlabel("QAOA depth p")
    axes[0, 0].set_ylabel("Mean final approximation ratio")
    axes[0, 0].set_xticks(sorted(subset["qaoa_depth"].unique()))
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(alpha=0.25)

    if trajectory_subset.empty:
        draw_no_data(axes[0, 1], "No stored Adam trajectories available")
    else:
        trajectory_mean = (
            trajectory_subset.groupby(["series", "adam_update"], observed=True)[
                "approx_ratio"
            ]
            .mean()
            .rename("mean_approx_ratio")
            .reset_index()
        )
        for series in order:
            values = trajectory_mean.loc[trajectory_mean["series"].eq(series)].sort_values(
                "adam_update"
            )
            if values.empty:
                continue
            axes[0, 1].plot(
                values["adam_update"],
                values["mean_approx_ratio"],
                linewidth=2,
                markersize=3.5,
                markevery=max(1, len(values) // 12),
                label=series,
                **styles[series],
            )
        add_sdp_baselines(axes[0, 1], subset)
        axes[0, 1].set_title("Adam trajectory averaged over depths")
        axes[0, 1].set_xlabel("Completed Adam updates")
        axes[0, 1].set_ylabel("Mean approximation ratio")
        axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0, 1].grid(alpha=0.25)

    gain = by_depth.dropna(subset=["mean_gain_over_circuit_initial"])
    if gain.empty:
        draw_no_data(axes[1, 0], "No stored Adam update-0 values available")
    else:
        for series in order:
            values = gain.loc[gain["series"].eq(series)].sort_values("qaoa_depth")
            if values.empty:
                continue
            axes[1, 0].plot(
                values["qaoa_depth"],
                values["mean_gain_over_circuit_initial"],
                linewidth=2,
                markersize=6,
                label=series,
                **styles[series],
            )
        axes[1, 0].axhline(0, color="black", linewidth=1)
        axes[1, 0].set_title("Final gain over Adam update 0")
        axes[1, 0].set_xlabel("QAOA depth p")
        axes[1, 0].set_ylabel("Mean approximation-ratio gain")
        axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1, 0].grid(alpha=0.25)

    runtime = by_depth.dropna(subset=["mean_full_duration_seconds"])
    if runtime.empty:
        draw_no_data(axes[1, 1], "No runtime values available")
    else:
        for series in order:
            values = runtime.loc[runtime["series"].eq(series)].sort_values("qaoa_depth")
            if values.empty:
                continue
            axes[1, 1].plot(
                values["qaoa_depth"],
                values["mean_full_duration_seconds"],
                linewidth=2,
                markersize=6,
                label=series,
                **styles[series],
            )
        axes[1, 1].set_title("Runtime by QAOA depth")
        axes[1, 1].set_xlabel("QAOA depth p")
        axes[1, 1].set_ylabel("Mean full duration (seconds)")
        axes[1, 1].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    baseline_handles, baseline_labels = axes[0, 1].get_legend_handles_labels()
    for handle, label in zip(baseline_handles, baseline_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.95),
            fontsize=9,
            handlelength=3.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_summaries(results: pd.DataFrame, trajectory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    identity = ["experiment", "setting", "family", "series"]
    return {
        "overall": aggregate_results(results, identity),
        "by_learning_rate": aggregate_results(results, identity + ["learning_rate"]),
        "by_depth": aggregate_results(results, identity + ["qaoa_depth"]),
        "by_dataset": aggregate_results(results, identity + ["dataset"]),
        "by_vertices": aggregate_results(results, identity + ["n_vertices"]),
        "by_edges": aggregate_results(results, identity + ["n_edges"]),
        "trajectory": aggregate_trajectory(trajectory),
        "pairwise": build_pairwise_summary(results),
    }


def row_metrics(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "setting",
        "dataset",
        "family",
        "series",
        "instance_key",
        "pair_key",
        "n_vertices",
        "n_edges",
        "qaoa_depth",
        "adam_iterations",
        "learning_rate",
        "initialisation_setting",
        "qaoa_seed_effective",
        "optimal_value_analysis",
        "final_energy_analysis",
        "final_ratio",
        "circuit_initial_ratio",
        "sdp_state_ratio",
        "gain_over_circuit_initial",
        "relative_gain_over_circuit_initial",
        "gain_over_sdp_state",
        "relative_gain_over_sdp_state",
        "duration_analysis",
        "full_duration_analysis",
        "updates_completed_analysis",
        "was_clipped",
        "source_file",
        "source_row",
    ]
    return results[[column for column in columns if column in results.columns]].copy()


def format_report_table(
    frame: pd.DataFrame,
    columns: list[str],
    percentage_columns: set[str] | None = None,
    percentage_point_columns: set[str] | None = None,
    seconds_columns: set[str] | None = None,
) -> str:
    display = frame[[column for column in columns if column in frame]].copy()
    percentage_columns = percentage_columns or set()
    percentage_point_columns = percentage_point_columns or set()
    seconds_columns = seconds_columns or set()

    def percentage(value: object) -> str:
        return "N/A" if pd.isna(value) else f"{100 * float(value):.4f}%"

    def percentage_points(value: object) -> str:
        return "N/A" if pd.isna(value) else f"{100 * float(value):+.4f} pp"

    def seconds(value: object) -> str:
        return "N/A" if pd.isna(value) else f"{float(value):.4f} s"

    for column in percentage_columns.intersection(display.columns):
        display[column] = display[column].map(percentage)
    for column in percentage_point_columns.intersection(display.columns):
        display[column] = display[column].map(percentage_points)
    for column in seconds_columns.intersection(display.columns):
        display[column] = display[column].map(seconds)
    if "dataset" in display:
        display["dataset"] = display["dataset"].map(format_dataset_name)
    return display.to_string(index=False)


def write_numerical_report(
    output_path: Path,
    summaries: dict[str, pd.DataFrame],
) -> None:
    prefixes = ("mean", "median", "min", "q25", "q75", "max")
    final_columns = [f"{prefix}_final_ratio" for prefix in prefixes]
    initial_columns = [f"{prefix}_circuit_initial_ratio" for prefix in prefixes]
    gain_columns = [f"{prefix}_gain_over_circuit_initial" for prefix in prefixes]
    sdp_gain_columns = [f"{prefix}_gain_over_sdp_state" for prefix in prefixes]
    runtime_columns = [f"{prefix}_full_duration_seconds" for prefix in prefixes]
    identity = ["experiment", "setting", "family", "series"]

    overall = summaries["overall"]
    by_dataset = pd.concat(
        [overall.assign(dataset="ALL_GRAPHS"), summaries["by_dataset"]],
        ignore_index=True,
    )

    lines = [
        "Final QAOA comparison: exact numerical values",
        "==============================================",
        "Ratios are percentages. Gains are percentage points. The box in each",
        "boxplot spans Q25 to Q75; its solid line is the median, its dotted line is",
        "the mean, and its whiskers are the observed minimum and maximum.",
        "",
    ]

    breakdowns = [
        ("OVERALL FINAL QAOA RATIO", overall, identity),
        (
            "FINAL QAOA RATIO BY LEARNING RATE",
            summaries["by_learning_rate"],
            identity + ["learning_rate"],
        ),
        (
            "FINAL QAOA RATIO BY QAOA DEPTH",
            summaries["by_depth"],
            identity + ["qaoa_depth"],
        ),
        (
            "FINAL QAOA RATIO BY GRAPH FAMILY",
            by_dataset,
            ["dataset", *identity],
        ),
        (
            "FINAL QAOA RATIO BY VERTEX COUNT",
            summaries["by_vertices"],
            ["dataset", "n_vertices", *identity],
        ),
        (
            "FINAL QAOA RATIO BY EDGE COUNT",
            summaries["by_edges"],
            ["dataset", "n_edges", *identity],
        ),
    ]
    for title, frame, group_columns in breakdowns:
        lines.extend(
            [
                title,
                format_report_table(
                    frame,
                    [*group_columns, "rows", "instances", *final_columns],
                    percentage_columns=set(final_columns),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "CIRCUIT INITIAL RATIO AND GAIN OVER ADAM UPDATE 0",
            format_report_table(
                overall,
                [
                    *identity,
                    "rows",
                    *initial_columns,
                    *gain_columns,
                    "improved",
                    "tied",
                    "worse",
                    "improvement_rate_over_circuit_initial",
                ],
                percentage_columns={
                    *initial_columns,
                    "improvement_rate_over_circuit_initial",
                },
                percentage_point_columns=set(gain_columns),
            ),
            "",
            "GAIN OVER THE SDP-SUPPLIED STATE",
            format_report_table(
                overall,
                [*identity, "rows", *sdp_gain_columns],
                percentage_point_columns=set(sdp_gain_columns),
            ),
            "",
            "RUNTIME PER BENCHMARK ROW",
            format_report_table(
                overall,
                [*identity, "rows", *runtime_columns],
                seconds_columns=set(runtime_columns),
            ),
            "",
            "PAIRED FINAL-STATE COMPARISONS",
            format_report_table(
                summaries["pairwise"],
                [
                    "dataset",
                    "left_series",
                    "right_series",
                    "matched_rows",
                    "mean_right_minus_left",
                    "median_right_minus_left",
                    "right_wins",
                    "ties",
                    "right_losses",
                    "right_win_rate",
                ],
                percentage_columns={"right_win_rate"},
                percentage_point_columns={
                    "mean_right_minus_left",
                    "median_right_minus_left",
                },
            ),
            "",
            "ADAM TRAJECTORY BY UPDATE",
            format_report_table(
                summaries["trajectory"],
                [
                    "experiment",
                    "setting",
                    "dataset",
                    "family",
                    "series",
                    "qaoa_depth",
                    "adam_update",
                    "rows",
                    "mean_approx_ratio",
                    "median_approx_ratio",
                    "min_approx_ratio",
                    "q25_approx_ratio",
                    "q75_approx_ratio",
                    "max_approx_ratio",
                ],
                percentage_columns={
                    "mean_approx_ratio",
                    "median_approx_ratio",
                    "min_approx_ratio",
                    "q25_approx_ratio",
                    "q75_approx_ratio",
                    "max_approx_ratio",
                },
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_view(
    results: pd.DataFrame,
    trajectory: pd.DataFrame,
    output_dir: Path,
    view_name: str,
    show: bool,
) -> None:
    view_dir = output_dir / view_name
    view_dir.mkdir(parents=True, exist_ok=True)
    summaries = build_summaries(results, trajectory)
    row_metrics(results).to_csv(view_dir / "final_qaoa_row_metrics.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(view_dir / f"final_qaoa_{name}.csv", index=False)
    report_path = view_dir / "final_qaoa_exact_percentages.txt"
    write_numerical_report(report_path, summaries)

    title_suffix = (
        "unclipped instances only"
        if view_name == "unclipped_warm_starts_only"
        else ""
    )
    for statistic in SUMMARY_ORDER:
        suffix = SUMMARY_SUFFIXES[statistic]
        plot_overview(
            results,
            summaries,
            view_dir / f"final_qaoa_quality_overview{suffix}.png",
            title_suffix,
            statistic,
            show,
        )
    plot_trajectory_by_depth(
        results,
        trajectory,
        view_dir / "qaoa_trajectory_by_adam_update_and_depth.png",
        title_suffix,
        "mean",
        show,
    )
    plot_trajectory_by_depth(
        results,
        trajectory,
        view_dir / "qaoa_trajectory_by_adam_update_and_depth_median.png",
        title_suffix,
        "median",
        show,
    )
    for statistic in SUMMARY_ORDER:
        suffix = SUMMARY_SUFFIXES[statistic]
        plot_size_trends(
            results,
            summaries,
            view_dir / f"final_qaoa_quality_by_graph_size{suffix}.png",
            title_suffix,
            statistic,
            show,
        )
    plot_learning_rate_comparisons(
        results,
        view_dir / "by_learning_rate",
        title_suffix,
        show,
    )

    family_dir = view_dir / "by_graph_family"
    family_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(results["dataset"].unique()):
        plot_dataset_detail(
            dataset,
            results,
            trajectory,
            family_dir / f"final_qaoa_analysis_{dataset}.png",
            title_suffix,
            show,
        )

    print(f"\n{view_name.replace('_', ' ').title()}:")
    print(
        summaries["overall"][
            [
                "series",
                "rows",
                "instances",
                "mean_final_ratio",
                "mean_gain_over_circuit_initial",
                "mean_gain_over_sdp_state",
                "improvement_rate_over_circuit_initial",
                "mean_full_duration_seconds",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_final_ratio": lambda value: f"{100 * value:.2f}%",
                "mean_gain_over_circuit_initial": lambda value: (
                    "N/A" if pd.isna(value) else f"{100 * value:+.2f} pp"
                ),
                "mean_gain_over_sdp_state": lambda value: (
                    "N/A" if pd.isna(value) else f"{100 * value:+.2f} pp"
                ),
                "improvement_rate_over_circuit_initial": lambda value: (
                    "N/A" if pd.isna(value) else f"{100 * value:.2f}%"
                ),
                "mean_full_duration_seconds": lambda value: (
                    "N/A" if pd.isna(value) else f"{value:.2f}"
                ),
            },
        )
    )
    print(f"Saved exact numerical report to: {report_path}")
    print(f"Saved figures and CSVs to: {view_dir}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "_final_qaoa_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = discover_latest_files(
        root,
        args.experiment,
        args.setting,
        args.dataset,
        args.family,
    )
    results, coverage = load_results(manifest)
    results, trajectory = extract_trajectory(results)
    manifest.to_csv(output_dir / "final_qaoa_input_manifest.csv", index=False)
    coverage.to_csv(output_dir / "final_qaoa_input_coverage.csv", index=False)

    series_count = results["series"].nunique()
    print(
        f"Loaded {len(results)} benchmark rows from {len(manifest)} CSVs "
        f"across {series_count} distinct series."
    )
    if series_count > 10:
        print(
            "Note: more than ten series were selected. Use repeated --setting, "
            "--experiment, or --family filters for a less crowded comparison."
        )

    views: list[tuple[str, pd.DataFrame]] = []
    if args.clipping in ["both", "included"]:
        views.append(("clipping_included", results))
    if args.clipping in ["both", "unclipped"]:
        unclipped = results.loc[~results["was_clipped"]].copy()
        if unclipped.empty:
            print("Warning: the unclipped-only view contains no rows and was skipped.")
        else:
            views.append(("unclipped_warm_starts_only", unclipped))

    for view_name, view_results in views:
        selected_indices = set(view_results.index)
        view_trajectory = trajectory.loc[
            trajectory["row_index"].isin(selected_indices)
        ].copy()
        write_view(view_results, view_trajectory, output_dir, view_name, args.show)

    print(f"\nSaved input manifest and coverage report to: {output_dir}")


if __name__ == "__main__":
    main()
