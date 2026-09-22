#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PRIMARY_FAMILIES = ("L1M1", "L2M2")
DATASET_ORDER = [
    "ALL_GRAPHS",
    "tf_176",
    "complete_2to12",
    "cycle_3to12",
    "path_2to12",
    "bipartite_454",
    "planar_clawfree_193",
    "regular_388",
]
DATASET_LABELS = {
    "ALL_GRAPHS": "All instances",
    "tf_176": "Triangle-free",
    "complete_2to12": "Complete",
    "cycle_3to12": "Cycle",
    "path_2to12": "Path",
    "bipartite_454": "Bipartite",
    "planar_clawfree_193": "Planar claw-free",
    "regular_388": "Regular",
}
DATASET_PATTERNS = {
    "trianglefree": "tf_176",
    "complete_graph": "complete_2to12",
    "cycle_graph": "cycle_3to12",
    "path_graph": "path_2to12",
    "bipartite": "bipartite_454",
    "planar_clawfree": "planar_clawfree_193",
    "regular": "regular_388",
}

METHOD_ORDER = ["L1 rounded", "L2 rounded", "L2 rotated"]
METHOD_COLORS = {
    "L1 rounded": "#2878B5",
    "L2 rounded": "#E28E2C",
    "L2 rotated": "#C43C39",
    "L2M1 rounded": "#777777",
    "L2M1 rotated": "#9467BD",
}
FAMILY_COLORS = {
    "L1M1": "#2878B5",
    "L2M1": "#777777",
    "L2M2": "#C43C39",
}
TOLERANCE = 1e-9
SUMMARY_ORDER = ("mean", "median", "range", "boxplot")
SUMMARY_PREFIXES = {
    "mean": ("mean",),
    "median": ("median",),
    "range": ("min", "max"),
}
SUMMARY_LABEL = {
    "mean": "Mean",
    "median": "Median",
    "range": "Minimum and maximum",
    "boxplot": "Distribution",
}
SUMMARY_FILENAME_SUFFIX = {
    "mean": "",
    "median": "_median",
    "range": "_min_max",
    "boxplot": "_boxplots",
}
STATISTIC_AGGREGATION = {
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "q25": lambda values: values.quantile(0.25),
    "q75": lambda values: values.quantile(0.75),
}


def parse_args() -> argparse.Namespace:
    qaoa_root = Path(__file__).resolve().parents[1]
    default_root = qaoa_root / "Results" / "logs" / "ADAM"
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired L1M1 and L2M2 SDP cache summaries overall and by "
            "graph family, vertex count, and edge count."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Directory containing SDP run directories (default: {default_root})",
    )
    parser.add_argument(
        "--experiment",
        default="FINAL_BENCHMARKS_AMPLIFIED",
        help=(
            "Only use summary paths containing this text. Pass an empty string "
            "to include every summary below --root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/FINAL/_sdp_comparison)",
    )
    parser.add_argument(
        "--include-l2m1",
        action="store_true",
        help="Also display the hybrid L2M1 results when they are available.",
    )
    parser.add_argument(
        "--include-preliminary",
        action="store_true",
        help="Also discover sdp_cache_summary_preliminary.csv files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    return parser.parse_args()


def family_name(level: object, solver_level: object) -> str | None:
    level_number = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
    solver_number = pd.to_numeric(pd.Series([solver_level]), errors="coerce").iloc[0]
    if pd.isna(level_number) or pd.isna(solver_number):
        return None
    return f"L{int(level_number)}M{int(solver_number)}"


def dataset_name(cache_dataset: object) -> str:
    raw = str(cache_dataset).lower().replace("-", "_")
    for pattern, name in DATASET_PATTERNS.items():
        if pattern in raw:
            return name
    return str(cache_dataset).split("__filehash_", 1)[0]


def format_dataset_name(name: str) -> str:
    return DATASET_LABELS.get(name, name.replace("_", " ").title())


def ordered_datasets(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    known = [name for name in DATASET_ORDER if name in present]
    return known + sorted(present.difference(known))


def discover_summary_files(
    root: Path, experiment: str, include_preliminary: bool
) -> list[Path]:
    pattern = "sdp_cache_summary*.csv" if include_preliminary else "sdp_cache_summary.csv"
    paths = [path for path in root.rglob(pattern) if path.is_file()]
    if experiment:
        paths = [path for path in paths if experiment in str(path)]
    return sorted(paths)


def load_results(
    root: Path,
    experiment: str,
    include_preliminary: bool,
    include_l2m1: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = discover_summary_files(root, experiment, include_preliminary)
    if not paths:
        raise FileNotFoundError(
            f"No matching SDP cache summaries found below {root} "
            f"for experiment filter {experiment!r}."
        )

    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    required = {
        "cache_dataset",
        "graph_index",
        "lasserre_level",
        "initial_solver_level_M",
        "sdp_objective_value",
        "rounded_energy_over_sdp_objective",
    }

    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            print(f"Warning: skipping {path}; missing columns: {', '.join(missing)}")
            continue
        if frame.empty:
            print(f"Warning: skipping empty summary {path}.")
            continue

        frame = frame.copy()
        frame["family"] = [
            family_name(level, solver_level)
            for level, solver_level in zip(
                frame["lasserre_level"], frame["initial_solver_level_M"]
            )
        ]
        allowed = {"L1M1", "L2M2"}
        if include_l2m1:
            allowed.add("L2M1")
        frame = frame.loc[frame["family"].isin(allowed)].copy()
        if frame.empty:
            continue

        frame["dataset"] = frame["cache_dataset"].map(dataset_name)
        graph_hash = frame.get("canonical_graph_hash", pd.Series(index=frame.index, dtype=object))
        fallback_hash = frame.get("graph_hash", pd.Series(index=frame.index, dtype=object))
        frame["graph_key"] = graph_hash.where(graph_hash.notna(), fallback_hash)
        missing_key = frame["graph_key"].isna() | frame["graph_key"].astype(str).eq("")
        frame.loc[missing_key, "graph_key"] = (
            "index:" + frame.loc[missing_key, "graph_index"].astype(str)
        )
        frame["source_file"] = str(path)
        frame["source_mtime_ns"] = path.stat().st_mtime_ns
        source_rows.append(
            {
                "source_file": str(path),
                "rows": len(frame),
                "families": ",".join(sorted(frame["family"].dropna().unique())),
                "datasets": ",".join(sorted(frame["dataset"].dropna().unique())),
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError("The discovered summaries contained no usable L1M1/L2M2 rows.")

    results = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "graph_index",
        "n_vertices",
        "n_edges",
        "sdp_seed",
        "gp_rounding_seed",
        "algorithm17_seed",
        "algorithm17_beta_star",
        "algorithm17_beta_optimisation_time_seconds",
        "sdp_objective_value",
        "rounded_solution_energy",
        "algorithm17_actual_energy",
        "algorithm17_analytic_F_value",
        "algorithm17_lower_bound_energy",
        "rounded_energy_over_sdp_objective",
        "algorithm17_energy_over_sdp_objective",
        "algorithm17_lower_bound_over_sdp_objective",
        "warm_start_compute_time_seconds",
        "exact_comparison_normalisation_factor",
        "exact_optimal_energy",
        "rounded_energy_over_exact_optimum",
        "algorithm17_energy_over_exact_optimum",
        "algorithm17_lower_bound_over_exact_optimum",
    ]
    for column in numeric_columns:
        if column in results:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    if "cache_created_at" in results:
        results["cache_created_at_parsed"] = pd.to_datetime(
            results["cache_created_at"], errors="coerce", utc=True
        )
    else:
        results["cache_created_at_parsed"] = pd.NaT

    exact_match = results.get(
        "exact_result_match_status", pd.Series("matched", index=results.index)
    ).astype(str).str.lower().eq("matched")
    for column in [
        "exact_optimal_energy",
        "rounded_energy_over_exact_optimum",
        "algorithm17_energy_over_exact_optimum",
        "algorithm17_lower_bound_over_exact_optimum",
    ]:
        if column in results:
            results.loc[~exact_match, column] = np.nan

    factor = results.get(
        "exact_comparison_normalisation_factor",
        pd.Series(1.0, index=results.index),
    ).replace(0, np.nan)
    normalized_objective = results["sdp_objective_value"] / factor
    results["sdp_objective_normalized"] = normalized_objective
    results["relaxation_over_exact_optimum"] = (
        normalized_objective / results.get("exact_optimal_energy")
    )

    sort_columns = ["cache_created_at_parsed", "source_mtime_ns", "source_file"]
    results = results.sort_values(sort_columns)
    duplicate_mask = results.duplicated(
        ["dataset", "graph_key", "family"], keep=False
    )
    if duplicate_mask.any():
        duplicate_groups = results.loc[
            duplicate_mask, ["dataset", "graph_key", "family"]
        ].drop_duplicates()
        print(
            f"Warning: {len(duplicate_groups)} instance/family combinations occur "
            "more than once; keeping the newest cache row."
        )
    results = results.drop_duplicates(
        ["dataset", "graph_key", "family"], keep="last"
    ).reset_index(drop=True)
    return results, pd.DataFrame(source_rows)


def prefix_family(frame: pd.DataFrame, family: str, prefix: str) -> pd.DataFrame:
    selected = frame.loc[frame["family"].eq(family)].copy()
    identifiers = ["dataset", "graph_key"]
    metadata = ["graph_index", "n_vertices", "n_edges", "cache_dataset"]
    values = [
        "sdp_seed",
        "gp_rounding_seed",
        "algorithm17_seed",
        "algorithm17_beta_mode",
        "algorithm17_beta_star",
        "algorithm17_beta_optimisation_time_seconds",
        "sdp_solver_mode",
        "sdp_scs_eps",
        "sdp_scs_max_iters",
        "sdp_objective_value",
        "sdp_objective_normalized",
        "relaxation_over_exact_optimum",
        "rounded_solution_energy",
        "algorithm17_actual_energy",
        "algorithm17_analytic_F_value",
        "algorithm17_lower_bound_energy",
        "rounded_energy_over_sdp_objective",
        "algorithm17_energy_over_sdp_objective",
        "algorithm17_lower_bound_over_sdp_objective",
        "exact_optimal_energy",
        "rounded_energy_over_exact_optimum",
        "algorithm17_energy_over_exact_optimum",
        "algorithm17_lower_bound_over_exact_optimum",
        "warm_start_compute_time_seconds",
        "source_file",
    ]
    columns = identifiers + [column for column in metadata + values if column in selected]
    selected = selected[columns]
    rename = {
        column: f"{prefix}_{column}"
        for column in selected.columns
        if column not in identifiers + metadata
    }
    return selected.rename(columns=rename)


def build_paired_results(
    results: pd.DataFrame, include_l2m1: bool
) -> pd.DataFrame:
    l1 = prefix_family(results, "L1M1", "l1")
    l2 = prefix_family(results, "L2M2", "l2")
    paired = l1.merge(
        l2,
        on=["dataset", "graph_key"],
        how="inner",
        suffixes=("", "_l2"),
        validate="one_to_one",
    )
    for column in ["graph_index", "n_vertices", "n_edges", "cache_dataset"]:
        duplicate = f"{column}_l2"
        if duplicate in paired:
            disagreement = paired[column].astype(str).ne(paired[duplicate].astype(str))
            if disagreement.any():
                raise ValueError(f"Paired L1/L2 rows disagree in {column}.")
            paired = paired.drop(columns=duplicate)

    paired["l2_rounded_minus_l1_rounded_exact"] = (
        paired["l2_rounded_energy_over_exact_optimum"]
        - paired["l1_rounded_energy_over_exact_optimum"]
    )
    paired["l2_rotated_minus_l1_rounded_exact"] = (
        paired["l2_algorithm17_energy_over_exact_optimum"]
        - paired["l1_rounded_energy_over_exact_optimum"]
    )
    paired["l2_rotation_gain_exact"] = (
        paired["l2_algorithm17_energy_over_exact_optimum"]
        - paired["l2_rounded_energy_over_exact_optimum"]
    )
    paired["l2_rotation_gain_relaxation"] = (
        paired["l2_algorithm17_energy_over_sdp_objective"]
        - paired["l2_rounded_energy_over_sdp_objective"]
    )
    paired["l2_relaxation_gap_reduction"] = (
        paired["l1_relaxation_over_exact_optimum"]
        - paired["l2_relaxation_over_exact_optimum"]
    )
    paired["l2_sampled_minus_analytic_expected_bound_exact"] = (
        paired["l2_algorithm17_energy_over_exact_optimum"]
        - paired["l2_algorithm17_lower_bound_over_exact_optimum"]
    )

    if include_l2m1 and results["family"].eq("L2M1").any():
        l2m1 = prefix_family(results, "L2M1", "l2m1")
        paired = paired.merge(
            l2m1,
            on=["dataset", "graph_key"],
            how="left",
            suffixes=("", "_l2m1"),
            validate="one_to_one",
        )
        paired = paired.drop(
            columns=[
                column
                for column in [
                    "graph_index_l2m1",
                    "n_vertices_l2m1",
                    "n_edges_l2m1",
                    "cache_dataset_l2m1",
                ]
                if column in paired
            ]
        )
    return paired


def build_state_long(paired: pd.DataFrame, include_l2m1: bool) -> pd.DataFrame:
    methods = [
        (
            "L1 rounded",
            "l1_rounded_energy_over_exact_optimum",
            "l1_rounded_energy_over_sdp_objective",
        ),
        (
            "L2 rounded",
            "l2_rounded_energy_over_exact_optimum",
            "l2_rounded_energy_over_sdp_objective",
        ),
        (
            "L2 rotated",
            "l2_algorithm17_energy_over_exact_optimum",
            "l2_algorithm17_energy_over_sdp_objective",
        ),
    ]
    if include_l2m1 and "l2m1_rounded_energy_over_exact_optimum" in paired:
        methods.extend(
            [
                (
                    "L2M1 rounded",
                    "l2m1_rounded_energy_over_exact_optimum",
                    "l2m1_rounded_energy_over_sdp_objective",
                ),
                (
                    "L2M1 rotated",
                    "l2m1_algorithm17_energy_over_exact_optimum",
                    "l2m1_algorithm17_energy_over_sdp_objective",
                ),
            ]
        )

    records = []
    identity = ["dataset", "graph_key", "graph_index", "n_vertices", "n_edges"]
    for method, exact_column, relaxation_column in methods:
        if exact_column not in paired or relaxation_column not in paired:
            continue
        part = paired[identity].copy()
        part["instance_key"] = part["dataset"].astype(str) + "|" + part["graph_key"].astype(str)
        part["method"] = method
        part["state_over_exact_optimum"] = paired[exact_column]
        part["state_over_relaxation"] = paired[relaxation_column]
        records.append(part)
    result = pd.concat(records, ignore_index=True)
    method_order = [method[0] for method in methods]
    result["method"] = pd.Categorical(
        result["method"], categories=method_order, ordered=True
    )
    return result


def aggregate_state(state_long: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        state_long.groupby(groups + ["method"], observed=True, dropna=False)
        .agg(
            instances=("instance_key", "nunique"),
            exact_ratio_count=("state_over_exact_optimum", "count"),
            mean_state_over_exact_optimum=("state_over_exact_optimum", "mean"),
            median_state_over_exact_optimum=("state_over_exact_optimum", "median"),
            min_state_over_exact_optimum=("state_over_exact_optimum", "min"),
            max_state_over_exact_optimum=("state_over_exact_optimum", "max"),
            q25_state_over_exact_optimum=(
                "state_over_exact_optimum",
                lambda values: values.quantile(0.25),
            ),
            q75_state_over_exact_optimum=(
                "state_over_exact_optimum",
                lambda values: values.quantile(0.75),
            ),
            std_state_over_exact_optimum=("state_over_exact_optimum", "std"),
            relaxation_ratio_count=("state_over_relaxation", "count"),
            mean_state_over_relaxation=("state_over_relaxation", "mean"),
            median_state_over_relaxation=("state_over_relaxation", "median"),
            min_state_over_relaxation=("state_over_relaxation", "min"),
            max_state_over_relaxation=("state_over_relaxation", "max"),
            q25_state_over_relaxation=(
                "state_over_relaxation",
                lambda values: values.quantile(0.25),
            ),
            q75_state_over_relaxation=(
                "state_over_relaxation",
                lambda values: values.quantile(0.75),
            ),
            std_state_over_relaxation=("state_over_relaxation", "std"),
        )
        .reset_index()
    )


def build_relaxation_long(
    paired: pd.DataFrame, include_l2m1: bool
) -> pd.DataFrame:
    records = []
    family_columns = {
        "L1M1": "l1_relaxation_over_exact_optimum",
        "L2M2": "l2_relaxation_over_exact_optimum",
    }
    if include_l2m1 and "l2m1_relaxation_over_exact_optimum" in paired:
        family_columns["L2M1"] = "l2m1_relaxation_over_exact_optimum"
    for family, column in family_columns.items():
        if column not in paired:
            continue
        part = paired[
            ["dataset", "graph_key", "graph_index", "n_vertices", "n_edges"]
        ].copy()
        part["instance_key"] = (
            part["dataset"].astype(str) + "|" + part["graph_key"].astype(str)
        )
        part["family"] = family
        part["relaxation_over_exact_optimum"] = paired[column]
        records.append(part)
    return pd.concat(records, ignore_index=True)


def aggregate_relaxations(
    relaxation_long: pd.DataFrame, groups: list[str]
) -> pd.DataFrame:
    return (
        relaxation_long.groupby(groups + ["family"], observed=True, dropna=False)
        .agg(
            instances=("instance_key", "nunique"),
            ratio_count=("relaxation_over_exact_optimum", "count"),
            mean_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "mean",
            ),
            median_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "median",
            ),
            min_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "min",
            ),
            max_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "max",
            ),
            q25_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                lambda values: values.quantile(0.25),
            ),
            q75_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
    )


def distribution_stats(values: pd.Series, stem: str) -> dict[str, float]:
    values = values.dropna()
    return {
        f"mean_{stem}": values.mean(),
        f"median_{stem}": values.median(),
        f"min_{stem}": values.min(),
        f"max_{stem}": values.max(),
        f"q25_{stem}": values.quantile(0.25),
        f"q75_{stem}": values.quantile(0.75),
    }


def comparison_stats(group: pd.DataFrame) -> pd.Series:
    final_gain = group["l2_rotated_minus_l1_rounded_exact"].dropna()
    rounded_gain = group["l2_rounded_minus_l1_rounded_exact"].dropna()
    rotation_gain = group["l2_rotation_gain_exact"].dropna()
    hierarchy_gain = group["l2_relaxation_gap_reduction"].dropna()
    sampled_minus_bound = group[
        "l2_sampled_minus_analytic_expected_bound_exact"
    ].dropna()
    result: dict[str, float | int] = {
        "instances": group["graph_key"].nunique(),
        "exact_matched_instances": len(final_gain),
        "l2_rotated_wins": int((final_gain > TOLERANCE).sum()),
        "ties": int((final_gain.abs() <= TOLERANCE).sum()),
        "l1_rounded_wins": int((final_gain < -TOLERANCE).sum()),
        "l2_rotated_win_rate": (final_gain > TOLERANCE).mean(),
        "l2_rotation_improvement_rate": (rotation_gain > TOLERANCE).mean(),
        "l2_tighter_relaxation_rate": (hierarchy_gain > TOLERANCE).mean(),
    }
    for values, stem in [
        (final_gain, "l2_rotated_minus_l1_rounded_exact"),
        (rounded_gain, "l2_rounded_minus_l1_rounded_exact"),
        (rotation_gain, "l2_rotation_gain_exact"),
        (hierarchy_gain, "l2_relaxation_gap_reduction"),
        (
            sampled_minus_bound,
            "l2_sampled_minus_analytic_expected_bound_exact",
        ),
    ]:
        result.update(distribution_stats(values, stem))
    return pd.Series(result)


def aggregate_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows = [{"dataset": "ALL_GRAPHS", **comparison_stats(paired).to_dict()}]
    for dataset, group in paired.groupby("dataset", observed=True):
        rows.append({"dataset": dataset, **comparison_stats(group).to_dict()})
    return pd.DataFrame(rows)


def aggregate_runtime(results: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        results.groupby(groups + ["family"], observed=True, dropna=False)
        .agg(
            instances=("graph_key", "nunique"),
            mean_runtime_seconds=("warm_start_compute_time_seconds", "mean"),
            median_runtime_seconds=("warm_start_compute_time_seconds", "median"),
            min_runtime_seconds=("warm_start_compute_time_seconds", "min"),
            max_runtime_seconds=("warm_start_compute_time_seconds", "max"),
            q25_runtime_seconds=(
                "warm_start_compute_time_seconds",
                lambda values: values.quantile(0.25),
            ),
            q75_runtime_seconds=(
                "warm_start_compute_time_seconds",
                lambda values: values.quantile(0.75),
            ),
            total_runtime_seconds=("warm_start_compute_time_seconds", "sum"),
        )
        .reset_index()
    )


def configuration_summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "sdp_seed",
        "gp_rounding_seed",
        "algorithm17_seed",
        "algorithm17_beta_mode",
        "sdp_solver_mode",
        "sdp_scs_eps",
        "sdp_scs_max_iters",
    ]
    columns = [column for column in columns if column in results]
    return results[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)


def metric_by_dataset(
    summary: pd.DataFrame,
    dataset_order: list[str],
    category_column: str,
    category_order: list[str],
    value_column: str,
) -> dict[str, pd.Series]:
    return {
        category: (
            summary.loc[summary[category_column].eq(category)]
            .set_index("dataset")
            .reindex(dataset_order)[value_column]
        )
        for category in category_order
        if summary[category_column].eq(category).any()
    }


def style_group_axis(ax: plt.Axes, x: np.ndarray, labels: list[str]) -> None:
    ax.set_xticks(x, labels)
    for index, (tick, label) in enumerate(zip(ax.get_xticklabels(), labels)):
        if label == DATASET_LABELS["ALL_GRAPHS"]:
            tick.set_fontweight("bold")
            if index + 1 < len(labels):
                separator = (float(x[index]) + float(x[index + 1])) / 2
                ax.axvline(
                    separator,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.6,
                    zorder=1,
                )
    ax.tick_params(axis="x", rotation=25)


def grouped_bars(
    ax: plt.Axes,
    values: dict[str, pd.Series],
    labels: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
    percent: bool = True,
) -> None:
    x = np.arange(len(labels))
    count = max(1, len(values))
    width = min(0.24, 0.8 / count)
    for index, (name, series) in enumerate(values.items()):
        offset = (index - (count - 1) / 2) * width
        ax.bar(x + offset, series.to_numpy(dtype=float), width, label=name, color=colors[name])
    style_group_axis(ax, x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)


def grouped_interval_bars(
    ax: plt.Axes,
    lower_values: dict[str, pd.Series],
    upper_values: dict[str, pd.Series],
    labels: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
    percent: bool = True,
) -> None:
    x = np.arange(len(labels))
    categories = [name for name in lower_values if name in upper_values]
    count = max(1, len(categories))
    width = min(0.24, 0.8 / count)
    for index, name in enumerate(categories):
        offset = (index - (count - 1) / 2) * width
        positions = x + offset
        lower = lower_values[name].to_numpy(dtype=float)
        upper = upper_values[name].to_numpy(dtype=float)
        ax.bar(
            positions,
            upper - lower,
            width,
            bottom=lower,
            label=name,
            color=colors[name],
            alpha=0.45,
            edgecolor=colors[name],
            linewidth=1.2,
        )
        ax.scatter(positions, lower, marker="_", s=90, color=colors[name], zorder=3)
        ax.scatter(positions, upper, marker="_", s=90, color=colors[name], zorder=3)
    style_group_axis(ax, x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)


def grouped_boxplots(
    ax: plt.Axes,
    frame: pd.DataFrame,
    group_column: str,
    groups: list[object],
    category_column: str,
    categories: list[str],
    value_column: str,
    labels: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
    percent: bool = True,
) -> None:
    x = np.arange(len(groups), dtype=float)
    present_categories = [
        category
        for category in categories
        if frame[category_column].eq(category).any()
    ]
    count = max(1, len(present_categories))
    width = min(0.22, 0.72 / count)

    for index, category in enumerate(present_categories):
        offset = (index - (count - 1) / 2) * width
        data = [
            frame.loc[
                frame[group_column].eq(group)
                & frame[category_column].eq(category),
                value_column,
            ].dropna().to_numpy(dtype=float)
            for group in groups
        ]
        valid = [(position, values) for position, values in zip(x + offset, data) if len(values)]
        if not valid:
            continue
        positions, values = zip(*valid)
        boxplot = ax.boxplot(
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
            whiskerprops={"color": colors[category], "linewidth": 1.0},
            capprops={"color": colors[category], "linewidth": 1.0},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(colors[category])
            box.set_edgecolor(colors[category])
            box.set_alpha(0.55)
        boxplot["boxes"][0].set_label(category)

    ax.plot([], [], color="black", linewidth=1.3, label="Median")
    ax.plot([], [], color="black", linestyle=":", linewidth=1.3, label="Mean")

    style_group_axis(ax, x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)


def plot_aggregate_summary_bars(
    ax: plt.Axes,
    summary: str,
    frame: pd.DataFrame,
    datasets: list[str],
    category_column: str,
    category_order: list[str],
    value_stem: str,
    labels: list[str],
    colors: dict[str, str],
    ylabel: str,
    title: str,
    percent: bool = True,
) -> None:
    prefixes = SUMMARY_PREFIXES[summary]
    values = [
        metric_by_dataset(
            frame,
            datasets,
            category_column,
            category_order,
            f"{prefix}_{value_stem}",
        )
        for prefix in prefixes
    ]
    if len(prefixes) == 1:
        grouped_bars(
            ax,
            values[0],
            labels,
            colors,
            ylabel,
            title,
            percent=percent,
        )
    else:
        grouped_interval_bars(
            ax,
            values[0],
            values[1],
            labels,
            colors,
            ylabel,
            title,
            percent=percent,
        )


def with_all_instances(frame: pd.DataFrame) -> pd.DataFrame:
    all_instances = frame.copy()
    all_instances["dataset"] = "ALL_GRAPHS"
    return pd.concat([all_instances, frame], ignore_index=True)


def annotate_win_rates(
    ax: plt.Axes,
    bars: object,
    win_rates: pd.Series,
) -> None:
    for bar, rate in zip(bars, win_rates):
        if pd.isna(rate):
            continue
        endpoint = float(bar.get_y() + bar.get_height())
        if np.isclose(endpoint, 0.0):
            offset = -7
            vertical_alignment = "top"
        elif endpoint < 0:
            offset = 5
            vertical_alignment = "bottom"
        else:
            offset = -5
            vertical_alignment = "top"
        ax.annotate(
            f"{100 * rate:.0f}% wins",
            (bar.get_x() + bar.get_width() / 2, endpoint),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )


def plot_overview(
    state_by_dataset: pd.DataFrame,
    relaxation_by_dataset: pd.DataFrame,
    paired_summary: pd.DataFrame,
    state_long: pd.DataFrame,
    relaxation_long: pd.DataFrame,
    paired: pd.DataFrame,
    state_output_path: Path,
    relaxation_output_path: Path,
    paired_output_path: Path,
    include_l2m1: bool,
    summary: str,
    show: bool,
) -> None:
    datasets = ordered_datasets(state_by_dataset["dataset"])
    labels = [format_dataset_name(name) for name in datasets]
    method_order = METHOD_ORDER.copy()
    if include_l2m1:
        method_order.extend(["L2M1 rounded", "L2M1 rotated"])

    label = SUMMARY_LABEL[summary]
    family_order = ["L1M1", "L2M2"] + (["L2M1"] if include_l2m1 else [])

    state_fig, state_axes = plt.subplots(1, 2, figsize=(18, 6.7))
    state_fig.suptitle(
        f"Lasserre-1 and Lasserre-2 rounded-state comparison: {label.lower()}",
        fontsize=19,
    )
    relaxation_fig, relaxation_ax = plt.subplots(figsize=(11.5, 6.7))
    relaxation_fig.suptitle(
        f"Lasserre-1 and Lasserre-2 relaxation tightness: {label.lower()}",
        fontsize=18,
    )
    paired_fig, paired_ax = plt.subplots(figsize=(11.5, 6.7))
    paired_fig.suptitle(
        f"Lasserre-2 rotated minus Lasserre-1 rounded: {label.lower()}",
        fontsize=18,
    )

    if summary == "boxplot":
        state_box = with_all_instances(state_long)
        relaxation_box = with_all_instances(relaxation_long)
        paired_box = with_all_instances(paired)
        grouped_boxplots(
            state_axes[0],
            state_box,
            "dataset",
            datasets,
            "method",
            method_order,
            "state_over_exact_optimum",
            labels,
            METHOD_COLORS,
            "State / exact optimum",
            "Rounded and rotated state quality",
        )
        grouped_boxplots(
            state_axes[1],
            state_box,
            "dataset",
            datasets,
            "method",
            method_order,
            "state_over_relaxation",
            labels,
            METHOD_COLORS,
            "State / SDP relaxation",
            "Fraction of the relaxation retained",
        )
        grouped_boxplots(
            relaxation_ax,
            relaxation_box,
            "dataset",
            datasets,
            "family",
            family_order,
            "relaxation_over_exact_optimum",
            labels,
            FAMILY_COLORS,
            "SDP bound / exact optimum",
            "Relaxation tightness (lower is tighter)",
        )
        grouped_boxplots(
            paired_ax,
            paired_box.assign(comparison="L2 rotated - L1 rounded"),
            "dataset",
            datasets,
            "comparison",
            ["L2 rotated - L1 rounded"],
            "l2_rotated_minus_l1_rounded_exact",
            labels,
            {"L2 rotated - L1 rounded": METHOD_COLORS["L2 rotated"]},
            "L2 rotated minus L1 rounded",
            "Paired final-state difference",
        )
    else:
        plot_aggregate_summary_bars(
            state_axes[0],
            summary,
            state_by_dataset,
            datasets,
            "method",
            method_order,
            "state_over_exact_optimum",
            labels,
            METHOD_COLORS,
            f"{label} state / exact optimum",
            "Rounded and rotated state quality",
        )
        plot_aggregate_summary_bars(
            state_axes[1],
            summary,
            state_by_dataset,
            datasets,
            "method",
            method_order,
            "state_over_relaxation",
            labels,
            METHOD_COLORS,
            f"{label} state / SDP relaxation",
            "Fraction of the relaxation retained",
        )
        plot_aggregate_summary_bars(
            relaxation_ax,
            summary,
            relaxation_by_dataset,
            datasets,
            "family",
            family_order,
            "relaxation_over_exact_optimum",
            labels,
            FAMILY_COLORS,
            f"{label} SDP bound / exact optimum",
            "Relaxation tightness (lower is tighter)",
        )

        paired_dataset = paired_summary.set_index("dataset").reindex(datasets)
        prefixes = SUMMARY_PREFIXES[summary]
        x = np.arange(len(datasets))
        if len(prefixes) == 1:
            gains = paired_dataset[
                f"{prefixes[0]}_l2_rotated_minus_l1_rounded_exact"
            ]
            colors = np.where(
                gains >= 0,
                METHOD_COLORS["L2 rotated"],
                METHOD_COLORS["L1 rounded"],
            )
            bars = paired_ax.bar(x, gains, color=colors)
        else:
            lower = paired_dataset[
                f"{prefixes[0]}_l2_rotated_minus_l1_rounded_exact"
            ]
            upper = paired_dataset[
                f"{prefixes[1]}_l2_rotated_minus_l1_rounded_exact"
            ]
            midpoint = (lower + upper) / 2
            colors = np.where(
                midpoint >= 0,
                METHOD_COLORS["L2 rotated"],
                METHOD_COLORS["L1 rounded"],
            )
            bars = paired_ax.bar(
                x,
                upper - lower,
                bottom=lower,
                color=colors,
                alpha=0.5,
                edgecolor=colors,
            )
            paired_ax.scatter(x, lower, marker="_", s=100, color=colors, zorder=3)
            paired_ax.scatter(x, upper, marker="_", s=100, color=colors, zorder=3)

        style_group_axis(paired_ax, x, labels)
        paired_ax.set_ylabel("L2 rotated minus L1 rounded")
        paired_ax.set_title(f"{label} paired final-state difference")
        win_rates = paired_dataset["l2_rotated_win_rate"]
        annotate_win_rates(paired_ax, bars, win_rates)

    relaxation_ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    paired_ax.axhline(0, color="black", linewidth=1)
    paired_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    paired_ax.grid(axis="y", alpha=0.25)
    paired_ax.margins(y=0.12)

    state_handles, state_legend_labels = state_axes[0].get_legend_handles_labels()
    if state_handles:
        state_fig.legend(
            state_handles,
            state_legend_labels,
            loc="upper center",
            ncol=min(5, len(state_handles)),
            bbox_to_anchor=(0.5, 0.91),
        )
    state_fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=2.0)
    state_fig.savefig(state_output_path, dpi=200, bbox_inches="tight")

    relaxation_handles, relaxation_labels = relaxation_ax.get_legend_handles_labels()
    if relaxation_handles:
        relaxation_fig.legend(
            relaxation_handles,
            relaxation_labels,
            loc="upper center",
            ncol=min(5, len(relaxation_handles)),
            bbox_to_anchor=(0.5, 0.91),
        )
    relaxation_fig.tight_layout(rect=(0, 0, 1, 0.82))
    relaxation_fig.savefig(relaxation_output_path, dpi=200, bbox_inches="tight")

    paired_handles, paired_labels = paired_ax.get_legend_handles_labels()
    if paired_handles:
        paired_fig.legend(
            paired_handles,
            paired_labels,
            loc="upper center",
            ncol=min(4, len(paired_handles)),
            bbox_to_anchor=(0.5, 0.91),
        )
        paired_rect_top = 0.82
    else:
        paired_rect_top = 0.90
    paired_fig.tight_layout(rect=(0, 0, 1, paired_rect_top))
    paired_fig.savefig(paired_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(state_fig)
    plt.close(relaxation_fig)
    plt.close(paired_fig)


def aggregate_by_x(
    frame: pd.DataFrame,
    x_column: str,
    columns: list[str],
    statistic_prefix: str,
) -> pd.DataFrame:
    aggregation = STATISTIC_AGGREGATION[statistic_prefix]
    return (
        frame.groupby(x_column, observed=True)[columns]
        .agg(aggregation)
        .reset_index()
        .sort_values(x_column)
    )


def line_plot_summary(
    ax: plt.Axes,
    summaries: dict[str, pd.DataFrame],
    summary: str,
    x_column: str,
    series: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
) -> None:
    prefixes = SUMMARY_PREFIXES[summary]
    for series_label, column, color in series:
        if len(prefixes) == 1:
            values = summaries[prefixes[0]]
            if column not in values or values[column].notna().sum() == 0:
                continue
            ax.plot(
                values[x_column],
                values[column],
                marker="o",
                linewidth=2,
                label=series_label,
                color=color,
            )
            continue

        lower = summaries[prefixes[0]]
        upper = summaries[prefixes[1]]
        if column not in lower or column not in upper:
            continue
        x = lower[x_column].to_numpy(dtype=float)
        lower_values = lower[column].to_numpy(dtype=float)
        upper_values = upper[column].to_numpy(dtype=float)
        ax.fill_between(x, lower_values, upper_values, color=color, alpha=0.14)
        ax.plot(x, lower_values, color=color, linestyle="--", linewidth=1.2)
        ax.plot(x, upper_values, color=color, linewidth=1.8, label=series_label)
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("n_", "Number of ").replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)


def wide_metric_long(
    frame: pd.DataFrame,
    identity_columns: list[str],
    mappings: list[tuple[str, str]],
    category_column: str,
    value_column: str,
) -> pd.DataFrame:
    records = []
    for category, column in mappings:
        part = frame[identity_columns].copy()
        part[category_column] = category
        part[value_column] = frame[column]
        records.append(part)
    return pd.concat(records, ignore_index=True)


def plot_dataset_detail(
    dataset: str,
    paired: pd.DataFrame,
    output_path: Path,
    summary: str,
    show: bool,
) -> None:
    subset = paired.loc[paired["dataset"].eq(dataset)].copy()
    quality_columns = [
        "l1_rounded_energy_over_exact_optimum",
        "l2_rounded_energy_over_exact_optimum",
        "l2_algorithm17_energy_over_exact_optimum",
    ]
    retention_columns = [
        "l1_rounded_energy_over_sdp_objective",
        "l2_rounded_energy_over_sdp_objective",
        "l2_algorithm17_energy_over_sdp_objective",
    ]
    relaxation_columns = [
        "l1_relaxation_over_exact_optimum",
        "l2_relaxation_over_exact_optimum",
    ]
    gain_columns = [
        "l2_rounded_minus_l1_rounded_exact",
        "l2_rotated_minus_l1_rounded_exact",
        "l2_rotation_gain_exact",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    label = SUMMARY_LABEL[summary]
    fig.suptitle(
        f"SDP comparison: {format_dataset_name(dataset)} ({label.lower()})",
        fontsize=18,
    )
    state_series = [
        ("L1 rounded", quality_columns[0], METHOD_COLORS["L1 rounded"]),
        ("L2 rounded", quality_columns[1], METHOD_COLORS["L2 rounded"]),
        ("L2 rotated", quality_columns[2], METHOD_COLORS["L2 rotated"]),
    ]
    retention_series = [
        ("L1 rounded", retention_columns[0], METHOD_COLORS["L1 rounded"]),
        ("L2 rounded", retention_columns[1], METHOD_COLORS["L2 rounded"]),
        ("L2 rotated", retention_columns[2], METHOD_COLORS["L2 rotated"]),
    ]
    relaxation_series = [
        ("L1 relaxation", relaxation_columns[0], FAMILY_COLORS["L1M1"]),
        ("L2 relaxation", relaxation_columns[1], FAMILY_COLORS["L2M2"]),
    ]
    gain_series = [
        ("L2 rounded - L1 rounded", gain_columns[0], METHOD_COLORS["L2 rounded"]),
        ("L2 rotated - L1 rounded", gain_columns[1], METHOD_COLORS["L2 rotated"]),
        ("L2 rotation gain", gain_columns[2], "#2A9D55"),
    ]

    if summary == "boxplot":
        vertex_groups = sorted(subset["n_vertices"].dropna().unique())
        edge_groups = sorted(subset["n_edges"].dropna().unique())
        vertex_labels = [str(int(value)) for value in vertex_groups]
        edge_labels = [str(int(value)) for value in edge_groups]
        identity = ["n_vertices", "n_edges", "graph_key"]
        state_long = wide_metric_long(
            subset,
            identity,
            [(label, column) for label, column, _ in state_series],
            "method",
            "value",
        )
        retention_long = wide_metric_long(
            subset,
            identity,
            [(label, column) for label, column, _ in retention_series],
            "method",
            "value",
        )
        relaxation_long = wide_metric_long(
            subset,
            identity,
            [(label, column) for label, column, _ in relaxation_series],
            "method",
            "value",
        )
        gain_long = wide_metric_long(
            subset,
            identity,
            [(label, column) for label, column, _ in gain_series],
            "method",
            "value",
        )
        state_colors = {label: color for label, _, color in state_series}
        retention_colors = {label: color for label, _, color in retention_series}
        relaxation_colors = {label: color for label, _, color in relaxation_series}
        gain_colors = {label: color for label, _, color in gain_series}
        grouped_boxplots(
            axes[0, 0], state_long, "n_vertices", vertex_groups, "method",
            list(state_colors), "value", vertex_labels, state_colors,
            "State / exact optimum", "State quality by graph order",
        )
        grouped_boxplots(
            axes[0, 1], retention_long, "n_vertices", vertex_groups, "method",
            list(retention_colors), "value", vertex_labels, retention_colors,
            "State / SDP relaxation", "Relaxation retention by graph order",
        )
        grouped_boxplots(
            axes[1, 0], relaxation_long, "n_vertices", vertex_groups, "method",
            list(relaxation_colors), "value", vertex_labels, relaxation_colors,
            "SDP bound / exact optimum", "Relaxation tightness by graph order",
        )
        grouped_boxplots(
            axes[1, 1], gain_long, "n_edges", edge_groups, "method",
            list(gain_colors), "value", edge_labels, gain_colors,
            "Approximation-ratio difference", "Paired differences by graph size",
        )
    else:
        prefixes = SUMMARY_PREFIXES[summary]
        by_vertices = {
            prefix: aggregate_by_x(
                subset,
                "n_vertices",
                quality_columns + retention_columns + relaxation_columns,
                prefix,
            )
            for prefix in prefixes
        }
        by_edges = {
            prefix: aggregate_by_x(subset, "n_edges", gain_columns, prefix)
            for prefix in prefixes
        }
        line_plot_summary(
            axes[0, 0], by_vertices, summary, "n_vertices", state_series,
            "State quality by graph order", f"{label} state / exact optimum",
        )
        line_plot_summary(
            axes[0, 1], by_vertices, summary, "n_vertices", retention_series,
            "Relaxation retention by graph order", f"{label} state / SDP relaxation",
        )
        line_plot_summary(
            axes[1, 0], by_vertices, summary, "n_vertices", relaxation_series,
            "Relaxation tightness by graph order", f"{label} SDP bound / exact optimum",
        )
        line_plot_summary(
            axes[1, 1], by_edges, summary, "n_edges", gain_series,
            "Paired differences by graph size", f"{label} approximation-ratio difference",
        )

    axes[1, 0].axhline(1.0, color="black", linewidth=1, linestyle=":")
    axes[1, 1].axhline(0, color="black", linewidth=1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
    for ax in axes.flat:
        if ax is not axes[0, 0]:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_runtime(
    runtime: pd.DataFrame,
    raw_results: pd.DataFrame,
    output_path: Path,
    summary: str,
    show: bool,
) -> None:
    datasets = ordered_datasets(runtime["dataset"])
    labels = [format_dataset_name(name) for name in datasets]
    family_order = [family for family in ["L1M1", "L2M1", "L2M2"] if runtime["family"].eq(family).any()]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    label = SUMMARY_LABEL[summary]
    if summary == "boxplot":
        grouped_boxplots(
            ax,
            with_all_instances(raw_results),
            "dataset",
            datasets,
            "family",
            family_order,
            "warm_start_compute_time_seconds",
            labels,
            FAMILY_COLORS,
            "Compute time per instance (seconds)",
            "SDP and rounding runtime by graph family",
            percent=False,
        )
        positive_values = raw_results["warm_start_compute_time_seconds"].dropna()
    else:
        plot_aggregate_summary_bars(
            ax,
            summary,
            runtime,
            datasets,
            "family",
            family_order,
            "runtime_seconds",
            labels,
            FAMILY_COLORS,
            f"{label} compute time per instance (seconds)",
            "SDP and rounding runtime by graph family",
            percent=False,
        )
        positive_values = pd.concat(
            [runtime[f"{prefix}_runtime_seconds"] for prefix in SUMMARY_PREFIXES[summary]]
        ).dropna()
    positive_values = positive_values.loc[positive_values > 0]
    if not positive_values.empty and positive_values.max() / max(positive_values.min(), 1e-12) > 20:
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} compute time per instance (seconds, log scale)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def report_sanity_checks(paired: pd.DataFrame) -> None:
    hierarchy_violation = paired["l2_relaxation_over_exact_optimum"] > (
        paired["l1_relaxation_over_exact_optimum"] + 1e-6
    )
    exact_columns = [
        "l1_rounded_energy_over_exact_optimum",
        "l2_rounded_energy_over_exact_optimum",
        "l2_algorithm17_energy_over_exact_optimum",
    ]
    over_optimum = pd.concat([paired[column] for column in exact_columns], axis=1).gt(1 + 1e-6).any(axis=1)
    print("\nSanity checks:")
    print(f"  L2 relaxation looser than L1 beyond tolerance: {int(hierarchy_violation.sum())}")
    print(f"  Rounded/rotated energy above exact optimum:     {int(over_optimum.sum())}")
    print(
        "  Note: the analytic Algorithm 17 bound is an expectation over the random "
        "rotations, so individual sampled states are not checked against it."
    )


def print_summary(state_overall: pd.DataFrame, paired_summary: pd.DataFrame) -> None:
    display = state_overall.copy()
    for column in ["mean_state_over_exact_optimum", "mean_state_over_relaxation"]:
        display[column] *= 100
    print("\nOverall state quality:")
    print(
        display[
            [
                "method",
                "instances",
                "exact_ratio_count",
                "mean_state_over_exact_optimum",
                "mean_state_over_relaxation",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_state_over_exact_optimum": lambda value: f"{value:.2f}%",
                "mean_state_over_relaxation": lambda value: f"{value:.2f}%",
            },
        )
    )

    overall = paired_summary.loc[paired_summary["dataset"].eq("ALL_GRAPHS")].iloc[0]
    print("\nPaired L2 rotated versus L1 rounded:")
    print(
        f"  Mean difference: {100 * overall['mean_l2_rotated_minus_l1_rounded_exact']:+.2f} pp\n"
        f"  Wins / ties / losses: {int(overall['l2_rotated_wins'])} / "
        f"{int(overall['ties'])} / {int(overall['l1_rounded_wins'])}\n"
        f"  L2 win rate: {100 * overall['l2_rotated_win_rate']:.2f}%\n"
        f"  Mean gain from L2 rotations: {100 * overall['mean_l2_rotation_gain_exact']:+.2f} pp\n"
        f"  L2 tighter-relaxation rate: {100 * overall['l2_tighter_relaxation_rate']:.2f}%"
    )


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
        display["dataset"] = display["dataset"].map(
            lambda value: DATASET_LABELS.get(value, str(value).replace("_", " "))
        )
    return display.to_string(index=False)


def write_numerical_report(
    output_path: Path,
    state_for_overview: pd.DataFrame,
    relaxation_for_overview: pd.DataFrame,
    paired_summary: pd.DataFrame,
    runtime_for_overview: pd.DataFrame,
    state_by_vertices: pd.DataFrame,
    state_by_edges: pd.DataFrame,
    relaxation_by_vertices: pd.DataFrame,
    relaxation_by_edges: pd.DataFrame,
) -> None:
    distribution_prefixes = ("mean", "median", "min", "q25", "q75", "max")
    state_exact = [f"{prefix}_state_over_exact_optimum" for prefix in distribution_prefixes]
    state_relaxation = [f"{prefix}_state_over_relaxation" for prefix in distribution_prefixes]
    relaxation = [
        f"{prefix}_relaxation_over_exact_optimum" for prefix in distribution_prefixes
    ]
    paired_difference = [
        f"{prefix}_l2_rotated_minus_l1_rounded_exact"
        for prefix in distribution_prefixes
    ]
    runtime = [f"{prefix}_runtime_seconds" for prefix in distribution_prefixes]

    lines = [
        "Lasserre SDP comparison: exact numerical values",
        "================================================",
        "Ratios are percentages. Differences are percentage points. The box in each",
        "boxplot spans Q25 to Q75; its solid line is the median, its dotted line is",
        "the mean, and its whiskers are the observed minimum and maximum.",
        "",
        "STATE QUALITY RELATIVE TO THE EXACT OPTIMUM",
        format_report_table(
            state_for_overview,
            ["dataset", "method", "instances", "exact_ratio_count", *state_exact],
            percentage_columns=set(state_exact),
        ),
        "",
        "FRACTION OF THE SDP RELAXATION RETAINED",
        format_report_table(
            state_for_overview,
            [
                "dataset",
                "method",
                "instances",
                "relaxation_ratio_count",
                *state_relaxation,
            ],
            percentage_columns=set(state_relaxation),
        ),
        "",
        "RELAXATION VALUE RELATIVE TO THE EXACT OPTIMUM",
        format_report_table(
            relaxation_for_overview,
            ["dataset", "family", "instances", "ratio_count", *relaxation],
            percentage_columns=set(relaxation),
        ),
        "",
        "PAIRED L2 ROTATED MINUS L1 ROUNDED",
        format_report_table(
            paired_summary,
            [
                "dataset",
                "exact_matched_instances",
                *paired_difference,
                "l2_rotated_wins",
                "ties",
                "l1_rounded_wins",
                "l2_rotated_win_rate",
                "l2_rotation_improvement_rate",
                "l2_tighter_relaxation_rate",
            ],
            percentage_columns={
                "l2_rotated_win_rate",
                "l2_rotation_improvement_rate",
                "l2_tighter_relaxation_rate",
            },
            percentage_point_columns=set(paired_difference),
        ),
        "",
        "SDP RUNTIME",
        format_report_table(
            runtime_for_overview,
            ["dataset", "family", "instances", *runtime, "total_runtime_seconds"],
            seconds_columns={*runtime, "total_runtime_seconds"},
        ),
        "",
        "STATE QUALITY BY VERTEX COUNT",
        format_report_table(
            state_by_vertices,
            [
                "dataset",
                "n_vertices",
                "method",
                "instances",
                *state_exact,
                *state_relaxation,
            ],
            percentage_columns={*state_exact, *state_relaxation},
        ),
        "",
        "STATE QUALITY BY EDGE COUNT",
        format_report_table(
            state_by_edges,
            [
                "dataset",
                "n_edges",
                "method",
                "instances",
                *state_exact,
                *state_relaxation,
            ],
            percentage_columns={*state_exact, *state_relaxation},
        ),
        "",
        "RELAXATION TIGHTNESS BY VERTEX COUNT",
        format_report_table(
            relaxation_by_vertices,
            ["dataset", "n_vertices", "family", "instances", *relaxation],
            percentage_columns=set(relaxation),
        ),
        "",
        "RELAXATION TIGHTNESS BY EDGE COUNT",
        format_report_table(
            relaxation_by_edges,
            ["dataset", "n_edges", "family", "instances", *relaxation],
            percentage_columns=set(relaxation),
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "FINAL" / "_sdp_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results, sources = load_results(
        root,
        args.experiment,
        args.include_preliminary,
        args.include_l2m1,
    )
    paired = build_paired_results(results, args.include_l2m1)
    if paired.empty:
        raise ValueError("No graph instances had both an L1M1 and an L2M2 result.")

    state_long = build_state_long(paired, args.include_l2m1)
    state_overall = aggregate_state(state_long, [])
    state_by_dataset = aggregate_state(state_long, ["dataset"])
    state_by_vertices = aggregate_state(state_long, ["dataset", "n_vertices"])
    state_by_edges = aggregate_state(state_long, ["dataset", "n_edges"])
    relaxation_long = build_relaxation_long(paired, args.include_l2m1)
    relaxation_overall = aggregate_relaxations(relaxation_long, [])
    relaxation_by_dataset = aggregate_relaxations(relaxation_long, ["dataset"])
    relaxation_by_vertices = aggregate_relaxations(
        relaxation_long, ["dataset", "n_vertices"]
    )
    relaxation_by_edges = aggregate_relaxations(
        relaxation_long, ["dataset", "n_edges"]
    )
    paired_summary = aggregate_paired(paired)
    runtime_overall = aggregate_runtime(results, [])
    runtime_by_dataset = aggregate_runtime(results, ["dataset"])
    configs = configuration_summary(results)

    state_for_overview = pd.concat(
        [state_overall.assign(dataset="ALL_GRAPHS"), state_by_dataset],
        ignore_index=True,
    )
    relaxation_for_overview = pd.concat(
        [relaxation_overall.assign(dataset="ALL_GRAPHS"), relaxation_by_dataset],
        ignore_index=True,
    )
    runtime_for_overview = pd.concat(
        [runtime_overall.assign(dataset="ALL_GRAPHS"), runtime_by_dataset],
        ignore_index=True,
    )

    outputs = {
        "sdp_source_files.csv": sources,
        "sdp_configuration_summary.csv": configs,
        "sdp_matched_instances.csv": paired,
        "sdp_state_overall.csv": state_overall,
        "sdp_state_by_dataset.csv": state_by_dataset,
        "sdp_state_by_vertices.csv": state_by_vertices,
        "sdp_state_by_edges.csv": state_by_edges,
        "sdp_relaxation_overall.csv": relaxation_overall,
        "sdp_relaxation_by_dataset.csv": relaxation_by_dataset,
        "sdp_relaxation_by_vertices.csv": relaxation_by_vertices,
        "sdp_relaxation_by_edges.csv": relaxation_by_edges,
        "sdp_paired_comparison.csv": paired_summary,
        "sdp_runtime_overall.csv": runtime_overall,
        "sdp_runtime_by_dataset.csv": runtime_by_dataset,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    report_path = output_dir / "sdp_comparison_exact_percentages.txt"
    write_numerical_report(
        report_path,
        state_for_overview,
        relaxation_for_overview,
        paired_summary,
        runtime_for_overview,
        state_by_vertices,
        state_by_edges,
        relaxation_by_vertices,
        relaxation_by_edges,
    )

    dataset_figure_dir = output_dir / "by_graph_family"
    dataset_figure_dir.mkdir(parents=True, exist_ok=True)
    comparison_paths = []
    runtime_paths = []
    for summary in SUMMARY_ORDER:
        suffix = SUMMARY_FILENAME_SUFFIX[summary]
        state_path = output_dir / f"sdp_quality_comparison{suffix}.png"
        relaxation_path = output_dir / f"sdp_relaxation_tightness{suffix}.png"
        paired_path = output_dir / f"sdp_paired_final_state_difference{suffix}.png"
        runtime_path = output_dir / f"sdp_runtime_comparison{suffix}.png"
        plot_overview(
            state_for_overview,
            relaxation_for_overview,
            paired_summary,
            state_long,
            relaxation_long,
            paired,
            state_path,
            relaxation_path,
            paired_path,
            args.include_l2m1,
            summary,
            args.show,
        )
        plot_runtime(
            runtime_for_overview,
            results,
            runtime_path,
            summary,
            args.show,
        )
        comparison_paths.extend([state_path, relaxation_path, paired_path])
        runtime_paths.append(runtime_path)

        for dataset in ordered_datasets(paired["dataset"]):
            plot_dataset_detail(
                dataset,
                paired,
                dataset_figure_dir / f"sdp_comparison_{dataset}{suffix}.png",
                summary,
                args.show,
            )

    print(f"Loaded {len(results)} unique cache rows from {len(sources)} summary files.")
    print(f"Matched L1M1/L2M2 graph instances: {len(paired)}")
    print_summary(state_overall, paired_summary)
    report_sanity_checks(paired)
    print(f"\nSaved {len(comparison_paths)} SDP comparison figures to: {output_dir}")
    print(f"Saved {len(runtime_paths)} runtime figures to: {output_dir}")
    print(f"Saved per-family figures to: {dataset_figure_dir}")
    print(f"Saved exact numerical report to: {report_path}")
    print(f"Saved supporting CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
