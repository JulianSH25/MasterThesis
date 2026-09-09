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
    "tf_176",
    "complete_2to12",
    "cycle_3to12",
    "path_2to12",
    "bipartite_454",
    "planar_clawfree_193",
    "regular_388",
]
DATASET_LABELS = {
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
            std_state_over_exact_optimum=("state_over_exact_optimum", "std"),
            relaxation_ratio_count=("state_over_relaxation", "count"),
            mean_state_over_relaxation=("state_over_relaxation", "mean"),
            median_state_over_relaxation=("state_over_relaxation", "median"),
            std_state_over_relaxation=("state_over_relaxation", "std"),
        )
        .reset_index()
    )


def aggregate_relaxations(paired: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    family_columns = {
        "L1M1": "l1_relaxation_over_exact_optimum",
        "L2M2": "l2_relaxation_over_exact_optimum",
    }
    if "l2m1_relaxation_over_exact_optimum" in paired:
        family_columns["L2M1"] = "l2m1_relaxation_over_exact_optimum"
    for family, column in family_columns.items():
        if column not in paired:
            continue
        part = paired[groups + ["graph_key"]].copy()
        part["family"] = family
        part["relaxation_over_exact_optimum"] = paired[column]
        records.append(part)
    long = pd.concat(records, ignore_index=True)
    return (
        long.groupby(groups + ["family"], observed=True, dropna=False)
        .agg(
            instances=("graph_key", "nunique"),
            ratio_count=("relaxation_over_exact_optimum", "count"),
            mean_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "mean",
            ),
            median_relaxation_over_exact_optimum=(
                "relaxation_over_exact_optimum",
                "median",
            ),
        )
        .reset_index()
    )


def comparison_stats(group: pd.DataFrame) -> pd.Series:
    final_gain = group["l2_rotated_minus_l1_rounded_exact"].dropna()
    rounded_gain = group["l2_rounded_minus_l1_rounded_exact"].dropna()
    rotation_gain = group["l2_rotation_gain_exact"].dropna()
    hierarchy_gain = group["l2_relaxation_gap_reduction"].dropna()
    return pd.Series(
        {
            "instances": group["graph_key"].nunique(),
            "exact_matched_instances": len(final_gain),
            "mean_l2_rotated_minus_l1_rounded_exact": final_gain.mean(),
            "median_l2_rotated_minus_l1_rounded_exact": final_gain.median(),
            "l2_rotated_wins": int((final_gain > TOLERANCE).sum()),
            "ties": int((final_gain.abs() <= TOLERANCE).sum()),
            "l1_rounded_wins": int((final_gain < -TOLERANCE).sum()),
            "l2_rotated_win_rate": (final_gain > TOLERANCE).mean(),
            "mean_l2_rounded_minus_l1_rounded_exact": rounded_gain.mean(),
            "mean_l2_rotation_gain_exact": rotation_gain.mean(),
            "l2_rotation_improvement_rate": (rotation_gain > TOLERANCE).mean(),
            "mean_l2_relaxation_gap_reduction": hierarchy_gain.mean(),
            "l2_tighter_relaxation_rate": (hierarchy_gain > TOLERANCE).mean(),
            "mean_l2_sampled_minus_analytic_expected_bound_exact": group[
                "l2_sampled_minus_analytic_expected_bound_exact"
            ].mean(),
        }
    )


def aggregate_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows = [{"dataset": "ALL_GRAPHS", **comparison_stats(paired).to_dict()}]
    for dataset, group in paired.groupby("dataset", observed=True):
        rows.append({"dataset": dataset, **comparison_stats(group).to_dict()})
    return pd.DataFrame(rows)


def aggregate_runtime(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["dataset", "family"], observed=True)
        .agg(
            instances=("graph_key", "nunique"),
            mean_runtime_seconds=("warm_start_compute_time_seconds", "mean"),
            median_runtime_seconds=("warm_start_compute_time_seconds", "median"),
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
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)


def plot_overview(
    state_by_dataset: pd.DataFrame,
    relaxation_by_dataset: pd.DataFrame,
    paired_summary: pd.DataFrame,
    output_path: Path,
    include_l2m1: bool,
    show: bool,
) -> None:
    datasets = ordered_datasets(state_by_dataset["dataset"])
    labels = [format_dataset_name(name) for name in datasets]
    method_order = METHOD_ORDER.copy()
    if include_l2m1:
        method_order.extend(["L2M1 rounded", "L2M1 rotated"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Lasserre-1 and Lasserre-2 SDP comparison", fontsize=19)

    exact_values = metric_by_dataset(
        state_by_dataset,
        datasets,
        "method",
        method_order,
        "mean_state_over_exact_optimum",
    )
    grouped_bars(
        axes[0, 0],
        exact_values,
        labels,
        METHOD_COLORS,
        "Mean state / exact optimum",
        "Rounded and rotated state quality",
    )

    relaxation_values = metric_by_dataset(
        state_by_dataset,
        datasets,
        "method",
        method_order,
        "mean_state_over_relaxation",
    )
    grouped_bars(
        axes[0, 1],
        relaxation_values,
        labels,
        METHOD_COLORS,
        "Mean state / SDP relaxation",
        "Fraction of the relaxation retained",
    )

    family_order = ["L1M1", "L2M2"] + (["L2M1"] if include_l2m1 else [])
    hierarchy_values = metric_by_dataset(
        relaxation_by_dataset,
        datasets,
        "family",
        family_order,
        "mean_relaxation_over_exact_optimum",
    )
    grouped_bars(
        axes[1, 0],
        hierarchy_values,
        labels,
        FAMILY_COLORS,
        "Mean SDP bound / exact optimum",
        "Relaxation tightness (lower is tighter)",
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=1, linestyle=":")

    paired_dataset = paired_summary.loc[paired_summary["dataset"].ne("ALL_GRAPHS")]
    paired_dataset = paired_dataset.set_index("dataset").reindex(datasets)
    gains = paired_dataset["mean_l2_rotated_minus_l1_rounded_exact"]
    colors = np.where(gains >= 0, METHOD_COLORS["L2 rotated"], METHOD_COLORS["L1 rounded"])
    bars = axes[1, 1].bar(np.arange(len(datasets)), gains, color=colors)
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_xticks(np.arange(len(datasets)), labels)
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_ylabel("L2 rotated minus L1 rounded")
    axes[1, 1].set_title("Mean paired final-state difference")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="y", alpha=0.25)
    win_rates = paired_dataset["l2_rotated_win_rate"]
    axes[1, 1].bar_label(
        bars,
        labels=[
            "" if pd.isna(rate) else f"{100 * rate:.0f}% wins"
            for rate in win_rates
        ],
        padding=4,
        fontsize=8,
    )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=3.0, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def mean_by_x(frame: pd.DataFrame, x_column: str, columns: list[str]) -> pd.DataFrame:
    return frame.groupby(x_column, observed=True)[columns].mean().reset_index().sort_values(x_column)


def line_plot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_column: str,
    series: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
) -> None:
    for label, column, color in series:
        if column not in summary or summary[column].notna().sum() == 0:
            continue
        ax.plot(summary[x_column], summary[column], marker="o", linewidth=2, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("n_", "Number of ").replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)


def plot_dataset_detail(dataset: str, paired: pd.DataFrame, output_path: Path, show: bool) -> None:
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
    by_vertices = mean_by_x(
        subset,
        "n_vertices",
        quality_columns + retention_columns + relaxation_columns,
    )
    by_edges = mean_by_x(subset, "n_edges", gain_columns)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"SDP comparison: {format_dataset_name(dataset)}", fontsize=18)
    state_series = [
        ("L1 rounded", quality_columns[0], METHOD_COLORS["L1 rounded"]),
        ("L2 rounded", quality_columns[1], METHOD_COLORS["L2 rounded"]),
        ("L2 rotated", quality_columns[2], METHOD_COLORS["L2 rotated"]),
    ]
    line_plot(
        axes[0, 0],
        by_vertices,
        "n_vertices",
        state_series,
        "State quality by graph order",
        "Mean state / exact optimum",
    )
    line_plot(
        axes[0, 1],
        by_vertices,
        "n_vertices",
        [
            ("L1 rounded", retention_columns[0], METHOD_COLORS["L1 rounded"]),
            ("L2 rounded", retention_columns[1], METHOD_COLORS["L2 rounded"]),
            ("L2 rotated", retention_columns[2], METHOD_COLORS["L2 rotated"]),
        ],
        "Relaxation retention by graph order",
        "Mean state / SDP relaxation",
    )
    line_plot(
        axes[1, 0],
        by_vertices,
        "n_vertices",
        [
            ("L1 relaxation", relaxation_columns[0], FAMILY_COLORS["L1M1"]),
            ("L2 relaxation", relaxation_columns[1], FAMILY_COLORS["L2M2"]),
        ],
        "Relaxation tightness by graph order",
        "Mean SDP bound / exact optimum",
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=1, linestyle=":")
    line_plot(
        axes[1, 1],
        by_edges,
        "n_edges",
        [
            ("L2 rounded - L1 rounded", gain_columns[0], METHOD_COLORS["L2 rounded"]),
            ("L2 rotated - L1 rounded", gain_columns[1], METHOD_COLORS["L2 rotated"]),
            ("L2 rotation gain", gain_columns[2], "#2A9D55"),
        ],
        "Paired differences by graph size",
        "Mean approximation-ratio difference",
    )
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


def plot_runtime(runtime: pd.DataFrame, output_path: Path, show: bool) -> None:
    datasets = ordered_datasets(runtime["dataset"])
    labels = [format_dataset_name(name) for name in datasets]
    family_order = [family for family in ["L1M1", "L2M1", "L2M2"] if runtime["family"].eq(family).any()]
    values = metric_by_dataset(
        runtime,
        datasets,
        "family",
        family_order,
        "mean_runtime_seconds",
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    grouped_bars(
        ax,
        values,
        labels,
        FAMILY_COLORS,
        "Mean compute time per instance (seconds)",
        "SDP and rounding runtime by graph family",
        percent=False,
    )
    positive_values = runtime["mean_runtime_seconds"].dropna()
    if not positive_values.empty and positive_values.max() / max(positive_values.min(), 1e-12) > 20:
        ax.set_yscale("log")
        ax.set_ylabel("Mean compute time per instance (seconds, log scale)")
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
    relaxation_overall = aggregate_relaxations(paired, [])
    relaxation_by_dataset = aggregate_relaxations(paired, ["dataset"])
    relaxation_by_vertices = aggregate_relaxations(paired, ["dataset", "n_vertices"])
    relaxation_by_edges = aggregate_relaxations(paired, ["dataset", "n_edges"])
    paired_summary = aggregate_paired(paired)
    runtime = aggregate_runtime(results)
    configs = configuration_summary(results)

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
        "sdp_runtime_by_dataset.csv": runtime,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    overview_path = output_dir / "sdp_quality_comparison.png"
    runtime_path = output_dir / "sdp_runtime_comparison.png"
    plot_overview(
        state_by_dataset,
        relaxation_by_dataset,
        paired_summary,
        overview_path,
        args.include_l2m1,
        args.show,
    )
    plot_runtime(runtime, runtime_path, args.show)

    dataset_figure_dir = output_dir / "by_graph_family"
    dataset_figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ordered_datasets(paired["dataset"]):
        plot_dataset_detail(
            dataset,
            paired,
            dataset_figure_dir / f"sdp_comparison_{dataset}.png",
            args.show,
        )

    print(f"Loaded {len(results)} unique cache rows from {len(sources)} summary files.")
    print(f"Matched L1M1/L2M2 graph instances: {len(paired)}")
    print_summary(state_overall, paired_summary)
    report_sanity_checks(paired)
    print(f"\nSaved overview to: {overview_path}")
    print(f"Saved per-family figures to: {dataset_figure_dir}")
    print(f"Saved supporting CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
