#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SETTING_PATTERN = re.compile(
    r"^(lr(?:001|005|01|05))_(noheuristic|h1_s(?:10|25|50))$"
)
TIMESTAMP_PATTERN = re.compile(r"_(\d{8}_\d{6})_pid")

METHOD_ORDER = ["No heuristic", "Heuristic 10", "Heuristic 25", "Heuristic 50"]
METHOD_LABELS = {
    "noheuristic": "No heuristic",
    "h1_s10": "Heuristic 10",
    "h1_s25": "Heuristic 25",
    "h1_s50": "Heuristic 50",
}
METHOD_COLORS = {
    "No heuristic": "#555555",
    "Heuristic 10": "#2878B5",
    "Heuristic 25": "#E28E2C",
    "Heuristic 50": "#C43C39",
}
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
    default_root = (
        qaoa_root
        / "Results"
        / "logs"
        / "ADAM"
        / "FINAL"
        / "FINAL_BENCHMARKS_STANDARD"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compare the newest QAOA-only no-heuristic, 10, 25, and 50 "
            "heuristic benchmark results."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Organised FINAL_BENCHMARKS_STANDARD directory (default: {default_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/_heuristic_comparison)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures after saving them.",
    )
    return parser.parse_args()


def result_file_sort_key(path: Path) -> tuple[str, int]:
    timestamp_match = TIMESTAMP_PATTERN.search(path.name)
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    return timestamp, path.stat().st_mtime_ns


def discover_latest_files(root: Path) -> list[tuple[Path, str, str, str]]:
    discovered: list[tuple[Path, str, str, str]] = []

    for setting_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        setting_match = SETTING_PATTERN.fullmatch(setting_dir.name)
        if setting_match is None:
            continue

        learning_rate_label, raw_method = setting_match.groups()
        for dataset_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir()):
            candidates = [
                path
                for path in dataset_dir.glob("*QAOA_only.csv")
                if not path.name.endswith("_clipped.csv")
            ]
            if not candidates:
                continue

            latest = max(candidates, key=result_file_sort_key)
            discovered.append(
                (latest, learning_rate_label, METHOD_LABELS[raw_method], dataset_dir.name)
            )

            if len(candidates) > 1:
                print(
                    f"Warning: {setting_dir.name}/{dataset_dir.name} contains "
                    f"{len(candidates)} QAOA-only CSVs; using {latest.name}."
                )

    return discovered


def load_results(root: Path) -> pd.DataFrame:
    files = discover_latest_files(root)
    if not files:
        raise FileNotFoundError(f"No QAOA-only result CSVs found below {root}")

    frames: list[pd.DataFrame] = []
    for path, learning_rate_label, method, dataset in files:
        frame = pd.read_csv(path)
        if "approx_ratio" not in frame.columns:
            print(f"Warning: skipping {path}; it has no approx_ratio column.")
            continue

        frame = frame.copy()
        frame["learning_rate_label"] = learning_rate_label
        frame["method"] = method
        frame["dataset"] = dataset
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("The discovered CSVs contained no usable QAOA-only results.")

    results = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "approx_ratio",
        "learning_rate_adam",
        "p",
        "precision/iterations",
        "qaoa_seed_used",
        "duration_seconds",
        "full duration_seconds",
    ]
    for column in numeric_columns:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=["approx_ratio", "learning_rate_adam", "p"])
    return results


def retain_matched_rows(results: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key_columns = [
        "learning_rate_adam",
        "dataset",
        "hog_graph_index",
        "edges",
        "weights",
        "p",
        "precision/iterations",
        "qaoa_seed_used",
    ]

    keyed = results.copy()
    for column in key_columns:
        if column not in keyed.columns:
            keyed[column] = "<missing>"
        keyed[column] = keyed[column].fillna("<missing>").astype(str)

    sort_columns = [column for column in ["finished_at", "source_file"] if column in keyed]
    if sort_columns:
        keyed = keyed.sort_values(sort_columns)

    keyed = keyed.drop_duplicates(key_columns + ["method"], keep="last")
    pivot = keyed.pivot(index=key_columns, columns="method", values="approx_ratio")
    matched_index = pivot.dropna(subset=METHOD_ORDER).index
    matched_keys = matched_index.to_frame(index=False)
    matched = keyed.merge(matched_keys, on=key_columns, how="inner")

    for column in [
        "learning_rate_adam",
        "p",
        "precision/iterations",
        "qaoa_seed_used",
    ]:
        matched[column] = pd.to_numeric(matched[column], errors="coerce")

    return matched, len(matched_index)


def build_summaries(matched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall = (
        matched.groupby("method", observed=True)
        .agg(
            rows=("approx_ratio", "size"),
            mean_approx_ratio=("approx_ratio", "mean"),
            median_approx_ratio=("approx_ratio", "median"),
            std_approx_ratio=("approx_ratio", "std"),
            min_approx_ratio=("approx_ratio", "min"),
            max_approx_ratio=("approx_ratio", "max"),
            q25_approx_ratio=("approx_ratio", lambda values: values.quantile(0.25)),
            q75_approx_ratio=("approx_ratio", lambda values: values.quantile(0.75)),
            mean_duration_seconds=("duration_seconds", "mean"),
            mean_full_duration_seconds=("full duration_seconds", "mean"),
            median_full_duration_seconds=("full duration_seconds", "median"),
            min_full_duration_seconds=("full duration_seconds", "min"),
            max_full_duration_seconds=("full duration_seconds", "max"),
            q25_full_duration_seconds=(
                "full duration_seconds",
                lambda values: values.quantile(0.25),
            ),
            q75_full_duration_seconds=(
                "full duration_seconds",
                lambda values: values.quantile(0.75),
            ),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )

    dataset_method = (
        matched.groupby(["dataset", "method"], observed=True)
        .agg(
            rows=("approx_ratio", "size"),
            mean_approx_ratio=("approx_ratio", "mean"),
            median_approx_ratio=("approx_ratio", "median"),
            min_approx_ratio=("approx_ratio", "min"),
            max_approx_ratio=("approx_ratio", "max"),
            q25_approx_ratio=("approx_ratio", lambda values: values.quantile(0.25)),
            q75_approx_ratio=("approx_ratio", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    macro_dataset = (
        dataset_method.groupby("method", observed=True)["mean_approx_ratio"]
        .mean()
        .rename("equal_dataset_mean_approx_ratio")
    )
    overall = overall.merge(macro_dataset, on="method", how="left")

    by_learning_rate = (
        matched.groupby(["learning_rate_adam", "method"], observed=True)
        .agg(
            rows=("approx_ratio", "size"),
            mean_approx_ratio=("approx_ratio", "mean"),
            median_approx_ratio=("approx_ratio", "median"),
            min_approx_ratio=("approx_ratio", "min"),
            max_approx_ratio=("approx_ratio", "max"),
            q25_approx_ratio=("approx_ratio", lambda values: values.quantile(0.25)),
            q75_approx_ratio=("approx_ratio", lambda values: values.quantile(0.75)),
            mean_full_duration_seconds=("full duration_seconds", "mean"),
            median_full_duration_seconds=("full duration_seconds", "median"),
            min_full_duration_seconds=("full duration_seconds", "min"),
            max_full_duration_seconds=("full duration_seconds", "max"),
        )
        .reset_index()
    )
    by_depth = (
        matched.groupby(["p", "method"], observed=True)
        .agg(
            rows=("approx_ratio", "size"),
            mean_approx_ratio=("approx_ratio", "mean"),
            median_approx_ratio=("approx_ratio", "median"),
            min_approx_ratio=("approx_ratio", "min"),
            max_approx_ratio=("approx_ratio", "max"),
            q25_approx_ratio=("approx_ratio", lambda values: values.quantile(0.25)),
            q75_approx_ratio=("approx_ratio", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )

    key_columns = [
        "learning_rate_adam",
        "dataset",
        "hog_graph_index",
        "edges",
        "weights",
        "p",
        "precision/iterations",
        "qaoa_seed_used",
    ]
    wide = matched.pivot(index=key_columns, columns="method", values="approx_ratio")
    paired_rows = []
    for method in METHOD_ORDER[1:]:
        difference = wide[method] - wide["No heuristic"]
        paired_rows.append(
            {
                "method": method,
                "matched_rows": len(difference),
                "mean_gain": difference.mean(),
                "median_gain": difference.median(),
                "min_gain": difference.min(),
                "max_gain": difference.max(),
                "q25_gain": difference.quantile(0.25),
                "q75_gain": difference.quantile(0.75),
                "wins": int((difference > 1e-12).sum()),
                "ties": int((difference.abs() <= 1e-12).sum()),
                "losses": int((difference < -1e-12).sum()),
                "win_rate": (difference > 1e-12).mean(),
            }
        )

    return {
        "overall": overall,
        "by_learning_rate": by_learning_rate,
        "by_depth": by_depth,
        "by_dataset": dataset_method,
        "paired_vs_no_heuristic": pd.DataFrame(paired_rows),
    }


def method_series(
    summary: pd.DataFrame,
    x_column: str,
    method: str,
    value_column: str = "mean_approx_ratio",
) -> pd.DataFrame:
    return summary.loc[summary["method"].eq(method)].sort_values(x_column)


def metric_summary_title(statistic: str, metric: str) -> str:
    if statistic == "boxplot":
        return f"{metric} distribution"
    if statistic == "range":
        return f"{metric}: minimum and maximum"
    return f"{SUMMARY_LABELS[statistic]} {metric.lower()}"


def format_dataset_name(name: str) -> str:
    replacements = {
        "bipartite_454": "Bipartite",
        "complete_2to12": "Complete",
        "cycle_3to12": "Cycle",
        "path_2to12": "Path",
        "planar_clawfree_193": "Planar claw-free",
        "regular_388": "Regular",
        "tf_176": "Triangle-free",
    }
    if name == "ALL_GRAPHS":
        return "All instances"
    return replacements.get(name, name.replace("_", " "))


def paired_gain_values(matched: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "learning_rate_adam",
        "dataset",
        "hog_graph_index",
        "edges",
        "weights",
        "p",
        "precision/iterations",
        "qaoa_seed_used",
    ]
    wide = matched.pivot(index=key_columns, columns="method", values="approx_ratio")
    records = []
    for method in METHOD_ORDER[1:]:
        differences = (wide[method] - wide["No heuristic"]).dropna()
        for gain in differences:
            records.append({"method": method, "gain": gain})
    return pd.DataFrame(records)


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
) -> None:
    x = np.arange(len(groups), dtype=float)
    width = min(0.2, 0.76 / max(1, len(METHOD_ORDER)))
    for index, method in enumerate(METHOD_ORDER):
        offset = (index - (len(METHOD_ORDER) - 1) / 2) * width
        data = [
            frame.loc[
                frame[group_column].eq(group) & frame["method"].eq(method),
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
            whiskerprops={"color": METHOD_COLORS[method], "linewidth": 1.0},
            capprops={"color": METHOD_COLORS[method], "linewidth": 1.0},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(METHOD_COLORS[method])
            box.set_edgecolor(METHOD_COLORS[method])
            box.set_alpha(0.55)
        boxplot["boxes"][0].set_label(method)

    axis.plot([], [], color="black", linewidth=1.3, label="Median")
    axis.plot([], [], color="black", linestyle=":", linewidth=1.3, label="Mean")
    style_group_axis(axis, x, labels)


def categorical_boxplots(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
) -> None:
    data = [
        frame.loc[frame["method"].eq(method), value_column].dropna().to_numpy(dtype=float)
        for method in METHOD_ORDER
    ]
    valid = [(method, values) for method, values in zip(METHOD_ORDER, data) if len(values)]
    if not valid:
        return
    labels, values = zip(*valid)
    colors = [METHOD_COLORS[label] for label in labels]
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
    axis.tick_params(axis="x", rotation=20)


def plot_series_summary(
    axis: plt.Axes,
    summary: pd.DataFrame,
    x_column: str,
    statistic: str,
) -> None:
    for method in METHOD_ORDER:
        series = method_series(summary, x_column, method)
        if series.empty:
            continue
        if statistic == "range":
            axis.fill_between(
                series[x_column],
                series["min_approx_ratio"],
                series["max_approx_ratio"],
                color=METHOD_COLORS[method],
                alpha=0.16,
            )
            axis.plot(
                series[x_column],
                series["min_approx_ratio"],
                color=METHOD_COLORS[method],
                linewidth=1.2,
            )
            axis.plot(
                series[x_column],
                series["max_approx_ratio"],
                color=METHOD_COLORS[method],
                linewidth=1.2,
                label=method,
            )
        else:
            axis.plot(
                series[x_column],
                series[f"{statistic}_approx_ratio"],
                marker="o",
                linewidth=2,
                label=method,
                color=METHOD_COLORS[method],
            )


def plot_quality_comparison(
    matched: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_path: Path,
    statistic: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    statistic_label = SUMMARY_LABELS[statistic]
    fig.suptitle(
        f"QAOA-only parameter-selection heuristic comparison: {statistic_label.lower()}",
        fontsize=18,
    )

    by_lr = summaries["by_learning_rate"]
    learning_rates = sorted(matched["learning_rate_adam"].dropna().unique())
    if statistic == "boxplot":
        grouped_boxplots(
            axes[0, 0],
            matched,
            "learning_rate_adam",
            learning_rates,
            [f"{value:g}" for value in learning_rates],
            "approx_ratio",
        )
    else:
        plot_series_summary(axes[0, 0], by_lr, "learning_rate_adam", statistic)
    if statistic != "boxplot":
        axes[0, 0].set_xscale("log")
    axes[0, 0].set_title(
        f"{metric_summary_title(statistic, 'Approximation ratio')} by learning rate"
    )
    axes[0, 0].set_xlabel("Adam learning rate")
    axes[0, 0].set_ylabel("Approximation ratio")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(alpha=0.25)

    by_depth = summaries["by_depth"]
    depths = sorted(matched["p"].dropna().unique())
    if statistic == "boxplot":
        grouped_boxplots(
            axes[0, 1],
            matched,
            "p",
            depths,
            [str(int(value)) for value in depths],
            "approx_ratio",
        )
    else:
        plot_series_summary(axes[0, 1], by_depth, "p", statistic)
    axes[0, 1].set_title(
        f"{metric_summary_title(statistic, 'Approximation ratio')} by QAOA depth"
    )
    axes[0, 1].set_xlabel("QAOA depth p")
    axes[0, 1].set_ylabel("Approximation ratio")
    if statistic != "boxplot":
        axes[0, 1].set_xticks(depths)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(alpha=0.25)

    dataset_order = ["ALL_GRAPHS", *sorted(matched["dataset"].unique())]
    dataset_labels = [format_dataset_name(name) for name in dataset_order]
    x = np.arange(len(dataset_order))
    dataset_results = with_all_instances(matched)
    if statistic == "boxplot":
        grouped_boxplots(
            axes[1, 0],
            dataset_results,
            "dataset",
            dataset_order,
            dataset_labels,
            "approx_ratio",
        )
    else:
        dataset_summary = (
            dataset_results.groupby(["dataset", "method"], observed=True)
            .agg(
                mean_approx_ratio=("approx_ratio", "mean"),
                median_approx_ratio=("approx_ratio", "median"),
                min_approx_ratio=("approx_ratio", "min"),
                max_approx_ratio=("approx_ratio", "max"),
            )
            .reset_index()
        )
        width = 0.2
        for index, method in enumerate(METHOD_ORDER):
            values = dataset_summary.loc[dataset_summary["method"].eq(method)].set_index(
                "dataset"
            ).reindex(dataset_order)
            positions = x + (index - 1.5) * width
            if statistic == "range":
                lower = values["min_approx_ratio"].to_numpy(dtype=float)
                upper = values["max_approx_ratio"].to_numpy(dtype=float)
                axes[1, 0].bar(
                    positions,
                    upper - lower,
                    width,
                    bottom=lower,
                    label=method,
                    color=METHOD_COLORS[method],
                    alpha=0.45,
                    edgecolor=METHOD_COLORS[method],
                )
                axes[1, 0].scatter(
                    positions,
                    lower,
                    marker="_",
                    s=80,
                    color=METHOD_COLORS[method],
                    zorder=3,
                )
                axes[1, 0].scatter(
                    positions,
                    upper,
                    marker="_",
                    s=80,
                    color=METHOD_COLORS[method],
                    zorder=3,
                )
            else:
                axes[1, 0].bar(
                    positions,
                    values[f"{statistic}_approx_ratio"],
                    width,
                    label=method,
                    color=METHOD_COLORS[method],
                )
        style_group_axis(axes[1, 0], x, dataset_labels)
    axes[1, 0].set_title(
        f"{metric_summary_title(statistic, 'Approximation ratio')} by graph family"
    )
    axes[1, 0].set_xlabel("Graph family")
    axes[1, 0].set_ylabel("Approximation ratio")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(axis="y", alpha=0.25)

    paired = summaries["paired_vs_no_heuristic"]
    if statistic == "boxplot":
        categorical_boxplots(axes[1, 1], paired_gain_values(matched), "gain")
    elif statistic == "range":
        positions = np.arange(len(paired))
        lower = paired["min_gain"].to_numpy(dtype=float)
        upper = paired["max_gain"].to_numpy(dtype=float)
        axes[1, 1].bar(
            positions,
            upper - lower,
            bottom=lower,
            color=[METHOD_COLORS[method] for method in paired["method"]],
            alpha=0.45,
        )
        axes[1, 1].scatter(positions, lower, marker="_", s=90, color="black", zorder=3)
        axes[1, 1].scatter(positions, upper, marker="_", s=90, color="black", zorder=3)
        axes[1, 1].set_xticks(positions, paired["method"])
    else:
        axes[1, 1].bar(
            paired["method"],
            paired[f"{statistic}_gain"],
            color=[METHOD_COLORS[method] for method in paired["method"]],
        )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title(
        metric_summary_title(statistic, "Paired gain over no heuristic")
    )
    axes[1, 1].set_xlabel("Parameter-selection method")
    axes[1, 1].set_ylabel("Approximation-ratio gain")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(6, len(handles)),
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_runtime_tradeoff(
    matched: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    output_path: Path,
    statistic: str,
    show: bool,
) -> None:
    overall = summaries["overall"].set_index("method").reindex(METHOD_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    statistic_label = SUMMARY_LABELS[statistic]
    fig.suptitle(
        f"QAOA-only heuristic quality and runtime: {statistic_label.lower()}",
        fontsize=18,
    )

    colors = [METHOD_COLORS[method] for method in overall["method"]]
    if statistic == "boxplot":
        categorical_boxplots(axes[0], matched, "full duration_seconds")
        categorical_boxplots(axes[1], matched, "approx_ratio")
    elif statistic == "range":
        positions = np.arange(len(overall))
        for axis, stem in zip(axes, ["full_duration_seconds", "approx_ratio"]):
            lower = overall[f"min_{stem}"].to_numpy(dtype=float)
            upper = overall[f"max_{stem}"].to_numpy(dtype=float)
            axis.bar(
                positions,
                upper - lower,
                bottom=lower,
                color=colors,
                alpha=0.45,
            )
            axis.scatter(positions, lower, marker="_", s=90, color="black", zorder=3)
            axis.scatter(positions, upper, marker="_", s=90, color="black", zorder=3)
            axis.set_xticks(positions, overall["method"])
            axis.tick_params(axis="x", rotation=20)
    else:
        duration_column = f"{statistic}_full_duration_seconds"
        ratio_column = f"{statistic}_approx_ratio"
        axes[0].bar(overall["method"], overall[duration_column], color=colors)
        for _, row in overall.iterrows():
            axes[1].scatter(
                row[duration_column],
                row[ratio_column],
                s=90,
                color=METHOD_COLORS[row["method"]],
            )
            axes[1].annotate(
                row["method"],
                (row[duration_column], row[ratio_column]),
                xytext=(7, 5),
                textcoords="offset points",
            )

    axes[0].set_title(
        f"{metric_summary_title(statistic, 'Full duration')} per benchmark row"
    )
    axes[0].set_xlabel("Parameter-selection method")
    axes[0].set_ylabel("Duration (seconds)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_title(
        metric_summary_title(statistic, "Approximation ratio")
        if statistic in {"range", "boxplot"}
        else "Quality-runtime trade-off"
    )
    axes[1].set_xlabel(
        "Parameter-selection method"
        if statistic in {"range", "boxplot"}
        else f"{statistic_label} full duration (seconds)"
    )
    axes[1].set_ylabel(f"{statistic_label} approximation ratio")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)

    if statistic == "boxplot":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90))

    fig.tight_layout(rect=(0, 0, 1, 0.82 if statistic == "boxplot" else 0.90))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def print_summary(summaries: dict[str, pd.DataFrame], matched_keys: int) -> None:
    overall = summaries["overall"].copy()
    overall["mean_approx_ratio"] *= 100
    overall["equal_dataset_mean_approx_ratio"] *= 100
    paired = summaries["paired_vs_no_heuristic"].copy()
    paired["mean_gain"] *= 100
    paired["win_rate"] *= 100

    print(f"\nMatched benchmark rows per method: {matched_keys}")
    print("\nOverall comparison:")
    print(
        overall[
            [
                "method",
                "mean_approx_ratio",
                "equal_dataset_mean_approx_ratio",
                "mean_full_duration_seconds",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_approx_ratio": lambda value: f"{value:.2f}%",
                "equal_dataset_mean_approx_ratio": lambda value: f"{value:.2f}%",
                "mean_full_duration_seconds": lambda value: f"{value:.2f}",
            },
        )
    )

    print("\nPaired comparison against no heuristic:")
    print(
        paired[["method", "mean_gain", "wins", "ties", "losses", "win_rate"]]
        .to_string(
            index=False,
            formatters={
                "mean_gain": lambda value: f"{value:+.2f} pp",
                "win_rate": lambda value: f"{value:.2f}%",
            },
        )
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
        display["dataset"] = display["dataset"].map(format_dataset_name)
    return display.to_string(index=False)


def write_numerical_report(
    output_path: Path,
    summaries: dict[str, pd.DataFrame],
    matched_keys: int,
) -> None:
    prefixes = ("mean", "median", "min", "q25", "q75", "max")
    ratio_columns = [f"{prefix}_approx_ratio" for prefix in prefixes]
    runtime_columns = [f"{prefix}_full_duration_seconds" for prefix in prefixes]
    gain_columns = [f"{prefix}_gain" for prefix in prefixes]
    overall = summaries["overall"]
    by_dataset = pd.concat(
        [
            overall[["method", "rows", *ratio_columns]].assign(
                dataset="ALL_GRAPHS"
            ),
            summaries["by_dataset"],
        ],
        ignore_index=True,
    )

    lines = [
        "QAOA-only heuristic comparison: exact numerical values",
        "======================================================",
        f"Matched benchmark rows per method: {matched_keys}",
        "Ratios are percentages. Gains are percentage points. The box in each",
        "boxplot spans Q25 to Q75; its solid line is the median, its dotted line is",
        "the mean, and its whiskers are the observed minimum and maximum.",
        "",
        "OVERALL APPROXIMATION RATIO",
        format_report_table(
            overall,
            [
                "method",
                "rows",
                *ratio_columns,
                "equal_dataset_mean_approx_ratio",
            ],
            percentage_columns={
                *ratio_columns,
                "equal_dataset_mean_approx_ratio",
            },
        ),
        "",
        "APPROXIMATION RATIO BY LEARNING RATE",
        format_report_table(
            summaries["by_learning_rate"],
            ["learning_rate_adam", "method", "rows", *ratio_columns],
            percentage_columns=set(ratio_columns),
        ),
        "",
        "APPROXIMATION RATIO BY QAOA DEPTH",
        format_report_table(
            summaries["by_depth"],
            ["p", "method", "rows", *ratio_columns],
            percentage_columns=set(ratio_columns),
        ),
        "",
        "APPROXIMATION RATIO BY GRAPH FAMILY",
        format_report_table(
            by_dataset,
            ["dataset", "method", "rows", *ratio_columns],
            percentage_columns=set(ratio_columns),
        ),
        "",
        "PAIRED GAIN OVER NO HEURISTIC",
        format_report_table(
            summaries["paired_vs_no_heuristic"],
            [
                "method",
                "matched_rows",
                *gain_columns,
                "wins",
                "ties",
                "losses",
                "win_rate",
            ],
            percentage_columns={"win_rate"},
            percentage_point_columns=set(gain_columns),
        ),
        "",
        "RUNTIME PER BENCHMARK ROW",
        format_report_table(
            overall,
            ["method", "rows", *runtime_columns],
            seconds_columns=set(runtime_columns),
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
        else root / "_heuristic_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(root)
    matched, matched_keys = retain_matched_rows(results)
    if matched.empty:
        raise ValueError(
            "No benchmark rows were shared by all four parameter-selection methods."
        )

    summaries = build_summaries(matched)
    for name, summary in summaries.items():
        summary.to_csv(output_dir / f"qaoa_only_heuristic_{name}.csv", index=False)
    report_path = output_dir / "qaoa_only_heuristic_exact_percentages.txt"
    write_numerical_report(report_path, summaries, matched_keys)

    for statistic in SUMMARY_ORDER:
        suffix = SUMMARY_SUFFIXES[statistic]
        quality_path = output_dir / f"qaoa_only_heuristic_quality_comparison{suffix}.png"
        runtime_path = output_dir / f"qaoa_only_heuristic_runtime_comparison{suffix}.png"
        plot_quality_comparison(matched, summaries, quality_path, statistic, args.show)
        plot_runtime_tradeoff(matched, summaries, runtime_path, statistic, args.show)
    print_summary(summaries, matched_keys)

    print(f"\nSaved {len(SUMMARY_ORDER)} quality comparisons to: {output_dir}")
    print(f"Saved {len(SUMMARY_ORDER)} runtime comparisons to: {output_dir}")
    print(f"Saved exact numerical report to: {report_path}")
    print(f"Saved supporting CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
