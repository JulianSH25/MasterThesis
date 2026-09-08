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
            mean_duration_seconds=("duration_seconds", "mean"),
            mean_full_duration_seconds=("full duration_seconds", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )

    dataset_method = (
        matched.groupby(["dataset", "method"], observed=True)["approx_ratio"]
        .mean()
        .rename("mean_approx_ratio")
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
            mean_full_duration_seconds=("full duration_seconds", "mean"),
        )
        .reset_index()
    )
    by_depth = (
        matched.groupby(["p", "method"], observed=True)
        .agg(rows=("approx_ratio", "size"), mean_approx_ratio=("approx_ratio", "mean"))
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
    return replacements.get(name, name.replace("_", " "))


def plot_quality_comparison(
    summaries: dict[str, pd.DataFrame], output_path: Path, show: bool
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("QAOA-only parameter-selection heuristic comparison", fontsize=18)

    by_lr = summaries["by_learning_rate"]
    for method in METHOD_ORDER:
        series = method_series(by_lr, "learning_rate_adam", method)
        axes[0, 0].plot(
            series["learning_rate_adam"],
            series["mean_approx_ratio"],
            marker="o",
            linewidth=2,
            label=method,
            color=METHOD_COLORS[method],
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Mean approximation ratio by learning rate")
    axes[0, 0].set_xlabel("Adam learning rate")
    axes[0, 0].set_ylabel("Mean approximation ratio")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(alpha=0.25)

    by_depth = summaries["by_depth"]
    for method in METHOD_ORDER:
        series = method_series(by_depth, "p", method)
        axes[0, 1].plot(
            series["p"],
            series["mean_approx_ratio"],
            marker="o",
            linewidth=2,
            label=method,
            color=METHOD_COLORS[method],
        )
    axes[0, 1].set_title("Mean approximation ratio by QAOA depth")
    axes[0, 1].set_xlabel("QAOA depth p")
    axes[0, 1].set_ylabel("Mean approximation ratio")
    axes[0, 1].set_xticks(sorted(by_depth["p"].unique()))
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(alpha=0.25)

    by_dataset = summaries["by_dataset"]
    dataset_order = sorted(by_dataset["dataset"].unique())
    x = np.arange(len(dataset_order))
    width = 0.2
    for index, method in enumerate(METHOD_ORDER):
        values = (
            by_dataset.loc[by_dataset["method"].eq(method)]
            .set_index("dataset")
            .reindex(dataset_order)["mean_approx_ratio"]
        )
        axes[1, 0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=method,
            color=METHOD_COLORS[method],
        )
    axes[1, 0].set_title("Mean approximation ratio by graph family")
    axes[1, 0].set_xlabel("Graph family")
    axes[1, 0].set_ylabel("Mean approximation ratio")
    axes[1, 0].set_xticks(x, [format_dataset_name(name) for name in dataset_order])
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].grid(axis="y", alpha=0.25)

    paired = summaries["paired_vs_no_heuristic"]
    axes[1, 1].bar(
        paired["method"],
        paired["mean_gain"],
        color=[METHOD_COLORS[method] for method in paired["method"]],
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Mean paired gain over no heuristic")
    axes[1, 1].set_xlabel("Parameter-selection method")
    axes[1, 1].set_ylabel("Approximation-ratio gain")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="y", alpha=0.25)
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(
            container,
            labels=[f"{100 * value:.2f} pp" for value in paired["mean_gain"]],
            padding=4,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.5, w_pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_runtime_tradeoff(
    summaries: dict[str, pd.DataFrame], output_path: Path, show: bool
) -> None:
    overall = summaries["overall"].set_index("method").reindex(METHOD_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("QAOA-only heuristic quality and runtime", fontsize=18)

    colors = [METHOD_COLORS[method] for method in overall["method"]]
    axes[0].bar(overall["method"], overall["mean_full_duration_seconds"], color=colors)
    axes[0].set_title("Mean full duration per benchmark row")
    axes[0].set_xlabel("Parameter-selection method")
    axes[0].set_ylabel("Mean duration (seconds)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    for _, row in overall.iterrows():
        axes[1].scatter(
            row["mean_full_duration_seconds"],
            row["mean_approx_ratio"],
            s=90,
            color=METHOD_COLORS[row["method"]],
        )
        axes[1].annotate(
            row["method"],
            (row["mean_full_duration_seconds"], row["mean_approx_ratio"]),
            xytext=(7, 5),
            textcoords="offset points",
        )
    axes[1].set_title("Quality-runtime trade-off")
    axes[1].set_xlabel("Mean full duration (seconds)")
    axes[1].set_ylabel("Mean approximation ratio")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)

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

    quality_path = output_dir / "qaoa_only_heuristic_quality_comparison.png"
    runtime_path = output_dir / "qaoa_only_heuristic_runtime_comparison.png"
    plot_quality_comparison(summaries, quality_path, args.show)
    plot_runtime_tradeoff(summaries, runtime_path, args.show)
    print_summary(summaries, matched_keys)

    print(f"\nSaved quality comparison to: {quality_path}")
    print(f"Saved runtime comparison to: {runtime_path}")
    print(f"Saved supporting CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
