from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EXP5_RESULTS = Path(
    "/Users/julian/PycharmProjects/PythonProject/MasterThesis/"
    "QAOA/Results/logs/ADAM/Exp5"
)

OVERWRITE = False
OUTPUT_SUFFIX = "_clipped"

QAOA_CODE_DIR = Path("/Users/julian/PycharmProjects/PythonProject/MasterThesis/QAOA/Code")
if str(QAOA_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(QAOA_CODE_DIR))

from Utils import get_exact_result_from_misc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add exact results, clipping metadata, and Adam approximation-ratio histories."
    )
    parser.add_argument(
        "name_addition",
        help='Exact result_name_suffix to process, e.g. "_Exp5_subexp3_KingAmplified".',
    )
    return parser.parse_args()


def parse_literal(value: Any, default=None):
    if isinstance(value, (list, tuple)):
        return value
    if value is None or pd.isna(value):
        return default
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def adam_energy_history(row: pd.Series, csv_path: Path) -> list[float]:
    history = parse_literal(row.get("adam_energy_history_normalized_json"), default=[])
    if not history:
        return []
    if not isinstance(history, (list, tuple)):
        raise ValueError(f"{csv_path.name}, row {row.name}: Adam energy history is not an array.")

    try:
        values = [float(value) for value in history]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{csv_path.name}, row {row.name}: invalid Adam energy history.") from exc

    completed = pd.to_numeric(row.get("adam_updates_completed"), errors="coerce")
    if pd.notna(completed) and len(values) != int(completed) + 1:
        raise ValueError(
            f"{csv_path.name}, row {row.name}: history has {len(values)} entries, "
            f"but adam_updates_completed={int(completed)} requires {int(completed) + 1}."
        )
    return values


def patch_exact_results(csv_path: Path, *, overwrite: bool = False) -> Path:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    for col in [
        "optimal_result",
        "approx_ratio",
        "approx_ratio_010101",
        "diff. approx. ratio",
        "sdp ws greater",
        "result_before_warm_start_clip",
        "warm_start_clip_value",
        "warm_start_clip_source",
        "adam_approx_ratio_history_json",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    exact_cache = {}
    found = 0
    missing = 0
    clipped = 0
    histories = 0

    for idx, row in df.iterrows():
        energy_history = adam_energy_history(row, csv_path)
        histories += bool(energy_history)

        edges = parse_literal(row.get("edges"))
        weights = parse_literal(row.get("weights"))
        n = row.get("n")
        m = row.get("m")

        if edges is None or pd.isna(n) or pd.isna(m):
            missing += 1
            continue

        if weights is None:
            weights = [1.0] * len(edges)

        n = int(float(n))
        m = int(float(m))

        cache_key = (str(edges), str(weights), n, m)
        if cache_key not in exact_cache:
            exact_cache[cache_key] = get_exact_result_from_misc(
                edges,
                weights,
                n,
                m,
                raise_if_missing=False,
            )

        optimal = exact_cache[cache_key]
        if optimal is None or optimal == 0:
            missing += 1
            continue

        found += 1
        df.at[idx, "optimal_result"] = optimal
        if energy_history:
            df.at[idx, "adam_approx_ratio_history_json"] = json.dumps(
                [energy / optimal for energy in energy_history]
            )

        result = pd.to_numeric(row.get("result"), errors="coerce")
        lasserre_level = pd.to_numeric(row.get("lasserre_level"), errors="coerce")
        clip_value = pd.NA
        clip_source = pd.NA

        initial_qaoa_energy = pd.to_numeric(
            row.get("initial_qaoa_input_energy_normalized"), errors="coerce"
        )
        if parse_bool(row.get("warm_start")) and pd.notna(initial_qaoa_energy):
            clip_value = initial_qaoa_energy
            clip_source = "initial_qaoa_input_energy_normalized"
        elif pd.notna(lasserre_level) and int(lasserre_level) == 2:
            algorithm17_actual_energy = pd.to_numeric(row.get("algorithm17_actual_energy"), errors="coerce")
            if pd.notna(algorithm17_actual_energy):
                clip_value = algorithm17_actual_energy
                clip_source = "algorithm17_actual_energy"
        elif pd.notna(lasserre_level) and int(lasserre_level) == 1:
            initial_sdp_statevector_energy = pd.to_numeric(row.get("initial_sdp_statevector_energy"), errors="coerce")
            if pd.notna(initial_sdp_statevector_energy):
                clip_value = initial_sdp_statevector_energy
                clip_source = "initial_sdp_statevector_energy"

        optimiser_result = energy_history[-1] if energy_history else result
        if pd.isna(result) and pd.notna(optimiser_result):
            result = optimiser_result
            df.at[idx, "result"] = result

        if pd.notna(clip_value):
            df.at[idx, "warm_start_clip_value"] = clip_value
            df.at[idx, "warm_start_clip_source"] = clip_source
            if pd.isna(optimiser_result) or clip_value > optimiser_result:
                df.at[idx, "result_before_warm_start_clip"] = optimiser_result
                result = clip_value if pd.isna(result) else max(float(result), float(clip_value))
                df.at[idx, "result"] = result
                clipped += 1

        if pd.notna(result):
            df.at[idx, "approx_ratio"] = result / optimal

        result_010101 = pd.to_numeric(row.get("result_010101"), errors="coerce")
        if pd.notna(result_010101):
            df.at[idx, "approx_ratio_010101"] = result_010101 / optimal

        approx = pd.to_numeric(df.at[idx, "approx_ratio"], errors="coerce")
        approx_010101 = pd.to_numeric(df.at[idx, "approx_ratio_010101"], errors="coerce")
        if pd.notna(approx) and pd.notna(approx_010101):
            df.at[idx, "diff. approx. ratio"] = round(approx, 6) - round(approx_010101, 6)
            df.at[idx, "sdp ws greater"] = round(approx, 6) >= round(approx_010101, 6)

    output_path = csv_path if overwrite else csv_path.with_name(f"{csv_path.stem}{OUTPUT_SUFFIX}{csv_path.suffix}")
    df.to_csv(output_path, index=False)

    print(
        f"{csv_path.name}: filled exact results for {found}/{len(df)} rows; "
        f"missing {missing}; histories {histories}; clipped {clipped} rows to warm-start energy; "
        f"wrote {output_path}"
    )
    return output_path


def main() -> None:
    args = parse_args()
    name_addition = args.name_addition.strip().lstrip("_")
    subexperiment_dir = EXP5_RESULTS / name_addition
    if not subexperiment_dir.is_dir():
        raise FileNotFoundError(f"Subexperiment directory does not exist: {subexperiment_dir}")

    csv_paths = sorted(
        path
        for path in subexperiment_dir.glob("*/*/*.csv")
        if not path.stem.endswith("_clipped")
    )
    print(f"Postprocessing {len(csv_paths)} CSVs in {subexperiment_dir}.")
    for path in csv_paths:
        patch_exact_results(path, overwrite=OVERWRITE)


if __name__ == "__main__":
    main()
