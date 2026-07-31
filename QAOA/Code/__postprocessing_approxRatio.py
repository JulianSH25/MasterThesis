from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EXP5_RESULTS = Path(
    "/Users/julian/PycharmProjects/PythonProject/MasterThesis/"
    "QAOA/Results/logs/ADAM/Exp5"
)

CSV_PATHS = sorted(
    path
    for path in EXP5_RESULTS.glob("*/*/*.csv")
    if not path.stem.endswith("_clipped")
)

OVERWRITE = False
OUTPUT_SUFFIX = "_clipped"

QAOA_CODE_DIR = Path("/Users/julian/PycharmProjects/PythonProject/MasterThesis/QAOA/Code")
if str(QAOA_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(QAOA_CODE_DIR))

from Utils import get_exact_result_from_misc


def parse_literal(value: Any, default=None):
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (list, tuple)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


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
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    exact_cache = {}
    found = 0
    missing = 0
    clipped = 0

    for idx, row in df.iterrows():
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

        result = pd.to_numeric(row.get("result"), errors="coerce")
        lasserre_level = pd.to_numeric(row.get("lasserre_level"), errors="coerce")
        clip_value = pd.NA
        clip_source = pd.NA

        if pd.notna(lasserre_level) and int(lasserre_level) == 2:
            algorithm17_actual_energy = pd.to_numeric(row.get("algorithm17_actual_energy"), errors="coerce")
            if pd.notna(algorithm17_actual_energy):
                clip_value = algorithm17_actual_energy
                clip_source = "algorithm17_actual_energy"
        elif pd.notna(lasserre_level) and int(lasserre_level) == 1:
            initial_sdp_statevector_energy = pd.to_numeric(row.get("initial_sdp_statevector_energy"), errors="coerce")
            if pd.notna(initial_sdp_statevector_energy):
                clip_value = initial_sdp_statevector_energy
                clip_source = "initial_sdp_statevector_energy"

        if pd.notna(clip_value):
            df.at[idx, "warm_start_clip_value"] = clip_value
            df.at[idx, "warm_start_clip_source"] = clip_source
            if pd.isna(result) or clip_value > result:
                df.at[idx, "result_before_warm_start_clip"] = result
                df.at[idx, "result"] = clip_value
                result = clip_value
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
        f"missing {missing}; clipped {clipped} rows to warm-start energy; wrote {output_path}"
    )
    return output_path


patched_paths = [patch_exact_results(path, overwrite=OVERWRITE) for path in CSV_PATHS]
