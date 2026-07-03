#!/usr/bin/env zsh
set -euo pipefail

QAOA_ROOT="${QAOA_ROOT:-/Users/julian/PycharmProjects/PythonProject/MasterThesis/QAOA}"
CONDA_EXE="${CONDA_EXE:-/opt/homebrew/bin/conda}"
CONDA_ENV="${CONDA_ENV:-MasterThesis}"

LOG_DIRS=(
  "/Users/julian/PycharmProjects/PythonProject/MasterThesis/QAOA/Results/logs/ADAM/20260629_121202_pid517551"
  "/Users/julian/PycharmProjects/PythonProject/MasterThesis/QAOA/Results/logs/ADAM/20260629_122532_pid549340"
)

if (( $# > 0 )); then
  LOG_DIRS=("$@")
fi

eval "$("$CONDA_EXE" shell.zsh hook)"
conda activate "$CONDA_ENV"

export QAOA_ROOT
python3 - "${LOG_DIRS[@]}" <<'PY'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import re
import ast
import csv
import math
import os
import json
import tempfile

QAOA_ROOT = Path(os.environ.get("QAOA_ROOT", ".")).expanduser().resolve()
LOG_DIRS = [Path(arg).expanduser().resolve() for arg in sys.argv[1:]]
TZ = ZoneInfo("Europe/Amsterdam")

# Optional. Used only for extracted rerun files.
MAPPING_CSV = QAOA_ROOT / "Code" / "HOG_graphs" / "stress_test_sdp_under_1.000_mapping.csv"

# SCS with eps=1e-3 can undershoot slightly.
SDP_UNDERSHOOT_TOL = 1e-2
STATE_OVERSHOOT_TOL = 1e-6
RESULT_NORMALISATION_FACTOR = 2.0

sys.path.insert(0, str(QAOA_ROOT))
sys.path.insert(0, str(QAOA_ROOT / "Code"))

from Code.Utils import get_exact_result_from_misc
from Code.InstanceGenerator import instance_generator


def load_index_mapping(path):
    if not path.exists():
        return {}

    mapping = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["new_index"])] = int(row["old_index"])

    return mapping


INDEX_MAPPING = load_index_mapping(MAPPING_CSV)


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def first_float(pattern, text):
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return float(m.group(1)) if m else None


def first_int(pattern, text):
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return int(m.group(1)) if m else None


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt(x, digits=4):
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def human_ts(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def human_duration(seconds):
    if seconds is None:
        return ""

    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    if m or h or d:
        parts.append(f"{m}m")
    parts.append(f"{s}s")

    return " ".join(parts)


def extract_benchmark_params(text):
    marker = "Benchmark parameters:"
    start = text.find(marker)
    if start < 0:
        return {}

    brace_start = text.find("{", start)
    if brace_start < 0:
        return {}

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                raw = text[brace_start:i + 1]
                return ast.literal_eval(raw)

    return {}


def write_temp_config(params):
    fd, path = tempfile.mkstemp(prefix="tmp_benchmark_config_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(params, f)
    return path


def get_edges_weights_from_log_context(text, hog_idx, nodes):
    params = extract_benchmark_params(text)

    if not params:
        raise ValueError("Could not parse Benchmark parameters from log.")

    graph_type = params.get("graph_generation_type")
    weighted = bool(params.get("weighted", False))
    random_weights = bool(params.get("random_weights", False))

    if graph_type is None:
        raise ValueError("graph_generation_type missing in Benchmark parameters.")

    graph_type_lower = str(graph_type).lower()

    if graph_type_lower == "hog":
        if hog_idx is None:
            raise ValueError("HOG graph selected but hog_idx is missing.")
        generator_n = hog_idx
    else:
        if nodes is None:
            raise ValueError(f"{graph_type} graph selected but node count is missing.")
        generator_n = nodes

    old_config = os.environ.get("BENCHMARK_CONFIG_FILE")
    tmp_config = write_temp_config(params)
    old_cwd = Path.cwd()

    try:
        os.environ["BENCHMARK_CONFIG_FILE"] = tmp_config
        os.chdir(QAOA_ROOT)

        edges, weights = instance_generator(
            graph_type,
            generator_n,
            weighted=weighted,
            random_weights=random_weights,
        )

    finally:
        os.chdir(old_cwd)

        if old_config is None:
            os.environ.pop("BENCHMARK_CONFIG_FILE", None)
        else:
            os.environ["BENCHMARK_CONFIG_FILE"] = old_config

        try:
            Path(tmp_config).unlink()
        except Exception:
            pass

    return edges, weights, graph_type, weighted, random_weights, generator_n


def graph_node_count(edges):
    return len({vertex for edge in edges for vertex in edge})


def process_log_dir(log_dir):
    rows = []

    for log_path in sorted(log_dir.glob("sdp_n*_rep*_seed*.log"), key=natural_key):
        text = log_path.read_text(errors="replace")

        file_n = first_int(r"sdp_n(\d+)_", log_path.name)
        repeat = first_int(r"_rep(\d+)_", log_path.name)
        hog_idx = first_int(r"Loaded HOG graph index\s+(\d+)", text)

        if hog_idx is None:
            hog_idx = file_n

        original_hog_idx = INDEX_MAPPING.get(hog_idx, hog_idx)
        nodes = first_int(r"Running QAOA for n=(\d+)\s+nodes", text)

        edge_count = first_int(
            r"Running QAOA for n=\d+\s+nodes,\s*m=(\d+)\s+edges",
            text,
        )

        if edge_count is None:
            edge_count = first_int(
                r"Loaded HOG graph index\s+\d+\s+with\s+(\d+)\s+edges",
                text,
            )

        rounded_energy = first_float(
            r"Energy of rounded cut/product state:\s*([-+]?[0-9]+(?:\.[0-9]+)?)",
            text,
        )

        alg17_lower = first_float(
            r"Algorithm 17 lower-bound energy:\s*([-+]?[0-9]+(?:\.[0-9]+)?)",
            text,
        )

        alg17_actual = first_float(
            r"Algorithm 17 actual entangled-state energy:\s*([-+]?[0-9]+(?:\.[0-9]+)?)",
            text,
        )

        sdp_objective = first_float(
            r"SDP objective value:\s*([-+]?[0-9]+(?:\.[0-9]+)?)",
            text,
        )

        cut_edges = first_int(
            r"(\d+)\s+in cut out of a total of\s+\d+\s+edges",
            text,
        )

        cut_total_edges = first_int(
            r"\d+\s+in cut out of a total of\s+(\d+)\s+edges",
            text,
        )

        if edge_count is None:
            edge_count = cut_total_edges

        start_ts = first_float(
            r"Instance generation.*?finished at stardate\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )

        duration_seconds = first_float(
            r"Warm-start:\s*SDP solve \+ rounding finished in\s*([0-9]+(?:\.[0-9]+)?)\s*seconds",
            text,
        )

        exact_result = None
        exact_error = ""
        parsed_num_nodes = None
        parsed_num_edges = None
        graph_type = None
        generator_n = None
        weighted = None
        random_weights = None

        try:
            edges, weights, graph_type, weighted, random_weights, generator_n = get_edges_weights_from_log_context(
                text=text,
                hog_idx=hog_idx,
                nodes=nodes,
            )

            parsed_num_nodes = graph_node_count(edges)
            parsed_num_edges = len(edges)

            exact_result = float(
                get_exact_result_from_misc(
                    edges=edges,
                    weights=weights,
                    n=parsed_num_nodes,
                    m=parsed_num_edges,
                    raise_if_missing=True,
                )
            )

        except Exception as e:
            exact_error = repr(e)

        finished = "(END)" in text or sdp_objective is not None
        cut_edges_raw = cut_edges

        # Full Lasserre 2 logs report these values on twice the exact-result scale.
        rounded_energy = None if rounded_energy is None else rounded_energy / RESULT_NORMALISATION_FACTOR
        alg17_lower = None if alg17_lower is None else alg17_lower / RESULT_NORMALISATION_FACTOR
        alg17_actual = None if alg17_actual is None else alg17_actual / RESULT_NORMALISATION_FACTOR
        sdp_objective = None if sdp_objective is None else sdp_objective / RESULT_NORMALISATION_FACTOR
        cut_edges = None if cut_edges is None else cut_edges / RESULT_NORMALISATION_FACTOR

        row = {
            "log_file": log_path.name,
            "file_n": file_n,
            "repeat": repeat,
            "hog_idx": hog_idx,
            "original_hog_idx": original_hog_idx,
            "nodes": nodes,
            "edges": edge_count,
            "parsed_num_nodes": parsed_num_nodes,
            "parsed_num_edges": parsed_num_edges,
            "graph_type": graph_type,
            "generator_n": generator_n,
            "weighted": weighted,
            "random_weights": random_weights,
            "finished": finished,
            "exact_result": exact_result,
            "rounded_energy": rounded_energy,
            "alg17_lower_bound_energy": alg17_lower,
            "alg17_actual_energy": alg17_actual,
            "sdp_objective_value": sdp_objective,
            "cut_edges": cut_edges,
            "cut_edges_raw": cut_edges_raw,
            "rounded_over_exact": ratio(rounded_energy, exact_result),
            "alg17_lower_bound_over_exact": ratio(alg17_lower, exact_result),
            "alg17_actual_over_exact": ratio(alg17_actual, exact_result),
            "sdp_objective_over_exact": ratio(sdp_objective, exact_result),
            "cut_edges_over_exact": ratio(cut_edges, exact_result),
            "rounded_over_sdp": ratio(rounded_energy, sdp_objective),
            "alg17_actual_over_sdp": ratio(alg17_actual, sdp_objective),
            "sdp_objective_over_edges": ratio(sdp_objective, edge_count),
            "start_time": human_ts(start_ts),
            "duration": human_duration(duration_seconds),
            "duration_seconds": duration_seconds,
            "exact_error": exact_error,
        }

        rows.append(row)

    out_csv = log_dir / "sdp_relaxation_vs_exact_summary.csv"

    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    valid = [r for r in rows if r["exact_result"] is not None]
    missing_exact = [r for r in rows if r["exact_result"] is None]

    print()
    print(f"QAOA root:             {QAOA_ROOT}")
    print(f"Log dir:               {log_dir}")
    print(f"Mapping CSV:           {MAPPING_CSV if MAPPING_CSV.exists() else 'not used'}")
    print(f"Result normalisation:  / {RESULT_NORMALISATION_FACTOR:g} (exact_result unchanged)")
    print(f"Parsed log files:      {len(rows)}")
    print(f"With exact result:     {len(valid)}")
    print(f"Missing exact result:  {len(missing_exact)}")
    print()

    print("Relaxation / rounding ratios vs exact optimum:")
    print("-" * 226)
    print(
        f"{'file':35} "
        f"{'type':>8} "
        f"{'gen_n':>5} "
        f"{'hog':>5} "
        f"{'orig':>5} "
        f"{'n':>4} "
        f"{'m':>4} "
        f"{'p_n':>4} "
        f"{'p_m':>4} "
        f"{'exact':>10} "
        f"{'round':>10} "
        f"{'Alg17 LB':>10} "
        f"{'Alg17 act':>10} "
        f"{'SDP':>10} "
        f"{'round/ex':>10} "
        f"{'LB/ex':>10} "
        f"{'act/ex':>10} "
        f"{'SDP/ex':>10} "
        f"{'cut/ex':>10} "
        f"{'duration':>12}"
    )
    print("-" * 226)

    for r in sorted(
        rows,
        key=lambda x: (
            x["sdp_objective_over_exact"]
            if x["sdp_objective_over_exact"] is not None
            else -math.inf
        ),
        reverse=True,
    ):
        print(
            f"{r['log_file']:35} "
            f"{str(r['graph_type']):>8} "
            f"{str(r['generator_n']):>5} "
            f"{str(r['hog_idx']):>5} "
            f"{str(r['original_hog_idx']):>5} "
            f"{str(r['nodes']):>4} "
            f"{str(r['edges']):>4} "
            f"{str(r['parsed_num_nodes']):>4} "
            f"{str(r['parsed_num_edges']):>4} "
            f"{fmt(r['exact_result'], 3):>10} "
            f"{fmt(r['rounded_energy'], 3):>10} "
            f"{fmt(r['alg17_lower_bound_energy'], 3):>10} "
            f"{fmt(r['alg17_actual_energy'], 3):>10} "
            f"{fmt(r['sdp_objective_value'], 3):>10} "
            f"{fmt(r['rounded_over_exact'], 4):>10} "
            f"{fmt(r['alg17_lower_bound_over_exact'], 4):>10} "
            f"{fmt(r['alg17_actual_over_exact'], 4):>10} "
            f"{fmt(r['sdp_objective_over_exact'], 4):>10} "
            f"{fmt(r['cut_edges_over_exact'], 4):>10} "
            f"{r['duration']:>12}"
        )

    def stats(name, values):
        values = [v for v in values if v is not None]
        if not values:
            return

        values = sorted(values)
        n_values = len(values)
        mean = sum(values) / n_values
        median = values[n_values // 2] if n_values % 2 else (values[n_values // 2 - 1] + values[n_values // 2]) / 2

        print(
            f"{name:30} "
            f"count={n_values:3d}  "
            f"min={min(values):.4f}  "
            f"median={median:.4f}  "
            f"mean={mean:.4f}  "
            f"max={max(values):.4f}"
        )

    print()
    print("Ratio statistics:")
    print("-" * 105)
    stats("rounded / exact", [r["rounded_over_exact"] for r in rows])
    stats("Alg17 lower / exact", [r["alg17_lower_bound_over_exact"] for r in rows])
    stats("Alg17 actual / exact", [r["alg17_actual_over_exact"] for r in rows])
    stats("SDP / exact", [r["sdp_objective_over_exact"] for r in rows])
    stats("cut_edges / exact", [r["cut_edges_over_exact"] for r in rows])

    suspicious = [
        r for r in rows
        if (
            (r["rounded_over_exact"] is not None and r["rounded_over_exact"] > 1.0 + STATE_OVERSHOOT_TOL)
            or (r["alg17_actual_over_exact"] is not None and r["alg17_actual_over_exact"] > 1.0 + STATE_OVERSHOOT_TOL)
            or (r["alg17_lower_bound_over_exact"] is not None and r["alg17_lower_bound_over_exact"] > 1.0 + STATE_OVERSHOOT_TOL)
            or (r["sdp_objective_over_exact"] is not None and r["sdp_objective_over_exact"] < 1.0 - SDP_UNDERSHOOT_TOL)
        )
    ]

    if suspicious:
        print()
        print("Suspicious cases:")
        print("-" * 105)
        for r in suspicious:
            print(
                f"{r['log_file']} "
                f"(hog={r['hog_idx']}, orig={r['original_hog_idx']}): "
                f"round/ex={fmt(r['rounded_over_exact'])}, "
                f"LB/ex={fmt(r['alg17_lower_bound_over_exact'])}, "
                f"act/ex={fmt(r['alg17_actual_over_exact'])}, "
                f"SDP/ex={fmt(r['sdp_objective_over_exact'])}"
            )

    if missing_exact:
        print()
        print("Missing exact result:")
        print("-" * 105)
        for r in missing_exact:
            print(f"{r['log_file']}: {r['exact_error']}")

    print()
    print(f"Written: {out_csv}")


if not LOG_DIRS:
    raise SystemExit("No log directories provided.")

for log_dir in LOG_DIRS:
    if not log_dir.exists():
        print(f"Skipping missing log dir: {log_dir}", file=sys.stderr)
        continue
    process_log_dir(log_dir)
PY
