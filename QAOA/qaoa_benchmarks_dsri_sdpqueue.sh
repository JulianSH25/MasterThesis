#!/usr/bin/env zsh

set -u

echo "Launcher PID: $$"
echo "Started at: $(date +'%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

if [[ -z "${MOSEKLM_LICENSE_FILE:-}" ]]; then
    if [[ -f "../mosek.lic" ]]; then
        export MOSEKLM_LICENSE_FILE="$(cd .. && pwd)/mosek.lic"
    elif [[ -f "$HOME/MasterThesis/mosek.lic" ]]; then
        export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"
    elif [[ -f "/home/coder/project/MasterThesis/mosek.lic" ]]; then
        export MOSEKLM_LICENSE_FILE="/home/coder/project/MasterThesis/mosek.lic"
    else
        echo "Warning: MOSEKLM_LICENSE_FILE is not set and no mosek.lic was found."
    fi
fi

if [[ -n "${MOSEKLM_LICENSE_FILE:-}" ]]; then
    echo "Using MOSEK license path: $MOSEKLM_LICENSE_FILE"
fi

AWK_BIN=$(command -v awk 2>/dev/null || echo /usr/bin/awk)
if [[ ! -x "$AWK_BIN" ]]; then
    echo "Error: awk is required but was not found."
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but was not found in PATH."
    echo "Install with: conda install -c conda-forge jq -y"
    exit 1
fi


if (( $# != 1 )); then
    echo "Usage: $0 <config-file>"
    echo "Example: $0 Code/run_configurations/benchmark_config_sdpqueue.json"
    exit 1
fi

config_arg="$1"
if [[ "$config_arg" == */* ]]; then
    config_source_file="$config_arg"
else
    config_source_file="Code/$config_arg"
fi

if [[ ! -f "$config_source_file" ]]; then
    echo "Error: config file not found: $config_source_file"
    exit 1
fi

result_name_suffix=$(jq -r '.result_name_suffix // ""' "$config_source_file")
if [[ ! "$result_name_suffix" =~ '^[A-Za-z0-9._-]*$' ]]; then
    echo "Error: result_name_suffix may contain only letters, numbers, dots, underscores, and hyphens."
    exit 1
fi

run_timestamp=$(date +"%Y%m%d_%H%M%S")
run_tag="${run_timestamp}_pid$$${result_name_suffix}"

config_snapshot_dir="Results/config_snapshots"
mkdir -p "$config_snapshot_dir"
config_file="${config_snapshot_dir}/benchmark_config_${run_tag}.json"
cp "$config_source_file" "$config_file"

echo "Using benchmark config snapshot: ${config_file}"

# -----------------------------
# Config
# -----------------------------
optimiser=$(jq -r '.optimiser' "$config_file")
if [[ "${optimiser:l}" == "exact" ]]; then
    iterations_list=(0)
    depth_list=(0)
else
    iterations_list=($(jq -r '.iterations_list[]' "$config_file"))
    depth_list=($(jq -r '.depth_list[]' "$config_file"))
fi

num_repeats=$(jq -r '.num_repeats // 1' "$config_file")
time_limit_seconds=$(jq -r '.time_limit // 0' "$config_file")

warm_start=$(jq -r '.warm_start // false' "$config_file")
warm_start_mode=$(jq -r '.warm_start_mode // "standard"' "$config_file")
lasserre_level=$(jq -r '.lasserre_level // "NA"' "$config_file")
initial_solver_level_M=$(jq -r '.initial_solver_level_M // "NA"' "$config_file")
configured_sdp_seed=$(jq -r '.sdp_seed // empty' "$config_file")

graph_generation_type=$(jq -r '.graph_generation_type // empty' "$config_file")
weighted=$(jq -r '.weighted // false' "$config_file")
relative_graph_adjList_path=$(jq -r '.relative_graph_adjList_path // empty' "$config_file")
n_start_raw=$(jq -r '.n_start // empty' "$config_file")
n_end_raw=$(jq -r '.n_end // empty' "$config_file")
max_graph_vertices=$(jq -r '.max_graph_vertices // empty' "$config_file")
easiest_first=$(jq -r '.easiest_first // false' "$config_file")

if [[ -n "$max_graph_vertices" && ! "$max_graph_vertices" =~ '^[1-9][0-9]*$' ]]; then
    echo "Error: max_graph_vertices must be a positive integer or null."
    exit 1
fi

sdp_max_parallel=$(jq -r '.sdp_max_parallel // 3' "$config_file")
qaoa_max_parallel=$(jq -r '.qaoa_max_parallel // 8' "$config_file")
queue_poll_interval_seconds=$(jq -r '.queue_poll_interval_seconds // 5' "$config_file")

sdp_threads=$(jq -r '.sdp_threads // 1' "$config_file")
qaoa_threads=$(jq -r '.qaoa_threads // 2' "$config_file")

# -----------------------------
# Slurm resource awareness
# -----------------------------
slurm_job_id="${SLURM_JOB_ID:-}"
slurm_job_name="${SLURM_JOB_NAME:-}"
slurm_node_list="${SLURM_JOB_NODELIST:-}"
slurm_cpus_per_task="${SLURM_CPUS_PER_TASK:-}"
slurm_mem_per_node_mb="${SLURM_MEM_PER_NODE:-}"

nominal_thread_budget=$(( sdp_max_parallel * sdp_threads + qaoa_max_parallel * qaoa_threads ))

if [[ -n "$slurm_job_id" ]]; then
    echo "Running inside Slurm job: ${slurm_job_id} (${slurm_job_name:-unknown})"
    echo "Slurm node list: ${slurm_node_list:-unknown}"
    echo "Slurm CPUs per task: ${slurm_cpus_per_task:-unknown}"
    echo "Nominal launcher thread budget: ${nominal_thread_budget}"

    if [[ -n "$slurm_cpus_per_task" && "$slurm_cpus_per_task" != "0" ]]; then
        if (( nominal_thread_budget > slurm_cpus_per_task )); then
            echo "Error: nominal launcher thread budget (${nominal_thread_budget}) exceeds SLURM_CPUS_PER_TASK (${slurm_cpus_per_task})."
            echo "Adjust sdp_max_parallel, qaoa_max_parallel, sdp_threads, qaoa_threads, or request more CPUs in the Slurm file."
            exit 1
        fi
    fi
else
    echo "Warning: not running inside a Slurm allocation. Heavy runs should be submitted with sbatch, not run on the login node."
fi

sdp_memory_limit_total_gb=$(jq -r '.sdp_memory_limit_total_gb // 80' "$config_file")
sdp_memory_limit_single_gb=$(jq -r '.sdp_memory_limit_single_gb // 80' "$config_file")
sdp_memory_poll_seconds=$(jq -r '.sdp_memory_poll_seconds // 5' "$config_file")
sdp_max_retries=$(jq -r '.sdp_max_retries // 3' "$config_file")

if [[ -n "$slurm_mem_per_node_mb" && "$slurm_mem_per_node_mb" != "0" ]]; then
    slurm_mem_per_node_gb=$(( slurm_mem_per_node_mb / 1024 ))
    echo "Slurm memory per node: ${slurm_mem_per_node_mb} MB (~${slurm_mem_per_node_gb} GB)"
    if (( sdp_memory_limit_total_gb > slurm_mem_per_node_gb )); then
        echo "Error: sdp_memory_limit_total_gb (${sdp_memory_limit_total_gb} GB) exceeds requested Slurm memory (~${slurm_mem_per_node_gb} GB)."
        echo "Adjust sdp_memory_limit_total_gb or request more memory in the Slurm file."
        exit 1
    fi
fi

persistent_warm_start_cache=$(jq -r '.persistent_warm_start_cache // true' "$config_file")
warm_start_cache_dir=$(jq -r '.warm_start_cache_dir // "Results/warm_start_cache"' "$config_file")
warm_start_cache_max_file_mb=$(jq -r '.warm_start_cache_max_file_mb // 512' "$config_file")
warm_start_cache_max_total_gb=$(jq -r '.warm_start_cache_max_total_gb // 60' "$config_file")

parameter_vector=$(jq -c '.parameter_vector' "$config_file")
singlet_injection=$(jq -r '.singlet_injection // false' "$config_file")
init_QAOAparams_close_to_zero=$(jq -r '.init_QAOAparams_close_to_zero // false' "$config_file")
use_correlations_as_initial_params=$(jq -r '.use_correlations_as_initial_params // false' "$config_file")
compare_with_010101=$(jq -r '.compare_with_010101 // false' "$config_file")
start_index_singlet=$(jq -r '.start_index_singlet // empty' "$config_file")
circuit_type=$(jq -r '.circuit_type // empty' "$config_file")
rerun_exclude_finished_instances=$(jq -r '.rerun_exclude_finished_instances // false' "$config_file")
warm_start_cache_producer_only=$(jq -r '.warm_start_cache_producer_only // false' "$config_file")
qaoa_only_from_existing_warm_start_cache=$(jq -r '.qaoa_only_from_existing_warm_start_cache // false' "$config_file")

if [[ "$warm_start_cache_producer_only" == "true" && "$qaoa_only_from_existing_warm_start_cache" == "true" ]]; then
    echo "Error: warm_start_cache_producer_only and qaoa_only_from_existing_warm_start_cache cannot both be true."
    exit 1
fi

if [[ "$qaoa_only_from_existing_warm_start_cache" == "true" && "${warm_start:l}" != "true" ]]; then
    echo "Error: qaoa_only_from_existing_warm_start_cache=true requires warm_start=true."
    exit 1
fi

main_file="Code/Main.py"
python_bin=$(command -v python || true)
if [[ -z "$python_bin" ]]; then
    echo "Error: python not found in PATH."
    exit 1
fi

mkdir -p "$warm_start_cache_dir"
mkdir -p Results/logs/${optimiser}

log_subdir="Results/logs/${optimiser}/${run_tag}"
status_subdir="${log_subdir}/status"
resource_requeue_csv="${log_subdir}/resource_requeues.csv"
failed_warm_start_csv="${log_subdir}/failed_warm_starts.csv"
failed_warm_start_adjlist="${log_subdir}/failed_warm_starts.adjlist"
mkdir -p "$log_subdir" "$status_subdir"

echo "time,n,repeat_idx,sdp_seed,pid,rss_mb,total_sdp_rss_mb,reason,retry_count,cache_path" > "$resource_requeue_csv"
echo "time,n,repeat_idx,sdp_seed,exit_code,reason,cache_path,log_file,status_file,failed_adjlist_file" > "$failed_warm_start_csv"
: > "$failed_warm_start_adjlist"

typeset -A failed_adjlist_written=()

benchmark_config_dump=$(jq -r 'to_entries[] | "  \(.key): \(.value|tojson)"' "$config_file")

# -----------------------------
# HOG handling
# -----------------------------
count_hog_graphs() {
    local path="$1"
    "$AWK_BIN" '
        BEGIN { count = 0; in_block = 0 }
        /^[[:space:]]*$/ { in_block = 0; next }
        {
            if (!in_block) {
                count += 1
                in_block = 1
            }
        }
        END { print count }
    ' "$path"
}

list_hog_graph_sizes() {
    local path="$1"
    "$python_bin" - "$path" <<'PY'
import sys
from pathlib import Path

content = Path(sys.argv[1]).read_text().strip()
for graph_index, raw_graph in enumerate(content.split("\n\n")):
    vertices = set()
    for line in raw_graph.splitlines():
        node, neighbours = line.split(":", 1)
        vertices.add(int(node.strip()))
        vertices.update(int(value) for value in neighbours.split())
    print(graph_index, len(vertices))
PY
}

hog_graph_hash_for_index() {
    local graph_index="$1"
    local path="$2"

    "$AWK_BIN" -v target="$graph_index" '
        BEGIN { count = -1; in_block = 0 }
        /^[[:space:]]*$/ { in_block = 0; next }
        {
            if (!in_block) {
                count += 1
                in_block = 1
            }
            if (count == target) {
                print $0
            }
        }
    ' "$path" | "$python_bin" -c 'import hashlib, sys; print(hashlib.sha1(sys.stdin.buffer.read()).hexdigest())'
}

if [[ "${graph_generation_type:l}" == "hog" ]]; then
    if [[ -z "$relative_graph_adjList_path" ]]; then
        echo "Error: relative_graph_adjList_path must be set for graph_generation_type=hog."
        exit 1
    fi

    hog_graph_path="Code/$relative_graph_adjList_path"

    if [[ ! -f "$hog_graph_path" ]]; then
        echo "Error: HOG adjacency-list file not found: $hog_graph_path"
        exit 1
    fi

    hog_graph_count=$(count_hog_graphs "$hog_graph_path")
    if [[ -z "$hog_graph_count" || "$hog_graph_count" == "0" ]]; then
        echo "Error: no HOG graphs found in $hog_graph_path."
        exit 1
    fi

    if [[ -z "$n_start_raw" || "$n_start_raw" == "null" ]]; then
        n_start=0
    else
        n_start="$n_start_raw"
    fi

    if [[ -z "$n_end_raw" || "$n_end_raw" == "null" ]]; then
        n_end=$((hog_graph_count - 1))
    else
        n_end="$n_end_raw"
    fi

    if (( n_start < 0 )); then
        echo "Error: n_start for HOG must be >= 0, got ${n_start}."
        exit 1
    fi

    if (( n_end >= hog_graph_count )); then
        echo "Error: n_end=${n_end} exceeds last HOG index $((hog_graph_count - 1))."
        exit 1
    fi

    if (( n_start > n_end )); then
        echo "Error: n_start=${n_start} must be <= n_end=${n_end}."
        exit 1
    fi

    echo "HOG index range: n=${n_start}..${n_end} out of ${hog_graph_count} graphs."
else
    if [[ -z "$n_start_raw" || -z "$n_end_raw" ]]; then
        echo "Error: n_start and n_end must be set for graph_generation_type=${graph_generation_type}."
        exit 1
    fi
    n_start="$n_start_raw"
    n_end="$n_end_raw"
fi

# -----------------------------
# Timeout
# -----------------------------
timeout_cmd=""
if (( time_limit_seconds > 0 )); then
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout ${time_limit_seconds}s"
    else
        echo "Warning: timeout not found. Continuing without time limits."
    fi
fi

# -----------------------------
# Cache helpers
# -----------------------------
cache_total_mb() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    du -sm "$dir" 2>/dev/null | "$AWK_BIN" '{print $1}'
}

file_size_mb() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo 0
        return
    fi
    du -sm "$file" 2>/dev/null | "$AWK_BIN" '{print $1}'
}

enforce_cache_limits() {
    local candidate_file="${1:-}"
    local candidate_mb total_mb max_total_mb oldest_file oldest_mb

    if [[ "$persistent_warm_start_cache" != "true" ]]; then
        return
    fi

    if [[ -n "$candidate_file" && -f "$candidate_file" ]]; then
        candidate_mb=$(file_size_mb "$candidate_file")
        if (( candidate_mb > warm_start_cache_max_file_mb )); then
            echo "Cache file exceeds per-file cap; deleting: ${candidate_file} (${candidate_mb} MB > ${warm_start_cache_max_file_mb} MB)"
            rm -f "$candidate_file"
            return
        fi
    fi

    max_total_mb=$(( warm_start_cache_max_total_gb * 1024 ))
    total_mb=$(cache_total_mb "$warm_start_cache_dir")

    while (( total_mb > max_total_mb )); do
        oldest_file=$(find "$warm_start_cache_dir" -type f -name "*.npz" -printf "%T@ %p\n" 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)
        if [[ -z "$oldest_file" || ! -f "$oldest_file" ]]; then
            break
        fi

        oldest_mb=$(file_size_mb "$oldest_file")
        echo "Evicting old cache file: ${oldest_file} (${oldest_mb} MB)"
        rm -f "$oldest_file"
        total_mb=$(cache_total_mb "$warm_start_cache_dir")
    done
}

safe_name() {
    local raw="$1"
    echo "$raw" \
      | sed 's/\.[^.]*$//' \
      | sed 's/[^A-Za-z0-9._-]/_/g' \
      | sed 's/_\+/_/g' \
      | sed 's/^_//' \
      | sed 's/_$//'
}

file_hash_short() {
    local file="$1"
    "$python_bin" - "$file" <<'PYHASH'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
h = hashlib.sha1()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest()[:8])
PYHASH
}

sha1_file_short() {
    local file="$1"

    if command -v sha1sum >/dev/null 2>&1; then
        sha1sum "$file" | "$AWK_BIN" '{print substr($1, 1, 8)}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 1 "$file" | "$AWK_BIN" '{print substr($1, 1, 8)}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha1 "$file" | "$AWK_BIN" '{print substr($NF, 1, 8)}'
    else
        echo "Error: need sha1sum, shasum, or openssl for hashing." >&2
        exit 1
    fi
}

build_persistent_warm_start_cache_path() {
    local n="$1"
    local repeat_idx_value="$2"
    local sdp_seed_value="$3"
    local cache_mode="${4:-$warm_start_mode}"
    local graph_hash graph_label seed_label cache_dataset_dir hog_file_base hog_file_safe hog_file_hash

    seed_label="noseed"
    if [[ -n "$sdp_seed_value" ]]; then
        seed_label="seed${sdp_seed_value}"
    fi

    if [[ "${graph_generation_type:l}" == "hog" ]]; then
        hog_file_base="$(basename "$hog_graph_path")"
        hog_file_safe="$(safe_name "$hog_file_base")"
        hog_file_hash="$(file_hash_short "$hog_graph_path")"

        cache_dataset_dir="${warm_start_cache_dir}/${hog_file_safe}__filehash_${hog_file_hash}"
        mkdir -p "$cache_dataset_dir"

        graph_hash=$(hog_graph_hash_for_index "$n" "$hog_graph_path")
        graph_label="hog_idx$(printf "%06d" "$n")_${graph_hash[1,12]}"
    else
        cache_dataset_dir="${warm_start_cache_dir}/${graph_generation_type}"
        mkdir -p "$cache_dataset_dir"

        graph_hash=$(echo "${graph_generation_type}_${weighted}_${n}" | "$python_bin" -c 'import hashlib, sys; print(hashlib.sha1(sys.stdin.buffer.read()).hexdigest())')
        graph_label="${graph_generation_type}_n${n}_${graph_hash[1,12]}"
    fi

    echo "${cache_dataset_dir}/${graph_label}_${seed_label}_L${lasserre_level}_M${initial_solver_level_M}_${cache_mode}.npz"
}

# -----------------------------
# Run key helpers
# -----------------------------
whitelist_file="runkey_whitelist.json"
if [[ -f "$whitelist_file" ]]; then
    whitelist=($(jq -r '.whitelist[]' "$whitelist_file"))
else
    whitelist=(
        n m p repeat_idx sdp_seed precision/iterations parameter_vector
        singlet_injection warm_start init_QAOAparams_close_to_zero
        use_correlations_as_initial_params compare_with_010101
        start_index_singlet circuit_type graph_generation_type weighted
        hog_graph_index
    )
fi

completed_file="completed_runs.txt"
typeset -A completed_map
if [[ "$rerun_exclude_finished_instances" == "true" && -f "$completed_file" ]]; then
    while read -r line; do
        completed_map["$line"]=1
    done < "$completed_file"
fi

typeset -A completed_csv_map
completed_result_csv_key_file=""
build_completed_instance_key() {
    local n="$1"
    local p="$2"
    local iterations="$3"
    local repeat_value="$4"
    local seed_value="$5"

    echo "graph_generation_type=${graph_generation_type}|weighted=${weighted}|relative_graph_adjList_path=${relative_graph_adjList_path}|warm_start=${warm_start}|warm_start_mode=${warm_start_mode}|lasserre_level=${lasserre_level}|initial_solver_level_M=${initial_solver_level_M}|hog_graph_index=${n}|repeat_idx=${repeat_value}|sdp_seed=${seed_value}|p=${p}|iterations=${iterations}"
}

load_completed_result_csv_paths() {
    local csv_key_file
    csv_key_file="$(mktemp)"
    completed_result_csv_key_file="$csv_key_file"

    "$python_bin" - "$config_file" "$csv_key_file" <<'PYCHECKDONE'
import json
import sys
from pathlib import Path

import pandas as pd

config_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
config = json.loads(config_path.read_text())

csv_paths = config.get("completed_result_csv_paths") or []
if isinstance(csv_paths, str):
    csv_paths = [csv_paths]

base_dir = Path.cwd()

current_graph_generation_type = str(config.get("graph_generation_type", ""))
current_weighted = str(bool(config.get("weighted", False))).lower()
current_relative_graph_adjlist_path = str(config.get("relative_graph_adjList_path", ""))
current_warm_start = str(bool(config.get("warm_start", False))).lower()
current_warm_start_mode = str(config.get("warm_start_mode", ""))
current_lasserre_level = str(config.get("lasserre_level", ""))
current_initial_solver_level_m = str(config.get("initial_solver_level_M", ""))
current_sdp_seed = config.get("sdp_seed", None)
seed_increment_per_repeat = 1

def scalar_to_string(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text

def bool_column_to_string(value):
    text = scalar_to_string(value).lower()
    if text in {"true", "1", "1.0", "yes"}:
        return "true"
    if text in {"false", "0", "0.0", "no"}:
        return "false"
    return text

def row_value(row, *names):
    for name in names:
        if name in row.index:
            value = row[name]
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return None

def config_match(row):
    checks = []

    if "graph_generation_type" in row.index:
        checks.append(scalar_to_string(row["graph_generation_type"]) == current_graph_generation_type)
    if "weighted" in row.index:
        checks.append(bool_column_to_string(row["weighted"]) == current_weighted)
    if "relative_graph_adjList_path" in row.index:
        checks.append(scalar_to_string(row["relative_graph_adjList_path"]) == current_relative_graph_adjlist_path)
    if "warm_start" in row.index:
        checks.append(bool_column_to_string(row["warm_start"]) == current_warm_start)
    if "warm_start_mode" in row.index:
        checks.append(scalar_to_string(row["warm_start_mode"]) == current_warm_start_mode)
    if "lasserre_level" in row.index:
        checks.append(scalar_to_string(row["lasserre_level"]) == current_lasserre_level)
    if "initial_solver_level_M" in row.index:
        checks.append(scalar_to_string(row["initial_solver_level_M"]) == current_initial_solver_level_m)

    # If none of these metadata columns exist, do not trust this CSV for skipping.
    return bool(checks) and all(checks)

keys = set()
used_rows = 0
skipped_rows = 0
missing_required_rows = 0
missing_files = []

for raw_path in csv_paths:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path

    if not path.exists():
        missing_files.append(str(path))
        continue

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        if not config_match(row):
            skipped_rows += 1
            continue

        hog_graph_index = row_value(row, "hog_graph_index", "n")
        repeat_idx = row_value(row, "benchmark_repeat_idx", "repeat_idx")
        p = row_value(row, "p")
        iterations = row_value(row, "precision/iterations", "iterations")

        if hog_graph_index is None or repeat_idx is None or p is None or iterations is None:
            missing_required_rows += 1
            continue

        seed = row_value(row, "sdp_seed", "sdp_seed.1")
        if seed is None and current_sdp_seed is not None:
            try:
                seed = int(current_sdp_seed) + (int(float(repeat_idx)) - 1) * seed_increment_per_repeat
            except Exception:
                seed = ""

        key = "|".join([
            f"graph_generation_type={current_graph_generation_type}",
            f"weighted={current_weighted}",
            f"relative_graph_adjList_path={current_relative_graph_adjlist_path}",
            f"warm_start={current_warm_start}",
            f"warm_start_mode={current_warm_start_mode}",
            f"lasserre_level={current_lasserre_level}",
            f"initial_solver_level_M={current_initial_solver_level_m}",
            f"hog_graph_index={scalar_to_string(hog_graph_index)}",
            f"repeat_idx={scalar_to_string(repeat_idx)}",
            f"sdp_seed={scalar_to_string(seed)}",
            f"p={scalar_to_string(p)}",
            f"iterations={scalar_to_string(iterations)}",
        ])
        keys.add(key)
        used_rows += 1

out_path.write_text("\n".join(sorted(keys)) + ("\n" if keys else ""))

print(f"completed_result_csv_paths configured: {len(csv_paths)}")
print(f"completed_result_csv matched rows: {used_rows}")
print(f"completed_result_csv unique completed keys: {len(keys)}")
print(f"completed_result_csv skipped non-matching rows: {skipped_rows}")
print(f"completed_result_csv rows missing required columns: {missing_required_rows}")
for missing in missing_files:
    print(f"Warning: completed result CSV not found: {missing}")
PYCHECKDONE

    if [[ -s "$csv_key_file" ]]; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                completed_csv_map[$line]=1
            fi
        done < "$csv_key_file"
    fi
}

normalize() {
    local v="$1"

    if [[ -z "$v" || "$v" == "null" ]]; then
        echo ""
        return
    fi

    if echo "$v" | jq -e . >/dev/null 2>&1; then
        echo "$v" | jq -c -S .
    else
        echo "$v" | "$AWK_BIN" '{$1=$1;print}'
    fi
}

compute_m() {
    local n="$1"
    if [[ "${graph_generation_type:l}" == "hog" ]]; then
        echo "n/a"
        return
    fi

    case "$graph_generation_type" in
        line) echo $((n - 1)) ;;
        cycle) echo "$n" ;;
        complete) echo $((n * (n - 1) / 2)) ;;
        *) echo "0" ;;
    esac
}

append_hog_graph_block_to_failed_adjlist() {
    local graph_index="$1"
    local graph_key="hog_${graph_index}"

    if [[ "${graph_generation_type:l}" != "hog" ]]; then
        return
    fi

    if [[ -n "${failed_adjlist_written[$graph_key]:-}" ]]; then
        return
    fi

    if [[ -z "${hog_graph_path:-}" || ! -f "$hog_graph_path" ]]; then
        return
    fi

    "$AWK_BIN" -v target="$graph_index" '
        BEGIN { count = -1; in_block = 0; printed = 0 }
        /^[[:space:]]*$/ {
            if (in_block && count == target) {
                print ""
                printed = 1
                exit
            }
            in_block = 0
            next
        }
        {
            if (!in_block) {
                count += 1
                in_block = 1
            }
            if (count == target) {
                print $0
            }
        }
        END {
            if (in_block && count == target && !printed) {
                print ""
            }
        }
    ' "$hog_graph_path" >> "$failed_warm_start_adjlist"

    failed_adjlist_written[$graph_key]=1
}

log_failed_warm_start_row() {
    local n_value="$1"
    local repeat_value="$2"
    local seed_value="$3"
    local exit_code_value="$4"
    local reason_value="$5"
    local cache_value="$6"
    local log_file_value="$7"
    local status_file_value="$8"

    local now
    now=$(date +'%Y-%m-%d %H:%M:%S')

    echo "${now},${n_value},${repeat_value},${seed_value},${exit_code_value},${reason_value},${cache_value},${log_file_value},${status_file_value},${failed_warm_start_adjlist}" >> "$failed_warm_start_csv"

    append_hog_graph_block_to_failed_adjlist "$n_value"
}

build_run_key() {
    local n="$1"
    local p="$2"
    local iterations="$3"
    local m key_string key value norm_value

    m=$(compute_m "$n")
    key_string=""

    for key in "${whitelist[@]}"; do
        case "$key" in
            n) value="$n" ;;
            m) value="$m" ;;
            p) value="$p" ;;
            repeat_idx) value="$repeat_idx" ;;
            sdp_seed) value="$derived_sdp_seed" ;;
            precision/iterations) value="$iterations" ;;
            parameter_vector) value="$parameter_vector" ;;
            singlet_injection) value="$singlet_injection" ;;
            warm_start) value="$warm_start" ;;
            init_QAOAparams_close_to_zero) value="$init_QAOAparams_close_to_zero" ;;
            use_correlations_as_initial_params) value="$use_correlations_as_initial_params" ;;
            compare_with_010101) value="$compare_with_010101" ;;
            start_index_singlet) value="$start_index_singlet" ;;
            circuit_type) value="$circuit_type" ;;
            graph_generation_type) value="$graph_generation_type" ;;
            weighted) value="$weighted" ;;
            hog_graph_index) value="$n" ;;
            *) value="" ;;
        esac

        norm_value=$(normalize "$value")

        if [[ -n "$key_string" ]]; then
            key_string+="|"
        fi
        key_string+="${key}=${norm_value}"
    done

    echo -n "$key_string" | "$python_bin" -c 'import hashlib, sys; print(hashlib.sha1(sys.stdin.buffer.read()).hexdigest())'
}

# Load completed result CSV keys if enabled
if [[ "$rerun_exclude_finished_instances" == "true" ]]; then
    load_completed_result_csv_paths
fi

# -----------------------------
# Queues and PID metadata
# -----------------------------
typeset -a pending_sdp_keys=()
typeset -a pending_qaoa_keys=()
typeset -a running_sdp_pids=()
typeset -a running_qaoa_pids=()

typeset -A sdp_n
typeset -A sdp_repeat
typeset -A sdp_seed
typeset -A sdp_cache
typeset -A sdp_retry
typeset -A sdp_producer_p
typeset -A sdp_producer_iterations

typeset -A qaoa_n
typeset -A qaoa_p
typeset -A qaoa_iterations
typeset -A qaoa_repeat
typeset -A qaoa_seed
typeset -A qaoa_cache
typeset -A qaoa_role

typeset -A pid_to_key
typeset -A pid_to_role
typeset -A pid_to_start_order

launch_counter=0
stop_launching=0

rss_mb_for_pid() {
    local pid="$1"
    local rss_kb
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | "$AWK_BIN" '{print $1}')
    rss_kb=${rss_kb:-0}
    echo $(( rss_kb / 1024 ))
}

refresh_running_arrays() {
    local pid key role status_file exit_code
    local -a new_sdp=()
    local -a new_qaoa=()

    for pid in "${running_sdp_pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            new_sdp+=("$pid")
        else
            key="${pid_to_key[$pid]:-}"
            if [[ -n "$key" ]]; then
                status_file="${status_subdir}/sdp_${key}.status"
                if [[ -f "$status_file" ]]; then
                    exit_code=$("$AWK_BIN" -F= '/^exit_code=/{print $2}' "$status_file")
                    if [[ "$exit_code" == "0" && -f "${sdp_cache[$key]}" ]]; then
                        echo "SDP producer finished successfully for key=${key}; cache=${sdp_cache[$key]}"
                        enforce_cache_limits "${sdp_cache[$key]}"
                        if [[ "$warm_start_cache_producer_only" == "true" ]]; then
                            echo "warm_start_cache_producer_only=true; not enqueueing QAOA jobs for key=${key}."
                        else
                            enqueue_qaoa_for_sdp_key "$key"
                        fi
                    elif [[ "$exit_code" == "137" || "$exit_code" == "143" ]]; then
                        # Killed by our memory guard. Already requeued there.
                        echo "SDP producer ended after kill/requeue for key=${key}, exit_code=${exit_code}"
                    else
                        log_failed_warm_start_row \
                            "${sdp_n[$key]}" \
                            "${sdp_repeat[$key]}" \
                            "${sdp_seed[$key]}" \
                            "${exit_code:-unknown}" \
                            "sdp_producer_failed" \
                            "${sdp_cache[$key]}" \
                            "${log_subdir}/sdp_${key}.log" \
                            "${status_file}"
                        echo "SDP producer failed for key=${key}, exit_code=${exit_code:-unknown}"
                    fi
                else
                    echo "Warning: SDP pid disappeared without status file: pid=${pid}, key=${key}"
                fi
            fi
        fi
    done

    for pid in "${running_qaoa_pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            new_qaoa+=("$pid")
        fi
    done

    running_sdp_pids=("${new_sdp[@]}")
    running_qaoa_pids=("${new_qaoa[@]}")
}

running_sdp_count() {
    refresh_running_arrays
    echo "${#running_sdp_pids[@]}"
}

running_qaoa_count() {
    refresh_running_arrays
    echo "${#running_qaoa_pids[@]}"
}

total_sdp_rss_mb() {
    local pid total rss
    total=0
    for pid in "${running_sdp_pids[@]}"; do
        rss=$(rss_mb_for_pid "$pid")
        total=$(( total + rss ))
    done
    echo "$total"
}

newest_sdp_pid() {
    local pid newest newest_order order
    newest=""
    newest_order=-1

    for pid in "${running_sdp_pids[@]}"; do
        order="${pid_to_start_order[$pid]:-0}"
        if (( order > newest_order )); then
            newest="$pid"
            newest_order="$order"
        fi
    done

    echo "$newest"
}

requeue_sdp_key_front() {
    local key="$1"
    pending_sdp_keys=("$key" "${pending_sdp_keys[@]}")
}

handle_sdp_memory_pressure() {
    local total_mb total_limit_mb single_limit_mb pid rss key retry newest now

    total_limit_mb=$(( sdp_memory_limit_total_gb * 1024 ))
    single_limit_mb=$(( sdp_memory_limit_single_gb * 1024 ))

    # First: single-process limit. If a single SDP exceeds the limit, fail it, not requeue.
    for pid in "${running_sdp_pids[@]}"; do
        rss=$(rss_mb_for_pid "$pid")
        if (( rss > single_limit_mb )); then
            key="${pid_to_key[$pid]:-}"
            now=$(date +'%Y-%m-%d %H:%M:%S')
            echo "${now},${sdp_n[$key]},${sdp_repeat[$key]},${sdp_seed[$key]},${pid},${rss},$(total_sdp_rss_mb),single_sdp_memory_limit_exceeded,${sdp_retry[$key]},${sdp_cache[$key]}" >> "$resource_requeue_csv"
            log_failed_warm_start_row \
                "${sdp_n[$key]}" \
                "${sdp_repeat[$key]}" \
                "${sdp_seed[$key]}" \
                "memory" \
                "single_sdp_memory_limit_exceeded" \
                "${sdp_cache[$key]}" \
                "${log_subdir}/sdp_${key}.log" \
                "${status_subdir}/sdp_${key}.status"

            echo "Killing SDP pid=${pid} permanently: single process RSS ${rss} MB > ${single_limit_mb} MB."
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    refresh_running_arrays

    # Second: global SDP memory pressure. Kill newest and requeue.
    while true; do
        total_mb=$(total_sdp_rss_mb)
        if (( total_mb <= total_limit_mb )); then
            break
        fi

        newest=$(newest_sdp_pid)
        if [[ -z "$newest" ]]; then
            break
        fi

        key="${pid_to_key[$newest]:-}"
        if [[ -z "$key" ]]; then
            break
        fi

        retry="${sdp_retry[$key]:-0}"
        now=$(date +'%Y-%m-%d %H:%M:%S')
        rss=$(rss_mb_for_pid "$newest")

        echo "${now},${sdp_n[$key]},${sdp_repeat[$key]},${sdp_seed[$key]},${newest},${rss},${total_mb},global_sdp_memory_pressure,${retry},${sdp_cache[$key]}" >> "$resource_requeue_csv"

        echo "Global SDP memory pressure: total=${total_mb} MB > ${total_limit_mb} MB. Killing newest SDP pid=${newest}, key=${key}, retry=${retry}."

        kill "$newest" 2>/dev/null || true
        sleep 2
        kill -9 "$newest" 2>/dev/null || true

        if (( retry < sdp_max_retries )); then
            sdp_retry[$key]=$(( retry + 1 ))
            rm -rf "${sdp_cache[$key]}.lock"
            rm -f "${sdp_cache[$key]}" "${sdp_cache[$key]}.tmp" "${sdp_cache[$key]}.failed"
            requeue_sdp_key_front "$key"
            echo "Requeued SDP key=${key}; new retry count=${sdp_retry[$key]}"
        else
            log_failed_warm_start_row \
                "${sdp_n[$key]}" \
                "${sdp_repeat[$key]}" \
                "${sdp_seed[$key]}" \
                "memory" \
                "global_memory_retries_exceeded" \
                "${sdp_cache[$key]}" \
                "${log_subdir}/sdp_${key}.log" \
                "${status_subdir}/sdp_${key}.status"
            echo "SDP key=${key} exceeded max memory retries; marking failed."
        fi

        refresh_running_arrays
    done
}

enqueue_qaoa_for_sdp_key() {
    local key="$1"
    local n repeat seed cache producer_p producer_iterations iterations p qkey

    n="${sdp_n[$key]}"
    repeat="${sdp_repeat[$key]}"
    seed="${sdp_seed[$key]}"
    cache="${sdp_cache[$key]}"
    producer_p="${sdp_producer_p[$key]}"
    producer_iterations="${sdp_producer_iterations[$key]}"

    if [[ ! -f "$cache" ]]; then
        echo "Cannot enqueue QAOA; cache missing: ${cache}"
        return
    fi

    for iterations in "${iterations_list[@]}"; do
        for p in "${depth_list[@]}"; do
            # In combined mode, the SDP producer also ran this pair and should have written its QAOA result.
            # In qaoa_only_from_existing_warm_start_cache mode, include all depth/iteration pairs because
            # the cache may have been produced by an SDP-only run that deliberately skipped QAOA.
            if [[ "$qaoa_only_from_existing_warm_start_cache" != "true" && "$p" == "$producer_p" && "$iterations" == "$producer_iterations" ]]; then
                echo "Skipping QAOA enqueue for producer pair: key=${key}, p=${p}, iterations=${iterations}"
                continue
            fi

            qkey="${key}_p${p}_it${iterations}"
            qaoa_n[$qkey]="$n"
            qaoa_p[$qkey]="$p"
            qaoa_iterations[$qkey]="$iterations"
            qaoa_repeat[$qkey]="$repeat"
            qaoa_seed[$qkey]="$seed"
            qaoa_cache[$qkey]="$cache"
            qaoa_role[$qkey]="qaoa_cached"
            pending_qaoa_keys+=("$qkey")
        done
    done
}

launch_sdp_key() {
    local key="$1"
    local n repeat seed cache producer_p producer_iterations seed_label log_file status_file cmd lock_dir pid

    n="${sdp_n[$key]}"
    repeat="${sdp_repeat[$key]}"
    seed="${sdp_seed[$key]}"
    cache="${sdp_cache[$key]}"
    producer_p="${sdp_producer_p[$key]}"
    producer_iterations="${sdp_producer_iterations[$key]}"

    if [[ -f "$cache" ]]; then
        echo "Cache already exists for SDP key=${key}: ${cache}"
        if [[ "$warm_start_cache_producer_only" == "true" ]]; then
            echo "warm_start_cache_producer_only=true; cache already exists, nothing else to do for key=${key}."
        else
            enqueue_qaoa_for_sdp_key "$key"
        fi
        return 0
    fi

    lock_dir="${cache}.lock"
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "Cache lock exists for key=${key}; requeueing for later: ${lock_dir}"
        requeue_sdp_key_front "$key"
        sleep "$queue_poll_interval_seconds"
        return 0
    fi

    seed_label="noseed"
    if [[ -n "$seed" ]]; then
        seed_label="seed${seed}"
    fi

    log_file="${log_subdir}/sdp_${key}.log"
    status_file="${status_subdir}/sdp_${key}.status"

    echo "Launching SDP producer key=${key}, n=${n}, repeat=${repeat}, seed=${seed:-none}, p=${producer_p}, iterations=${producer_iterations}, retry=${sdp_retry[$key]}, cache=${cache}"

    cmd="${python_bin} ${main_file} ${producer_iterations} ${producer_p} ${n} ${n} Results/logs/${optimiser}/qaoa_results_${optimiser}_${run_tag}"

    if [[ -n "$timeout_cmd" ]]; then
        cmd="${timeout_cmd} ${cmd}"
    fi

    nohup zsh -c "
        export OMP_NUM_THREADS=${sdp_threads}
        export OPENBLAS_NUM_THREADS=${sdp_threads}
        export MKL_NUM_THREADS=${sdp_threads}
        export NUMEXPR_NUM_THREADS=${sdp_threads}
        export MOSEK_NUM_THREADS=${sdp_threads}
        export MOSEKLM_LICENSE_FILE=\"${MOSEKLM_LICENSE_FILE}\"
        export BENCHMARK_CONFIG_FILE=\"${config_file}\"
        export BENCHMARK_REPEAT_IDX=\"${repeat}\"

        if [[ -n \"${seed}\" ]]; then
            export SDP_SEED_OVERRIDE=${seed}
        fi

        export WARM_START_CACHE_PATH=\"${cache}\"

        if [[ \"${warm_start_cache_producer_only}\" == \"true\" ]]; then
            export WARM_START_CACHE_PRODUCER_ONLY=1
        fi

        ${cmd}
        exit_code=\$?

        timed_out=0
        warm_start_failed=0

        if [[ -n \"${timeout_cmd}\" ]] && (( exit_code == 124 )); then
            timed_out=1
        fi

        if (( exit_code == 42 )); then
            warm_start_failed=1
        fi

        rm -rf \"${lock_dir}\"

        {
            echo \"finished_at=\$(date +'%Y-%m-%d %H:%M:%S')\"
            echo \"exit_code=\${exit_code}\"
            echo \"timed_out=\${timed_out}\"
            echo \"warm_start_failed=\${warm_start_failed}\"
            echo \"role=sdp_producer\"
            echo \"key=${key}\"
            echo \"n=${n}\"
            echo \"p=${producer_p}\"
            echo \"iterations=${producer_iterations}\"
            echo \"repeat_idx=${repeat}\"
            echo \"sdp_seed=${seed}\"
            echo \"warm_start_cache_path=${cache}\"
            echo \"threads=${sdp_threads}\"
        } > \"${status_file}\"

        exit \${exit_code}
    " > "$log_file" 2>&1 &

    pid=$!

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        running_sdp_pids+=("$pid")
        pid_to_key[$pid]="$key"
        pid_to_role[$pid]="sdp"
        launch_counter=$(( launch_counter + 1 ))
        pid_to_start_order[$pid]="$launch_counter"
        echo "Launched SDP PID=${pid}, key=${key}"
    else
        echo "Warning: failed to launch SDP key=${key}; requeueing."
        rm -rf "$lock_dir"
        requeue_sdp_key_front "$key"
    fi
}

launch_qaoa_key() {
    local key="$1"
    local n p iterations repeat seed cache role seed_label log_file status_file cmd pid run_key completed_instance_key

    n="${qaoa_n[$key]}"
    p="${qaoa_p[$key]}"
    iterations="${qaoa_iterations[$key]}"
    repeat="${qaoa_repeat[$key]}"
    seed="${qaoa_seed[$key]}"
    cache="${qaoa_cache[$key]}"
    role="${qaoa_role[$key]}"

    seed_label="noseed"
    if [[ -n "$seed" ]]; then
        seed_label="seed${seed}"
    fi

    # These globals are used by build_run_key.
    repeat_idx="$repeat"
    derived_sdp_seed="$seed"
    run_key=$(build_run_key "$n" "$p" "$iterations")
    completed_instance_key=$(build_completed_instance_key "$n" "$p" "$iterations" "$repeat" "$seed")

    if [[ "$rerun_exclude_finished_instances" == "true" && -n "${completed_map[$run_key]:-}" ]]; then
        echo "Skipping already completed QAOA job via completed_runs.txt: n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat}"
        return 0
    fi

    if [[ "$rerun_exclude_finished_instances" == "true" && -n "${completed_csv_map[$completed_instance_key]-}" ]]; then
        echo "Skipping already completed QAOA job via completed_result_csv_paths: n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat}, seed=${seed:-none}"
        return 0
    fi

    if [[ -n "$cache" && ! -f "$cache" ]]; then
        echo "QAOA cache missing; dropping/requeueing not implemented for key=${key}: ${cache}"
        return 1
    fi

    log_file="${log_subdir}/${role}_${key}.log"
    status_file="${status_subdir}/${role}_${key}.status"

    echo "Launching QAOA key=${key}, n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat}, seed=${seed:-none}, cache=${cache:-none}"

    cmd="${python_bin} ${main_file} ${iterations} ${p} ${n} ${n} Results/logs/${optimiser}/qaoa_results_${optimiser}_${run_tag}"

    if [[ -n "$timeout_cmd" ]]; then
        cmd="${timeout_cmd} ${cmd}"
    fi

    nohup zsh -c "
        export OMP_NUM_THREADS=${qaoa_threads}
        export OPENBLAS_NUM_THREADS=${qaoa_threads}
        export MKL_NUM_THREADS=${qaoa_threads}
        export NUMEXPR_NUM_THREADS=${qaoa_threads}
        export MOSEK_NUM_THREADS=${qaoa_threads}
        export MOSEKLM_LICENSE_FILE=\"${MOSEKLM_LICENSE_FILE}\"
        export BENCHMARK_CONFIG_FILE=\"${config_file}\"
        export BENCHMARK_REPEAT_IDX=\"${repeat}\"

        if [[ -n \"${seed}\" ]]; then
            export SDP_SEED_OVERRIDE=${seed}
        fi

        if [[ -n \"${cache}\" ]]; then
            export WARM_START_CACHE_PATH=\"${cache}\"
        fi

        ${cmd}
        exit_code=\$?

        timed_out=0
        warm_start_failed=0

        if [[ -n \"${timeout_cmd}\" ]] && (( exit_code == 124 )); then
            timed_out=1
        fi

        if (( exit_code == 42 )); then
            warm_start_failed=1
        fi

        {
            echo \"finished_at=\$(date +'%Y-%m-%d %H:%M:%S')\"
            echo \"exit_code=\${exit_code}\"
            echo \"timed_out=\${timed_out}\"
            echo \"warm_start_failed=\${warm_start_failed}\"
            echo \"role=${role}\"
            echo \"key=${key}\"
            echo \"n=${n}\"
            echo \"p=${p}\"
            echo \"iterations=${iterations}\"
            echo \"repeat_idx=${repeat}\"
            echo \"sdp_seed=${seed}\"
            echo \"warm_start_cache_path=${cache}\"
            echo \"threads=${qaoa_threads}\"
        } > \"${status_file}\"

        exit \${exit_code}
    " > "$log_file" 2>&1 &

    pid=$!

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        running_qaoa_pids+=("$pid")
        pid_to_key[$pid]="$key"
        pid_to_role[$pid]="qaoa"
        launch_counter=$(( launch_counter + 1 ))
        pid_to_start_order[$pid]="$launch_counter"
        echo "Launched QAOA PID=${pid}, key=${key}"
    else
        echo "Warning: failed to launch QAOA key=${key}"
    fi
}

# -----------------------------
# Build initial queues
# -----------------------------
jobs_file="$(mktemp)"
skipped_graph_count=0
if [[ "${graph_generation_type:l}" == "hog" ]]; then
    while read -r n vertex_count; do
        if (( n < n_start || n > n_end )); then
            continue
        fi
        if [[ -n "$max_graph_vertices" ]] && (( vertex_count > max_graph_vertices )); then
            skipped_graph_count=$(( skipped_graph_count + 1 ))
            continue
        fi
        echo "${vertex_count} ${n}" >> "$jobs_file"
    done < <(list_hog_graph_sizes "$hog_graph_path")
else
    for (( n=n_start; n<=n_end; n++ )); do
        if [[ -n "$max_graph_vertices" ]] && (( n > max_graph_vertices )); then
            skipped_graph_count=$(( skipped_graph_count + 1 ))
            continue
        fi
        echo "${n} ${n}" >> "$jobs_file"
    done
fi

if (( skipped_graph_count > 0 )); then
    echo "Graph-size filter skipped ${skipped_graph_count} graph(s) above max_graph_vertices=${max_graph_vertices}."
fi

if [[ ! -s "$jobs_file" ]]; then
    rm -f "$jobs_file"
    echo "No graphs remain after applying max_graph_vertices=${max_graph_vertices:-none}; nothing to run."
    exit 0
fi

if [[ "$easiest_first" == "true" ]]; then
    sort -n "$jobs_file" -o "$jobs_file"
else
    "$python_bin" - "$jobs_file" <<'PY'
import random
import sys
from pathlib import Path

jobs_path = Path(sys.argv[1])
jobs = [line for line in jobs_path.read_text().splitlines() if line.strip()]
random.shuffle(jobs)
jobs_path.write_text("\n".join(jobs) + "\n")
PY
fi

producer_iterations="${iterations_list[1]}"
producer_p="${depth_list[1]}"

while read -r score n; do
    for (( repeat_idx=1; repeat_idx<=num_repeats; repeat_idx++ )); do
        derived_sdp_seed=""
        if [[ -n "$configured_sdp_seed" && "$configured_sdp_seed" != "null" ]]; then
            seed_increment_per_repeat=1
            derived_sdp_seed=$(( configured_sdp_seed + (repeat_idx - 1) * seed_increment_per_repeat ))
        fi

        if [[ "${warm_start:l}" == "true" ]]; then
            cache_lookup_mode="$warm_start_mode"
            if [[ "${warm_start_mode:l}" == "amplified_king" || "${warm_start_mode:l}" == "entangled_king" ]]; then
                cache_lookup_mode="amplified"
            fi
            cache_path=$(build_persistent_warm_start_cache_path "$n" "$repeat_idx" "$derived_sdp_seed" "$cache_lookup_mode")
            if [[ ! -f "$cache_path" && ( "${warm_start_mode:l}" == "standard" || "${warm_start_mode:l}" == "entangled" ) ]]; then
                amplified_cache_path=$(build_persistent_warm_start_cache_path "$n" "$repeat_idx" "$derived_sdp_seed" "amplified")
                if [[ -f "$amplified_cache_path" ]]; then
                    echo "Requested ${warm_start_mode:l} cache missing for n=${n}, repeat=${repeat_idx}, seed=${derived_sdp_seed:-none}; using compatible amplified cache: ${amplified_cache_path}"
                    cache_path="$amplified_cache_path"
                fi
            fi
            sdp_key="n${n}_rep${repeat_idx}_seed${derived_sdp_seed:-none}"

            sdp_n[$sdp_key]="$n"
            sdp_repeat[$sdp_key]="$repeat_idx"
            sdp_seed[$sdp_key]="${derived_sdp_seed:-}"
            sdp_cache[$sdp_key]="$cache_path"
            sdp_retry[$sdp_key]=0
            sdp_producer_p[$sdp_key]="$producer_p"
            sdp_producer_iterations[$sdp_key]="$producer_iterations"

            if [[ -f "$cache_path" ]]; then
                if [[ "$warm_start_cache_producer_only" == "true" ]]; then
                    echo "Existing cache found for ${sdp_key}; warm_start_cache_producer_only=true, so no QAOA is enqueued."
                else
                    echo "Existing cache found for ${sdp_key}; enqueueing QAOA directly."
                    enqueue_qaoa_for_sdp_key "$sdp_key"
                fi
            else
                if [[ "$qaoa_only_from_existing_warm_start_cache" == "true" ]]; then
                    echo "Missing cache for ${sdp_key}; qaoa_only_from_existing_warm_start_cache=true, so SDP producer is not enqueued: ${cache_path}"
                    log_failed_warm_start_row \
                        "$n" \
                        "$repeat_idx" \
                        "${derived_sdp_seed:-}" \
                        "missing_cache" \
                        "qaoa_only_missing_warm_start_cache" \
                        "$cache_path" \
                        "" \
                        ""
                else
                    pending_sdp_keys+=("$sdp_key")
                fi
            fi
        else
            for iterations in "${iterations_list[@]}"; do
                for p in "${depth_list[@]}"; do
                    qkey="n${n}_rep${repeat_idx}_p${p}_it${iterations}"
                    qaoa_n[$qkey]="$n"
                    qaoa_p[$qkey]="$p"
                    qaoa_iterations[$qkey]="$iterations"
                    qaoa_repeat[$qkey]="$repeat_idx"
                    qaoa_seed[$qkey]=""
                    qaoa_cache[$qkey]=""
                    qaoa_role[$qkey]="qaoa_no_warmstart"
                    pending_qaoa_keys+=("$qkey")
                done
            done
        fi
    done
done < "$jobs_file"

rm -f "$jobs_file"

echo "Benchmark configuration from ${config_file}:"
echo "${benchmark_config_dump}"

echo ""
echo "DSRI SDP/QAOA queue launcher settings:"
echo "  warm_start: ${warm_start}"
echo "  max_graph_vertices: ${max_graph_vertices:-none}"
echo "  pending_sdp_initial: ${#pending_sdp_keys[@]}"
echo "  pending_qaoa_initial: ${#pending_qaoa_keys[@]}"
echo "  sdp_max_parallel: ${sdp_max_parallel}"
echo "  qaoa_max_parallel: ${qaoa_max_parallel}"
echo "  sdp_memory_limit_total_gb: ${sdp_memory_limit_total_gb}"
echo "  sdp_memory_limit_single_gb: ${sdp_memory_limit_single_gb}"
echo "  sdp_max_retries: ${sdp_max_retries}"
echo "  sdp_threads: ${sdp_threads}"
echo "  qaoa_threads: ${qaoa_threads}"
echo "  warm_start_cache_dir: ${warm_start_cache_dir}"
echo "  warm_start_cache_max_file_mb: ${warm_start_cache_max_file_mb}"
echo "  warm_start_cache_max_total_gb: ${warm_start_cache_max_total_gb}"
echo "  warm_start_cache_producer_only: ${warm_start_cache_producer_only}"
echo "  qaoa_only_from_existing_warm_start_cache: ${qaoa_only_from_existing_warm_start_cache}"
echo "  completed_runs_txt_entries: ${#completed_map[@]}"
echo "  completed_result_csv_entries: ${#completed_csv_map[@]}"
echo "  nominal_thread_budget: ${nominal_thread_budget}"
echo "  slurm_job_id: ${slurm_job_id:-none}"
echo "  slurm_node_list: ${slurm_node_list:-none}"
echo "  slurm_cpus_per_task: ${slurm_cpus_per_task:-none}"
echo "  slurm_mem_per_node_mb: ${slurm_mem_per_node_mb:-none}"
echo "  python_bin: ${python_bin}"
echo "  MOSEKLM_LICENSE_FILE: ${MOSEKLM_LICENSE_FILE}"
echo ""

last_memory_poll=0

# -----------------------------
# Main event loop
# -----------------------------
while true; do
    refresh_running_arrays

    now_epoch=$(date +%s)
    if (( now_epoch - last_memory_poll >= sdp_memory_poll_seconds )); then
        handle_sdp_memory_pressure
        last_memory_poll="$now_epoch"
    fi

    # Launch SDP producers up to SDP cap.
    while (( ${#pending_sdp_keys[@]} > 0 && ${#running_sdp_pids[@]} < sdp_max_parallel )); do
        key="${pending_sdp_keys[1]}"
        pending_sdp_keys=("${pending_sdp_keys[@]:1}")
        launch_sdp_key "$key"
        refresh_running_arrays
        handle_sdp_memory_pressure
    done

    # Launch QAOA consumers up to QAOA cap.
    while (( ${#pending_qaoa_keys[@]} > 0 && ${#running_qaoa_pids[@]} < qaoa_max_parallel )); do
        key="${pending_qaoa_keys[1]}"
        pending_qaoa_keys=("${pending_qaoa_keys[@]:1}")
        launch_qaoa_key "$key"
        refresh_running_arrays
    done

    echo "Loop status: pending_sdp=${#pending_sdp_keys[@]}, running_sdp=${#running_sdp_pids[@]}, pending_qaoa=${#pending_qaoa_keys[@]}, running_qaoa=${#running_qaoa_pids[@]}, total_sdp_rss_mb=$(total_sdp_rss_mb), cache_size_mb=$(cache_total_mb "$warm_start_cache_dir")"

    if (( ${#pending_sdp_keys[@]} == 0 && ${#running_sdp_pids[@]} == 0 && ${#pending_qaoa_keys[@]} == 0 && ${#running_qaoa_pids[@]} == 0 )); then
        break
    fi

    sleep "$queue_poll_interval_seconds"
done

echo "All queues drained."
echo "Finished at: $(date +'%Y-%m-%d %H:%M:%S')"
echo "Run tag: ${run_tag}"
echo "Log directory: ${log_subdir}"
echo "Resource requeue CSV: ${resource_requeue_csv}"
echo "Failed warm-start CSV: ${failed_warm_start_csv}"
echo "Failed warm-start adjacency-list file: ${failed_warm_start_adjlist}"
echo "Warm-start cache dir: ${warm_start_cache_dir}"
echo "Warm-start cache size now: $(cache_total_mb "$warm_start_cache_dir") MB"
