#!/usr/bin/env zsh

set -u

echo "Launcher PID: $$"
echo "Started at: $(date +'%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

export MOSEKLM_LICENSE_FILE="/home/coder/project/MasterThesis/mosek.lic"

AWK_BIN=$(command -v awk 2>/dev/null || echo /usr/bin/awk)
if [[ ! -x "$AWK_BIN" ]]; then
    echo "Error: awk is required but was not found in PATH or at /usr/bin/awk."
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but was not found in PATH."
    echo "Install with: conda install -c conda-forge jq -y"
    exit 1
fi

if command -v sha1sum >/dev/null 2>&1; then
    SHA1_CMD="sha1sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA1_CMD="shasum"
else
    echo "Error: sha1sum or shasum is required but was not found."
    exit 1
fi

mkdir -p Results/logs

# -----------------------------
# Usage
# -----------------------------
if (( $# != 1 )); then
    echo "Usage: $0 <config-file>"
    echo ""
    echo "Examples:"
    echo "  $0 benchmark_config_exact.json"
    echo "  $0 run_configurations/benchmark_config.json"
    echo "  $0 Code/benchmark_config_exact.json"
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

run_timestamp=$(date +"%Y%m%d_%H%M%S")
run_tag="${run_timestamp}_pid$$"
config_snapshot_dir="Results/config_snapshots"
mkdir -p "$config_snapshot_dir"

config_file="${config_snapshot_dir}/benchmark_config_${run_tag}.json"
cp "$config_source_file" "$config_file"

echo "Using benchmark config snapshot: ${config_file}"

rerun_exclude_finished_instances=$(jq -r '.rerun_exclude_finished_instances // false' "$config_file")

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

optimiser=$(jq -r '.optimiser' "$config_file")

if [[ "${optimiser:l}" == "exact" ]]; then
    iterations_list=(0)
    depth_list=(0)
else
    iterations_list=($(jq -r '.iterations_list[]' "$config_file"))
    depth_list=($(jq -r '.depth_list[]' "$config_file"))
fi

n_start_raw=$(jq -r '.n_start // empty' "$config_file")
n_end_raw=$(jq -r '.n_end // empty' "$config_file")
time_limit_seconds=$(jq -r '.time_limit // 0' "$config_file")

timeout_streak_limit_raw=$(jq -r '.failed_instance_termination_thrsh // empty' "$config_file")
timeout_streak_limit_enabled=1
if [[ -z "$timeout_streak_limit_raw" || "$timeout_streak_limit_raw" == "null" ]]; then
    timeout_streak_limit_enabled=0
    timeout_streak_limit=0
else
    timeout_streak_limit="$timeout_streak_limit_raw"
    if (( timeout_streak_limit <= 0 )); then
        timeout_streak_limit_enabled=0
    fi
fi

num_repeats=$(jq -r '.num_repeats // 1' "$config_file")

parameter_vector=$(jq -c '.parameter_vector' "$config_file")
singlet_injection=$(jq -r '.singlet_injection // false' "$config_file")
warm_start=$(jq -r '.warm_start // false' "$config_file")
warm_start_correlations=$(jq -r '.warm_start_correlations // empty' "$config_file")
init_QAOAparams_close_to_zero=$(jq -r '.init_QAOAparams_close_to_zero // false' "$config_file")
use_correlations_as_initial_params=$(jq -r '.use_correlations_as_initial_params // false' "$config_file")
compare_with_010101=$(jq -r '.compare_with_010101 // false' "$config_file")
start_index_singlet=$(jq -r '.start_index_singlet // empty' "$config_file")
circuit_type=$(jq -r '.circuit_type // empty' "$config_file")
graph_generation_type=$(jq -r '.graph_generation_type // empty' "$config_file")
weighted=$(jq -r '.weighted // false' "$config_file")
relative_graph_adjList_path=$(jq -r '.relative_graph_adjList_path // empty' "$config_file")

# -----------------------------
# HOG helpers
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

    n_start=0
    n_end=$((hog_graph_count - 1))

    echo "HOG circuit detected. Using graph indices n=${n_start}..${n_end} from ${hog_graph_path}."
else
    if [[ -z "$n_start_raw" || -z "$n_end_raw" ]]; then
        echo "Error: n_start and n_end must be set for graph_generation_type=${graph_generation_type}."
        exit 1
    fi

    n_start="$n_start_raw"
    n_end="$n_end_raw"
fi

# -----------------------------
# DSRI/Linux resource settings
# -----------------------------

max_parallel=$(jq -r '.max_parallel // 1' "$config_file")
queue_poll_interval_seconds=$(jq -r '.queue_poll_interval_seconds // 2' "$config_file")


blas_threads=$(jq -r '.blas_threads // 8' "$config_file")

chip_name=$(lscpu 2>/dev/null | "$AWK_BIN" -F': +' '/Model name/ {print $2; exit}')
logical_cores=$(nproc 2>/dev/null || echo 1)
physical_cores="$logical_cores"

available_ram_mb() {
    local mem_limit mem_current mem_available_kb

    # cgroup v2: container-aware memory limit
    if [[ -f /sys/fs/cgroup/memory.max && -f /sys/fs/cgroup/memory.current ]]; then
        mem_limit=$(cat /sys/fs/cgroup/memory.max)
        mem_current=$(cat /sys/fs/cgroup/memory.current)

        if [[ "$mem_limit" != "max" && -n "$mem_limit" && -n "$mem_current" ]]; then
            echo $(( (mem_limit - mem_current) / 1024 / 1024 ))
            return
        fi
    fi

    # fallback: node-level available memory
    mem_available_kb=$("$AWK_BIN" '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
    if [[ -n "$mem_available_kb" ]]; then
        echo $(( mem_available_kb / 1024 ))
    else
        echo 999999
    fi
}

mkdir -p Results/logs/${optimiser}

benchmark_config_dump=$(jq -r 'to_entries[] | "  \(.key): \(.value|tojson)"' "$config_file")

main_file="Code/Main.py"
python_bin=$(command -v python || true)
if [[ -z "${python_bin}" ]]; then
    echo "Error: could not find python in PATH."
    exit 1
fi

log_subdir="Results/logs/${optimiser}/${run_tag}"
mkdir -p "$log_subdir"

status_subdir="${log_subdir}/status"
mkdir -p "$status_subdir"

warm_start_cache_subdir="${log_subdir}/warm_start_cache"
mkdir -p "$warm_start_cache_subdir"

failed_subdir="${log_subdir}/failed_warm_starts"
mkdir -p "$failed_subdir"

failed_warm_start_csv="${failed_subdir}/failed_warm_starts_${run_tag}.csv"
failed_warm_start_adjlist="${failed_subdir}/failed_warm_starts_${run_tag}.adjlist"

if [[ ! -f "$failed_warm_start_csv" ]]; then
    echo "run_tag,failed_at,n,p,iterations,repeat_idx,sdp_seed,exit_code,reason,warm_start_cache_path,log_file,status_file,graph_generation_type,adjacency_list_file" > "$failed_warm_start_csv"
fi

typeset -A failed_adjlist_written=()

csv_escape() {
    local value="$1"
    value=${value//\"/\"\"}
    echo "\"${value}\""
}

append_hog_graph_block_to_failed_adjlist() {
    local graph_index="$1"
    local graph_key="hog_${graph_index}"

    if [[ -n "${failed_adjlist_written[$graph_key]:-}" ]]; then
        return
    fi

    if [[ "${graph_generation_type:l}" != "hog" ]]; then
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

log_failed_warm_start() {
    local n_value="$1"
    local p_value="$2"
    local iterations_value="$3"
    local repeat_idx_value="$4"
    local sdp_seed_value="$5"
    local exit_code_value="$6"
    local reason_value="$7"
    local warm_start_cache_path_value="$8"
    local log_file_value="$9"
    local status_file_value="${10}"

    local failed_at
    failed_at=$(date +'%Y-%m-%d %H:%M:%S')

    {
        csv_escape "$run_tag"; printf ","
        csv_escape "$failed_at"; printf ","
        csv_escape "$n_value"; printf ","
        csv_escape "$p_value"; printf ","
        csv_escape "$iterations_value"; printf ","
        csv_escape "$repeat_idx_value"; printf ","
        csv_escape "$sdp_seed_value"; printf ","
        csv_escape "$exit_code_value"; printf ","
        csv_escape "$reason_value"; printf ","
        csv_escape "$warm_start_cache_path_value"; printf ","
        csv_escape "$log_file_value"; printf ","
        csv_escape "$status_file_value"; printf ","
        csv_escape "$graph_generation_type"; printf ","
        csv_escape "$failed_warm_start_adjlist"; printf "\n"
    } >> "$failed_warm_start_csv"

    append_hog_graph_block_to_failed_adjlist "$n_value"
}

echo "Benchmark configuration from ${config_file}:"
echo "${benchmark_config_dump}"

echo ""
echo "DSRI resource settings:"
echo "  node_model: ${chip_name:-unknown}"
echo "  visible_logical_cores: ${logical_cores}"
echo "  blas_threads: ${blas_threads}"
echo "  max_parallel: ${max_parallel}"
echo "  queue_poll_interval_seconds: ${queue_poll_interval_seconds}"
echo "  available_ram_mb_now_for_logging_only: $(available_ram_mb)"
echo "  python_bin: ${python_bin}"
echo ""

stop_launching=0

# -----------------------------
# Timeout command
# -----------------------------
timeout_cmd=""
if (( time_limit_seconds > 0 )); then
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout ${time_limit_seconds}s"
    else
        echo "Warning: no timeout command found. Continuing without time limits."
        timeout_cmd=""
    fi
fi

# -----------------------------
# Build sorted instance list
# -----------------------------
jobs_file="$(mktemp)"

for (( n=n_start; n<=n_end; n++ )); do
    score=$(( n + 1 ))
    echo "${score} ${n}" >> "$jobs_file"
done

easiest_first=$(jq -r '.easiest_first // false' "$config_file")

if [[ "$easiest_first" == "true" ]]; then
    sort -n "$jobs_file" -o "$jobs_file"
else
    sort -nr "$jobs_file" -o "$jobs_file"
fi

build_warm_start_cache_path() {
    local n="$1"
    local repeat_idx_value="$2"
    local sdp_seed_value="$3"

    local seed_label="noseed"
    if [[ -n "$sdp_seed_value" ]]; then
        seed_label="seed${sdp_seed_value}"
    fi

    local mode
    local level
    local initial_m
    mode=$(jq -r '.warm_start_mode // "standard"' "$config_file")
    level=$(jq -r '.lasserre_level // "NA"' "$config_file")
    initial_m=$(jq -r '.initial_solver_level_M // "NA"' "$config_file")

    echo "${warm_start_cache_subdir}/${run_tag}_n${n}_rep${repeat_idx_value}_${seed_label}_L${level}_M${initial_m}_${mode}.npz"
}

# -----------------------------
# Process tracking
# -----------------------------
typeset -a running_pids=()

refresh_running_pids() {
    local pid
    local -a still_running=()

    for pid in "${running_pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            still_running+=("$pid")
        fi
    done

    running_pids=("${still_running[@]}")
}

count_running_jobs() {
    refresh_running_pids
    echo "${#running_pids[@]}"
}

current_timeout_streak() {
    local file timed_out streak
    streak=0

    for file in $(ls -t "$status_subdir"/*.status 2>/dev/null); do
        timed_out=$("$AWK_BIN" -F= '/^timed_out=/{print $2}' "$file" 2>/dev/null)
        if [[ "$timed_out" == "1" ]]; then
            streak=$(( streak + 1 ))
        else
            break
        fi
    done

    echo "$streak"
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

build_run_key() {
    local n="$1"
    local p="$2"
    local iterations="$3"

    local m
    m=$(compute_m "$n")

    local key_string=""
    local key value norm_value

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

    echo -n "$key_string" | $SHA1_CMD | "$AWK_BIN" '{print $1}'
}

# -----------------------------
# Launch jobs
# -----------------------------
while read -r score n; do
    if (( stop_launching )); then
        echo "Stopping further job launches because the timeout streak limit was reached."
        break
    fi

    for (( repeat_idx=1; repeat_idx<=num_repeats; repeat_idx++ )); do
        configured_sdp_seed=$(jq -r '.sdp_seed // empty' "$config_file")
        derived_sdp_seed=""

        if [[ -n "$configured_sdp_seed" ]]; then
            seed_increment_per_repeat=1
            derived_sdp_seed=$(( configured_sdp_seed + (repeat_idx - 1) * seed_increment_per_repeat ))
            echo "Derived SDP seed for repeat ${repeat_idx}: ${derived_sdp_seed} (base: ${configured_sdp_seed}, increment: ${seed_increment_per_repeat})"
        fi

        warm_start_cache_path=""
        if [[ "${warm_start:l}" == "true" ]]; then
            warm_start_cache_path=$(build_warm_start_cache_path "$n" "$repeat_idx" "$derived_sdp_seed")
            rm -f "$warm_start_cache_path" "${warm_start_cache_path}.tmp" "${warm_start_cache_path}.failed"
            echo "Warm-start cache for n=${n}, repeat=${repeat_idx}, seed=${derived_sdp_seed:-none}: ${warm_start_cache_path}"
        fi

        seed_label="noseed"
        if [[ -n "${derived_sdp_seed}" ]]; then
            seed_label="seed${derived_sdp_seed}"
        fi

        warm_start_failed=0

        for iterations in "${iterations_list[@]}"; do
            for p in "${depth_list[@]}"; do
                if (( warm_start_failed )); then
                    echo "Skipping n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat_idx} because warm-start generation failed for this graph+seed."
                    continue
                fi

                echo ""
                echo "===================="
                echo "Benchmark setting: n=${n}, p=${p}, iterations=${iterations}"
                echo "Repeat run: ${repeat_idx}/${num_repeats}"
                echo "===================="
                echo ""

                while true; do
                    current_jobs=$(count_running_jobs)
                    current_jobs=${current_jobs:-0}

                    timeout_streak=$(current_timeout_streak)
                    timeout_streak=${timeout_streak:-0}

                    if (( timeout_streak_limit_enabled && timeout_streak >= timeout_streak_limit )); then
                        echo "Timeout streak limit reached (${timeout_streak}/${timeout_streak_limit}). Stopping further job launches."
                        stop_launching=1
                        break 3
                    fi

                    echo "Queue check: current_jobs=${current_jobs}, max_parallel=${max_parallel}, tracked_pids=${#running_pids[@]}, available_ram_mb=$(available_ram_mb)"

                    if (( current_jobs < max_parallel )); then
                        break
                    fi

                    sleep ${queue_poll_interval_seconds}
                done

                run_key=$(build_run_key "$n" "$p" "$iterations")

                if [[ "$rerun_exclude_finished_instances" == "true" ]]; then
                    if [[ -n "${completed_map[$run_key]:-}" ]]; then
                        echo "Skipping already completed job: n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat_idx}"
                        continue
                    fi
                fi

                log_file="${log_subdir}/${run_tag}_n${n}_p${p}_it${iterations}_rep${repeat_idx}_${seed_label}.log"
                status_file="${status_subdir}/${run_tag}_n${n}_p${p}_it${iterations}_rep${repeat_idx}_${seed_label}.status"

                echo "Starting job: n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat_idx}, seed=${derived_sdp_seed:-none}, score=${score}, python_bin=${python_bin}, available_ram_mb=$(available_ram_mb), max_parallel=${max_parallel}, tracked_jobs=$(count_running_jobs)"

                cmd="${python_bin} ${main_file} ${iterations} ${p} ${n} ${n} Results/logs/${optimiser}/qaoa_results_${optimiser}_${run_tag}"

                if [[ -n "$timeout_cmd" ]]; then
                    cmd="${timeout_cmd} ${cmd}"
                fi

                nohup zsh -c "
                    export OMP_NUM_THREADS=${blas_threads}
                    export OPENBLAS_NUM_THREADS=${blas_threads}
                    export MKL_NUM_THREADS=${blas_threads}
                    export NUMEXPR_NUM_THREADS=${blas_threads}
                    export BENCHMARK_CONFIG_FILE=\"${config_file}\"
                    export BENCHMARK_REPEAT_IDX=\"${repeat_idx}\"

                    if [[ -n \"${derived_sdp_seed}\" ]]; then
                        export SDP_SEED_OVERRIDE=${derived_sdp_seed}
                    fi

                    if [[ -n \"${warm_start_cache_path}\" ]]; then
                        export WARM_START_CACHE_PATH=\"${warm_start_cache_path}\"
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
                        echo \"n=${n}\"
                        echo \"p=${p}\"
                        echo \"iterations=${iterations}\"
                        echo \"repeat_idx=${repeat_idx}\"
                        echo \"sdp_seed=${derived_sdp_seed}\"
                        echo \"warm_start_cache_path=${warm_start_cache_path}\"
                    } > \"${status_file}\"

                    exit \${exit_code}
                " > "${log_file}" 2>&1 &

                launched_pid=$!

                if [[ -n "${launched_pid:-}" ]] && kill -0 "$launched_pid" 2>/dev/null; then
                    running_pids+=("$launched_pid")
                    echo "Launched PID: ${launched_pid}"
                else
                    echo "Warning: failed to register launched job for n=${n}, p=${p}, iterations=${iterations}, repeat=${repeat_idx}. Continuing with remaining jobs."
                    continue
                fi
            done
        done

        if [[ -n "${warm_start_cache_path}" ]]; then
            rm -f "$warm_start_cache_path" "${warm_start_cache_path}.tmp"
            echo "Deleted warm-start cache for n=${n}, repeat=${repeat_idx}, seed=${derived_sdp_seed:-none}: ${warm_start_cache_path}"
        fi

        if (( stop_launching )); then
            break
        fi
    done
done < "$jobs_file"

rm -f "$jobs_file"

echo "All jobs launched. Waiting for remaining ${#running_pids[@]} jobs to finish..."

for pid in "${running_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
    fi
done

refresh_running_pids
echo "Remaining tracked jobs after final wait: ${#running_pids[@]}"
echo "Benchmark configuration from ${config_file}:"$'\n'"${benchmark_config_dump}"
echo "Run tag: ${run_tag}"
echo "Failed warm-start CSV: ${failed_warm_start_csv}"
echo "Failed warm-start adjacency-list retry file: ${failed_warm_start_adjlist}"
echo "Detected node model: ${chip_name:-unknown}; visible_logical_cores: ${logical_cores}; blas_threads: ${blas_threads}; max_parallel: ${max_parallel}; queue_poll_interval_seconds: ${queue_poll_interval_seconds}; timeout_streak_limit: ${timeout_streak_limit}; timeout_streak_limit_enabled: ${timeout_streak_limit_enabled}; python_bin: ${python_bin}"
