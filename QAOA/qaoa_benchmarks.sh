#!/bin/zsh

set -u

AWK_BIN=$(command -v awk 2>/dev/null || echo /usr/bin/awk)
if [[ ! -x "$AWK_BIN" ]]; then
    echo "Error: awk is required but was not found in PATH or at /usr/bin/awk."
    exit 1
fi

mkdir -p Results/logs

# -----------------------------
# Parameter settings
# -----------------------------
config_source_file="Code/benchmark_config.json"

run_timestamp=$(date +"%Y%m%d_%H%M%S")
config_snapshot_dir="Results/config_snapshots"
mkdir -p "$config_snapshot_dir"

config_file="${config_snapshot_dir}/benchmark_config_${run_timestamp}_$$.json"
cp "$config_source_file" "$config_file"

echo "Using benchmark config snapshot: ${config_file}"

rerun_exclude_finished_instances=$(jq -r '.rerun_exclude_finished_instances // false' "$config_file")
whitelist_file="runkey_whitelist.json"
whitelist=($(jq -r '.whitelist[]' "$whitelist_file"))
completed_file="completed_runs.txt"
typeset -A completed_map

if [[ "$rerun_exclude_finished_instances" == "true" && -f "$completed_file" ]]; then
    while read -r line; do
        completed_map["$line"]=1
    done < "$completed_file"
fi

iterations_list=($(jq -r '.iterations_list[]' "$config_file"))
depth_list=($(jq -r '.depth_list[]' "$config_file"))
n_start=$(jq -r '.n_start' "$config_file")
n_end=$(jq -r '.n_end' "$config_file")
time_limit_seconds=$(jq -r '.time_limit' "$config_file") # 3 hours
timeout_streak_limit=$(jq -r '.failed_instance_termination_thrsh' "$config_file")
optimiser=$(jq -r '.optimiser' "$config_file")
num_repeats=$(jq -r '.num_repeats // 1' "$config_file")

parameter_vector=$(jq -c '.parameter_vector' "$config_file")
singlet_injection=$(jq -r '.singlet_injection' "$config_file")
warm_start=$(jq -r '.warm_start' "$config_file")
warm_start_correlations=$(jq -r '.warm_start_correlations' "$config_file")
init_QAOAparams_close_to_zero=$(jq -r '.init_QAOAparams_close_to_zero' "$config_file")
use_correlations_as_initial_params=$(jq -r '.use_correlations_as_initial_params' "$config_file")
compare_with_010101=$(jq -r '.compare_with_010101' "$config_file")
start_index_singlet=$(jq -r '.start_index_singlet' "$config_file")
graph_generation_type=$(jq -r '.graph_generation_type' "$config_file")
weighted=$(jq -r '.weighted' "$config_file")
relative_graph_adjList_path=$(jq -r '.relative_graph_adjList_path // empty' "$config_file")

cpu_util_threshold=$(jq -r '.cpu_util_threshold // 85' "$config_file")
use_cpu_limit=0
if [[ "${optimiser:l}" == "adam" ]]; then
    use_cpu_limit=1
fi

mkdir -p Results/logs/${optimiser}

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but was not found in PATH."
    exit 1
fi

benchmark_config_dump=$(jq -r 'to_entries[] | "  \(.key): \(.value|tojson)"' "$config_file")

# Benchmark config

main_file="Code/Main.py"
python_bin=$(command -v python)
if [[ -z "${python_bin}" ]]; then
    echo "Error: could not find python in PATH."
    exit 1
fi

# Shared timestamp for all jobs launched by this script run.
log_subdir="Results/logs/${optimiser}/${run_timestamp}"
mkdir -p "$log_subdir"
status_subdir="${log_subdir}/status"
mkdir -p "$status_subdir"

echo "Benchmark configuration from ${config_file}:"
echo "${benchmark_config_dump}"

stop_launching=0

# Detect chip / SoC name once.
chip_name=$(system_profiler SPHardwareDataType 2>/dev/null | "$AWK_BIN" -F': ' '/Chip|Processor Name/ {print $2; exit}')

# Detect core count for thread settings.
# Prefer physical cores for compute-heavy BLAS operations.
physical_cores=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "")
logical_cores=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "")
if [[ -z "${physical_cores}" || "${physical_cores}" == "0" ]]; then
    physical_cores=${logical_cores}
fi
if [[ -z "${physical_cores}" || "${physical_cores}" == "0" ]]; then
    physical_cores=1
fi
blas_threads=${physical_cores}

# Run jobs in a lower-priority / efficiency-oriented mode on selected chips.
# taskpolicy supports background QoS clamps on macOS.
use_background_mode=0
nice_value=-20
if [[ "${chip_name}" == *"A18 Pro"* ]]; then
    use_background_mode=1
    nice_value=10
fi

# If we are not root, negative nice values are not permitted.
# Fall back to default priority on non-Neo machines in that case.
if (( EUID != 0 )) && (( nice_value < 0 )); then
    echo "Warning: negative nice requires sudo/root. Falling back to nice_value=0."
    nice_value=0
fi

# Optional RAM-based launch limiting.
# Set use_ram_limit=1 to enable it.
use_ram_limit=1

# Once free RAM falls to or below this threshold, only one parallel job is allowed.
# Minimum free RAM to keep available before launching another job.
min_free_ram_mb=1024

# Max number of concurrent jobs.
# Set to 1 for sequential execution to avoid system crashes with OpenBLAS.
# OpenBLAS is compiled with USE_OPENMP=0, causing resource contention in parallel mode.
max_parallel=1

# Gradually ramp up concurrency instead of immediately jumping to max_parallel.
# This reduces the chance of suddenly launching many large-RAM jobs at once.
current_parallel_cap=1
# Require several consecutive "healthy RAM" samples before increasing the cap again.
ram_recovery_samples_required=15
ram_recovery_sample_interval_seconds=1
healthy_ram_streak=0


available_ram_mb() {
    local page_size free_pages speculative_pages inactive_pages bytes
    page_size=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
    free_pages=$(vm_stat | "$AWK_BIN" '/Pages free/ {gsub("\\.", "", $3); print $3}')
    speculative_pages=$(vm_stat | "$AWK_BIN" '/Pages speculative/ {gsub("\\.", "", $3); print $3}')
    inactive_pages=$(vm_stat | "$AWK_BIN" '/Pages inactive/ {gsub("\\.", "", $3); print $3}')

    free_pages=${free_pages:-0}
    speculative_pages=${speculative_pages:-0}
    inactive_pages=${inactive_pages:-0}

    bytes=$(( (free_pages + speculative_pages + inactive_pages) * page_size ))
    echo $(( bytes / 1024 / 1024 ))
}

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

# -----------------------------
# Pick timeout command
# -----------------------------
timeout_cmd=""
if (( time_limit_seconds > 0 )); then
    if command -v gtimeout >/dev/null 2>&1; then
        timeout_cmd="gtimeout ${time_limit_seconds}s"
    elif command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout ${time_limit_seconds}s"
    else
        echo "Warning: no timeout command found. Continuing without time limits."
        echo "On macOS, install coreutils and use gtimeout."
        timeout_cmd=""
    fi
fi

# -----------------------------
# Build and sort jobs
# Columns:
# score iterations p m
# -----------------------------
jobs_file="$(mktemp)"

for iterations in "${iterations_list[@]}"; do
    for p in "${depth_list[@]}"; do
        if [[ "$graph_generation_type" == "HOG" ]]; then
            if [[ -z "Code/$relative_graph_adjList_path" ]]; then
                echo "Error: relative_graph_adjList_path must be set for graph_generation_type=HOG."
                exit 1
            fi

            hog_graph_count=$(count_hog_graphs "Code/$relative_graph_adjList_path")
            if [[ -z "$hog_graph_count" || "$hog_graph_count" == "0" ]]; then
                echo "Error: no HOG graphs found in Code/$relative_graph_adjList_path."
                exit 1
            fi

            n_start=0
            n_end=$((hog_graph_count - 1))
        fi

        for (( n=n_start; n<=n_end; n++ )); do
            score=$(( n * p * p * iterations ))
            echo "${score} ${iterations} ${p} ${n}" >> "$jobs_file"
        done
    done
done

easiest_first=$(jq -r '.easiest_first // false' "$config_file")

if [[ "$easiest_first" == "true" ]]; then
    sort -n "$jobs_file" -o "$jobs_file"
else
    sort -nr "$jobs_file" -o "$jobs_file"
fi

# Master repeat loop
for (( repeat_idx=1; repeat_idx<=num_repeats; repeat_idx++ )); do
    echo ""
    echo "===================="
    echo "Repeat run: ${repeat_idx}/${num_repeats}"
    echo "===================="
    echo ""
    
    # Reset stop_launching flag for each repeat
    stop_launching=0
    
    # Compute derived seed for this repeat (if sdp_seed is configured in config)
    configured_sdp_seed=$(jq -r '.sdp_seed // empty' "$config_file")
    derived_sdp_seed=""
    if [[ -n "$configured_sdp_seed" ]]; then
        seed_increment_per_repeat=1000000
        derived_sdp_seed=$(( configured_sdp_seed + (repeat_idx - 1) * seed_increment_per_repeat ))
        echo "Derived SDP seed for repeat ${repeat_idx}: ${derived_sdp_seed} (base: ${configured_sdp_seed}, increment: ${seed_increment_per_repeat})"
    fi

# -----------------------------
# Helper: track launched PIDs directly
# -----------------------------
typeset -a running_pids=()

cpu_usage_percent() {
    # sample CPU twice and compute usage
    local usage
    usage=$(top -l 2 -n 0 | grep "CPU usage" | tail -n 1 | "$AWK_BIN" '{print $3}' | sed 's/%//')
    echo "${usage:-0}"
}

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

allowed_parallel_jobs() {
    local free_mb cpu_usage

    free_mb=$(available_ram_mb)

    # RAM constraint
    if (( use_ram_limit )) && (( free_mb <= min_free_ram_mb )); then
        echo 1
        return
    fi

    # CPU constraint (only for Adam)
    if (( use_cpu_limit )); then
        cpu_usage=$(cpu_usage_percent)
        if (( cpu_usage >= cpu_util_threshold )); then
            echo 1
            return
        fi
    fi

    # Default case
    echo "$current_parallel_cap"
}

maybe_increase_parallel_cap() {
    local free_mb

    free_mb=$(available_ram_mb)
    if (( use_ram_limit )) && (( free_mb <= min_free_ram_mb )); then
        current_parallel_cap=1
        healthy_ram_streak=0
        return
    fi

    healthy_ram_streak=$(( healthy_ram_streak + 1 ))
    if (( healthy_ram_streak < ram_recovery_samples_required )); then
        return
    fi

    healthy_ram_streak=0
    if (( current_parallel_cap < max_parallel )); then
        current_parallel_cap=$(( current_parallel_cap + 1 ))
    fi
}

maybe_reduce_parallel_cap() {
    local free_mb

    free_mb=$(available_ram_mb)
    if (( use_ram_limit )) && (( free_mb <= min_free_ram_mb )); then
        current_parallel_cap=1
        healthy_ram_streak=0
    fi

    cpu_usage=$(cpu_usage_percent)
    if (( use_cpu_limit )) && (( cpu_usage >= cpu_util_threshold )); then
        current_parallel_cap=1
        healthy_ram_streak=0
    fi
}

normalize() {
    local v="$1"

    # Python: None -> ""
    if [[ -z "$v" || "$v" == "null" ]]; then
        echo ""
        return
    fi

    # Try JSON normalization (like json.dumps(sort_keys=True))
    if echo "$v" | jq -e . >/dev/null 2>&1; then
        echo "$v" | jq -c -S .
    else
        # fallback: string strip
        echo "$v" | "$AWK_BIN" '{$1=$1;print}'
    fi
}

compute_m() {
    local n="$1"
    case "$graph_generation_type" in
        line) echo $((n - 1)) ;;
        cycle) echo "$n" ;;
        complete) echo $((n * (n - 1) / 2)) ;;
        HOG) echo "n/a" ;;
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
            precision/iterations) value="$iterations" ;;  # IMPORTANT: match whitelist exactly
            parameter_vector) value="$parameter_vector" ;;
            singlet_injection) value="$singlet_injection" ;;
            warm_start) value="$warm_start" ;;
            #warm_start_correlations) value="$warm_start_correlations" ;;
            init_QAOAparams_close_to_zero) value="$init_QAOAparams_close_to_zero" ;;
            use_correlations_as_initial_params) value="$use_correlations_as_initial_params" ;;
            compare_with_010101) value="$compare_with_010101" ;;
            start_index_singlet) value="$start_index_singlet" ;;
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
        echo "SH: $key_string"
    done

    # SHA1 like Python
    echo -n "$key_string" | shasum | "$AWK_BIN" '{print $1}'
}

# -----------------------------
# Launch jobs
# -----------------------------
    while read -r score iterations p n; do
        if (( stop_launching )); then
            echo "Stopping further job launches because the timeout streak limit was reached."
            break
        fi
        while true; do
            maybe_reduce_parallel_cap
            maybe_increase_parallel_cap
            ram_limited_parallel=$(allowed_parallel_jobs)
            ram_limited_parallel=${ram_limited_parallel:-0}

            current_jobs=$(count_running_jobs)
            current_jobs=${current_jobs:-0}

            timeout_streak=$(current_timeout_streak)
            timeout_streak=${timeout_streak:-0}
        if (( timeout_streak >= timeout_streak_limit )); then
            echo "Timeout streak limit reached (${timeout_streak}/${timeout_streak_limit}). Stopping further job launches."
            stop_launching=1
            break 2
        fi

        echo "Queue check: current_jobs=${current_jobs}, allowed_parallel=${ram_limited_parallel}, current_parallel_cap=${current_parallel_cap}, healthy_ram_streak=${healthy_ram_streak}/${ram_recovery_samples_required}, tracked_pids=${#running_pids[@]}, free_ram_mb=$(available_ram_mb)"

        if (( current_jobs < ram_limited_parallel )); then
            break
        fi

        sleep ${ram_recovery_sample_interval_seconds}
    done

    run_key=$(build_run_key "$n" "$p" "$iterations")

    if [[ "$rerun_exclude_finished_instances" == "true" ]]; then
        if [[ -n "${completed_map[$run_key]:-}" ]]; then
            echo "Skipping already completed job: n=${n}, p=${p}, iterations=${iterations}"
            continue
        fi
    fi

    log_file="${log_subdir}/${run_timestamp}_n${n}_p${p}_it${iterations}.log"

    echo "Starting job: n=${n}, p=${p}, iterations=${iterations}, score=${score}, chip=${chip_name:-unknown}, python_bin=${python_bin}, free_ram_mb=$(available_ram_mb), allowed_parallel=${ram_limited_parallel}, current_parallel_cap=${current_parallel_cap}, healthy_ram_streak=${healthy_ram_streak}/${ram_recovery_samples_required}, tracked_jobs=$(count_running_jobs)"

    status_file="${status_subdir}/${run_timestamp}_n${n}_p${p}_it${iterations}.status"
    cmd="${python_bin} ${main_file} ${iterations} ${p} ${n} ${n} Results/logs/${optimiser}/qaoa_results_${optimiser}_${run_timestamp}"
    if (( use_background_mode )); then
        cmd="taskpolicy -c background ${cmd}"
    fi
    if [[ -n "$timeout_cmd" ]]; then
        cmd="${timeout_cmd} ${cmd}"
    fi

    nohup zsh -c "
        export OMP_NUM_THREADS=${blas_threads}
        export OPENBLAS_NUM_THREADS=${blas_threads}
        export MKL_NUM_THREADS=${blas_threads}
        export NUMEXPR_NUM_THREADS=${blas_threads}
        export BENCHMARK_CONFIG_FILE="${config_file}"
        if [[ -n \"${derived_sdp_seed}\" ]]; then
            export SDP_SEED_OVERRIDE=${derived_sdp_seed}
        fi
        nice -n ${nice_value} ${cmd}
        exit_code=\$?
        timed_out=0
        if [[ -n \"${timeout_cmd}\" ]] && (( exit_code == 124 )); then
            timed_out=1
        fi
        {
            echo \"finished_at=\$(date +'%Y-%m-%d %H:%M:%S')\"
            echo \"exit_code=\${exit_code}\"
            echo \"timed_out=\${timed_out}\"
            echo \"n=${n}\"
            echo \"p=${p}\"
            echo \"iterations=${iterations}\"
        } > \"${status_file}\"
        exit \${exit_code}
    " > "${log_file}" 2>&1 &

    launched_pid=$!
    if [[ -n "${launched_pid:-}" ]] && kill -0 "$launched_pid" 2>/dev/null; then
        running_pids+=("$launched_pid")
        echo "Launched PID: ${launched_pid}; current_parallel_cap=${current_parallel_cap}"
    else
        echo "Warning: failed to register launched job for n=${n}, p=${p}, iterations=${iterations}. Continuing with remaining jobs."
        maybe_reduce_parallel_cap
    fi
    done < "$jobs_file"

    echo "Completed jobs for repeat run: ${repeat_idx}/${num_repeats}"
    echo ""
done

rm -f "$jobs_file"

echo "All repeat runs submitted (total repeats: ${num_repeats})."
echo "Benchmark configuration from ${config_file}:"$'\n'"${benchmark_config_dump}"
echo "Detected chip: ${chip_name:-unknown}; physical_cores: ${physical_cores}; logical_cores: ${logical_cores}; blas_threads: ${blas_threads}; background mode: ${use_background_mode}; nice_value: ${nice_value}; max_parallel: ${max_parallel}; current_parallel_cap: ${current_parallel_cap}; use_ram_limit: ${use_ram_limit}; min_free_ram_mb: ${min_free_ram_mb}; ram_recovery_samples_required: ${ram_recovery_samples_required}; ram_recovery_sample_interval_seconds: ${ram_recovery_sample_interval_seconds}; timeout_streak_limit: ${timeout_streak_limit}; python_bin: ${python_bin}"