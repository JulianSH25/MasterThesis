#!/bin/zsh

set -u

mkdir -p logs
mkdir -p logs/COBYLA

# -----------------------------
# Parameter settings
# -----------------------------
config_file="benchmark_config.json"

iterations_list=($(jq -r '.iterations_list[]' "$config_file"))
depth_list=($(jq -r '.depth_list[]' "$config_file"))
m_start=$(jq -r '.m_start' "$config_file")
m_end=$(jq -r '.m_end' "$config_file")
time_limit_seconds=$(jq -r '.time_limit' "$config_file") # 3 hours
timeout_streak_limit=$(jq -r '.failed_instance_termination_thrsh' "$config_file")

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but was not found in PATH."
    exit 1
fi

benchmark_config_dump=$(jq -r 'to_entries[] | "  \(.key): \(.value|tojson)"' "$config_file")

# Benchmark config

main_file="Main.py"
python_bin=$(command -v python)
if [[ -z "${python_bin}" ]]; then
    echo "Error: could not find python in PATH."
    exit 1
fi

# Shared timestamp for all jobs launched by this script run.
run_timestamp=$(date +"%Y%m%d_%H%M%S")
log_subdir="logs/COBYLA/${run_timestamp}"
mkdir -p "$log_subdir"
status_subdir="${log_subdir}/status"
mkdir -p "$status_subdir"

echo "Benchmark configuration from ${config_file}:"
echo "${benchmark_config_dump}"

stop_launching=0

# Detect chip / SoC name once.
chip_name=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip|Processor Name/ {print $2; exit}')

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
min_free_ram_mb=256

# Max number of concurrent jobs.
# Default to the number of physical CPU cores, with a fallback to 4.
max_parallel=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

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
    free_pages=$(vm_stat | awk '/Pages free/ {gsub("\\.", "", $3); print $3}')
    speculative_pages=$(vm_stat | awk '/Pages speculative/ {gsub("\\.", "", $3); print $3}')
    inactive_pages=$(vm_stat | awk '/Pages inactive/ {gsub("\\.", "", $3); print $3}')

    free_pages=${free_pages:-0}
    speculative_pages=${speculative_pages:-0}
    inactive_pages=${inactive_pages:-0}

    bytes=$(( (free_pages + speculative_pages + inactive_pages) * page_size ))
    echo $(( bytes / 1024 / 1024 ))
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
        for (( m=m_start; m<=m_end; m++ )); do
            score=$(( m * p * p * iterations ))
            echo "${score} ${iterations} ${p} ${m}" >> "$jobs_file"
        done
    done
done

sort -n "$jobs_file" -o "$jobs_file"

# -----------------------------
# Helper: track launched PIDs directly
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
        timed_out=$(awk -F= '/^timed_out=/{print $2}' "$file" 2>/dev/null)
        if [[ "$timed_out" == "1" ]]; then
            streak=$(( streak + 1 ))
        else
            break
        fi
    done

    echo "$streak"
}

allowed_parallel_jobs() {
    local free_mb

    if (( ! use_ram_limit )); then
        echo "$current_parallel_cap"
        return
    fi

    free_mb=$(available_ram_mb)
    if (( free_mb <= min_free_ram_mb )); then
        echo 1
    else
        echo "$current_parallel_cap"
    fi
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
        current_parallel_cap=$(( current_parallel_cap * 2 ))
        if (( current_parallel_cap > max_parallel )); then
            current_parallel_cap=$max_parallel
        fi
    fi
}

maybe_reduce_parallel_cap() {
    local free_mb

    free_mb=$(available_ram_mb)
    if (( use_ram_limit )) && (( free_mb <= min_free_ram_mb )); then
        current_parallel_cap=1
        healthy_ram_streak=0
    fi
}

# -----------------------------
# Launch jobs
# -----------------------------
while read -r score iterations p m; do
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

    log_file="${log_subdir}/${run_timestamp}_m${m}_p${p}_it${iterations}.log"

    echo "Starting job: m=${m}, p=${p}, iterations=${iterations}, score=${score}, chip=${chip_name:-unknown}, python_bin=${python_bin}, free_ram_mb=$(available_ram_mb), allowed_parallel=${ram_limited_parallel}, current_parallel_cap=${current_parallel_cap}, healthy_ram_streak=${healthy_ram_streak}/${ram_recovery_samples_required}, tracked_jobs=$(count_running_jobs)"

    status_file="${status_subdir}/${run_timestamp}_m${m}_p${p}_it${iterations}.status"
    cmd="${python_bin} ${main_file} ${iterations} ${p} ${m} ${m} logs/COBYLA/qaoa_results_COBYLA_${run_timestamp}.csv"
    if (( use_background_mode )); then
        cmd="taskpolicy -c background ${cmd}"
    fi
    if [[ -n "$timeout_cmd" ]]; then
        cmd="${timeout_cmd} ${cmd}"
    fi

    nohup zsh -c "
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
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
            echo \"m=${m}\"
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
        echo "Warning: failed to register launched job for m=${m}, p=${p}, iterations=${iterations}. Continuing with remaining jobs."
        maybe_reduce_parallel_cap
    fi
done < "$jobs_file"

rm -f "$jobs_file"

echo "All jobs submitted."
echo "Benchmark configuration from ${config_file}:"$'\n'"${benchmark_config_dump}"
echo "Detected chip: ${chip_name:-unknown}; background mode: ${use_background_mode}; nice_value: ${nice_value}; max_parallel: ${max_parallel}; current_parallel_cap: ${current_parallel_cap}; use_ram_limit: ${use_ram_limit}; min_free_ram_mb: ${min_free_ram_mb}; ram_recovery_samples_required: ${ram_recovery_samples_required}; ram_recovery_sample_interval_seconds: ${ram_recovery_sample_interval_seconds}; timeout_streak_limit: ${timeout_streak_limit}; python_bin: ${python_bin}"