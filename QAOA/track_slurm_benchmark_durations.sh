#!/usr/bin/env bash

set -u

if (( $# < 1 || $# > 3 )); then
    echo "Usage: $0 JOB_NAME_REGEX [OUTPUT_TSV] [POLL_SECONDS]" >&2
    exit 2
fi

job_name_regex=$1
output_tsv=${2:-Results/slurm/benchmark_durations.tsv}
poll_seconds=${3:-60}

if ! [[ $poll_seconds =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer." >&2
    exit 2
fi

mkdir -p "$(dirname "$output_tsv")"

declare -A tracked_jobs=()
declare -A recorded_jobs=()

if [[ -s $output_tsv ]]; then
    while IFS=$'\t' read -r job_id _; do
        [[ $job_id == "job_id" || -z $job_id ]] && continue
        recorded_jobs["$job_id"]=1
    done < "$output_tsv"
else
    printf 'job_id\tjob_name\tstate\tsubmit\tstart\tend\telapsed\telapsed_seconds\tnode_list\texit_code\n' \
        > "$output_tsv"
fi

terminal_state() {
    case ${1%% *} in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

echo "Tracking jobs matching /$job_name_regex/ every ${poll_seconds}s."
echo "Writing completed job timings to $output_tsv"

seen_any=false

while true; do
    active_matches=0

    while IFS='|' read -r job_id job_name; do
        job_id=${job_id//[[:space:]]/}
        job_name=${job_name%"${job_name##*[![:space:]]}"}
        [[ -z $job_id ]] && continue

        if [[ $job_name =~ $job_name_regex && $job_name != submit_* ]]; then
            tracked_jobs["$job_id"]=$job_name
            seen_any=true
            ((active_matches += 1))
        fi
    done < <(squeue -h -u "$USER" -t PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED -o "%i|%200j")

    unfinished=0
    for job_id in "${!tracked_jobs[@]}"; do
        [[ -n ${recorded_jobs[$job_id]+x} ]] && continue

        record=$(sacct -n -P -j "$job_id" \
            --format=JobIDRaw,JobName%200,State,Submit,Start,End,Elapsed,ElapsedRaw,NodeList%100,ExitCode \
            | awk -F'|' -v id="$job_id" '$1 == id { print; exit }')

        if [[ -z $record ]]; then
            ((unfinished += 1))
            continue
        fi

        IFS='|' read -r account_job_id job_name state submit start end elapsed elapsed_raw node_list exit_code <<< "$record"
        state=${state%+}

        if ! terminal_state "$state"; then
            ((unfinished += 1))
            continue
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$account_job_id" "$job_name" "$state" "$submit" "$start" "$end" \
            "$elapsed" "$elapsed_raw" "$node_list" "$exit_code" >> "$output_tsv"
        recorded_jobs["$job_id"]=1
        echo "$(date -Is)  $job_id  $job_name  $state  $elapsed"
    done

    if $seen_any && (( active_matches == 0 && unfinished == 0 )); then
        echo "All tracked benchmark jobs have finished."
        exit 0
    fi

    sleep "$poll_seconds"
done
