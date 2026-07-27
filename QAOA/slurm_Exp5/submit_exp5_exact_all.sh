#!/bin/bash
set -euo pipefail

MAX_PARALLEL_JOBS=32
POLL_SECONDS=30

active_exp_jobs() {
  squeue -h -u "$USER" -t PD,R,CF,CG -o "%.200j" | awk '$1 ~ /^Exp5_/ {n++} END {print n+0}'
}

submit_when_slot_free() {
  local slurm_file="$1"
  while [ "$(active_exp_jobs)" -ge "$MAX_PARALLEL_JOBS" ]; do
    sleep "$POLL_SECONDS"
  done
  sbatch "$slurm_file"
}

submit_when_slot_free "slurm_Exp5/exact/complete_2to12/exp5_exact_complete_2to12.slurm"
submit_when_slot_free "slurm_Exp5/exact/cycle_3to12/exp5_exact_cycle_3to12.slurm"
submit_when_slot_free "slurm_Exp5/exact/path_2to12/exp5_exact_path_2to12.slurm"
submit_when_slot_free "slurm_Exp5/exact/bipartite_454/exp5_exact_bipartite_454.slurm"
submit_when_slot_free "slurm_Exp5/exact/planar_clawfree_193/exp5_exact_planar_clawfree_193.slurm"
submit_when_slot_free "slurm_Exp5/exact/regular_388/exp5_exact_regular_388.slurm"
