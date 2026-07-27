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

submit_when_slot_free "slurm_Exp5/sdp_cache/complete_2to12/exp5_sdp_cache_complete_2to12_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/complete_2to12/exp5_sdp_cache_complete_2to12_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/complete_2to12/exp5_sdp_cache_complete_2to12_L2M2.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/cycle_3to12/exp5_sdp_cache_cycle_3to12_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/cycle_3to12/exp5_sdp_cache_cycle_3to12_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/cycle_3to12/exp5_sdp_cache_cycle_3to12_L2M2.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/path_2to12/exp5_sdp_cache_path_2to12_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/path_2to12/exp5_sdp_cache_path_2to12_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/path_2to12/exp5_sdp_cache_path_2to12_L2M2.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/bipartite_454/exp5_sdp_cache_bipartite_454_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/bipartite_454/exp5_sdp_cache_bipartite_454_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/bipartite_454/exp5_sdp_cache_bipartite_454_L2M2.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/planar_clawfree_193/exp5_sdp_cache_planar_clawfree_193_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/planar_clawfree_193/exp5_sdp_cache_planar_clawfree_193_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/planar_clawfree_193/exp5_sdp_cache_planar_clawfree_193_L2M2.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/regular_388/exp5_sdp_cache_regular_388_L1M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/regular_388/exp5_sdp_cache_regular_388_L2M1.slurm"
submit_when_slot_free "slurm_Exp5/sdp_cache/regular_388/exp5_sdp_cache_regular_388_L2M2.slurm"
