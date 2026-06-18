#!/bin/bash
#SBATCH --job-name=qaoa_sdp_cache
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"

zsh qaoa_benchmarks_dsri_sdpqueue.sh Code/run_configurations/benchmark_config_sdpqueue_L2M2_cache_only.json