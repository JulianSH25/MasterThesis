#!/bin/bash
#SBATCH --job-name=cache_size_test
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --time=1-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/Exp3/SDP/benchm_config_L2M2_1094.json
