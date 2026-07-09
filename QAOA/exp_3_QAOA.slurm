#!/bin/bash
#SBATCH --job-name=Exp3_QAOA_L2M2_78_instances
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/Exp3/QAOA/benchm_config_78instances.json
