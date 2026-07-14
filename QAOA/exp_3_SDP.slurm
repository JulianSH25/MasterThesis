#!/bin/bash
#SBATCH --job-name=Exp3_SDP_L2M2_74_instances_5rep
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/Exp3/SDP/benchm_config_74instances_3reg_L2M2.json
