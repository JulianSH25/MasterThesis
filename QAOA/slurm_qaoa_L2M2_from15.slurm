#!/bin/bash
#SBATCH --job-name=qaoa_L2M2_from15
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=48G
#SBATCH --partition=research
#SBATCH --account=research
#SBATCH --nodelist=dacsvm-cpunode01.fse-cslab.nl

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/benchmark_config_L2M2_qaoa_from15.json
