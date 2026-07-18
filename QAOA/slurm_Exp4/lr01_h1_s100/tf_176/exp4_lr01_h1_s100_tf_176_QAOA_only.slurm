#!/bin/bash
#SBATCH --job-name=Exp4_lr01_h1_s100_tf_176_QAOA_only
#SBATCH --output=Results/slurm/Exp4/stdout-%x-%j.out
#SBATCH --error=Results/slurm/Exp4/stderr-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"

mkdir -p Results/slurm/Exp4 Results/logs

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate masterthesis

export MOSEKLM_LICENSE_FILE="$HOME/MasterThesis/mosek.lic"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/Exp4/lr01_h1_s100/tf_176/benchm_config_tf_176_lr01_h1_s100_QAOA_only.json
