#!/bin/bash
#SBATCH --job-name=Exp5_rerun2508_03_complete_2to12_lr01_h1_s50_QAOA_only_Exp5_subexp4_standardWithHeuristic_rerun2508_03
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --partition=research
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p Results/slurm

if [ -f "/home/i6408800/miniforge3/bin/activate" ]; then
  source "/home/i6408800/miniforge3/bin/activate" "masterthesis"
elif [ -f "$HOME/miniforge3/bin/activate" ]; then
  source "$HOME/miniforge3/bin/activate" "masterthesis"
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate" "masterthesis"
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
  source "$HOME/anaconda3/bin/activate" "masterthesis"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "masterthesis"
else
  echo "Error: could not find conda activation script for environment masterthesis." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

zsh qaoa_benchmarks_dsri_queuedJobs.sh Code/run_configurations/Exp5_rerun2508_03/lr01_h1_s50/complete_2to12/benchm_config_complete_2to12_lr01_h1_s50_QAOA_only.json
