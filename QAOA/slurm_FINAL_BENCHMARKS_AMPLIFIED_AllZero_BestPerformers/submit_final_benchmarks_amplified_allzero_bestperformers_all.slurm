#!/bin/bash
#SBATCH --job-name=submit_FINAL_BENCHMARKS_AMPLIFIED_AllZero_BestPerformers_all
#SBATCH --output=Results/slurm/stdout-%x-%j.out
#SBATCH --error=Results/slurm/stderr-%x-%j.err
#SBATCH --partition=research
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash slurm_FINAL_BENCHMARKS_AMPLIFIED_AllZero_BestPerformers/submit_final_benchmarks_amplified_allzero_bestperformers_all.sh
