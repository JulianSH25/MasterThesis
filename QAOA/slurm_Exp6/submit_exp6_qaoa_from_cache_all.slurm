#!/bin/bash
#SBATCH --job-name=submit_Exp6_qaoa_from_cache
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
bash slurm_Exp6/submit_exp6_qaoa_from_cache_all.sh
