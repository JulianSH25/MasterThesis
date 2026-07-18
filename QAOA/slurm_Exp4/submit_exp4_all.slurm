#!/bin/bash
#SBATCH --job-name=Exp4_submit_all
#SBATCH --output=Results/slurm/Exp4/stdout-%x-%j.out
#SBATCH --error=Results/slurm/Exp4/stderr-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=research
#SBATCH --account=research

set -euo pipefail

cd "$HOME/MasterThesis/QAOA"
mkdir -p Results/slurm/Exp4
bash slurm_Exp4/submit_exp4_all.sh
