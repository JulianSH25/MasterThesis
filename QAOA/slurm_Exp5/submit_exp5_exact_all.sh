#!/bin/bash
set -euo pipefail

sbatch slurm_Exp5/exact/complete_2to12/exp5_exact_complete_2to12.slurm
sbatch slurm_Exp5/exact/cycle_3to12/exp5_exact_cycle_3to12.slurm
sbatch slurm_Exp5/exact/path_2to12/exp5_exact_path_2to12.slurm
