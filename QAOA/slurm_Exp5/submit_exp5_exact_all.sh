#!/bin/bash
set -euo pipefail

sbatch slurm_Exp5/exact/path_2to12/exp5_exact_path_2to12.slurm
sbatch slurm_Exp5/exact/bipartite_454/exp5_exact_bipartite_454.slurm
sbatch slurm_Exp5/exact/planar_clawfree_193/exp5_exact_planar_clawfree_193.slurm
sbatch slurm_Exp5/exact/regular_388/exp5_exact_regular_388.slurm
