#!/bin/bash
#SBATCH --job-name=recoil_sweep
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-79%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/recoil_sweep_%A_%a.out

set -euo pipefail

module load conda/python3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optomechanics-levitation

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"
python -u test/recoil_heating_size_sweep.py
