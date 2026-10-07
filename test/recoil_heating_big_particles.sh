#!/bin/bash
#SBATCH --job-name=recoil_big
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-39%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

module load conda/python3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optomechanics-levitation

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$SLURM_SUBMIT_DIR"
python -u test/recoil_heating_big_particles.py