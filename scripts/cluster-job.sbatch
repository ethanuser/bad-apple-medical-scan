#!/usr/bin/env bash
# Slurm job for any Slurm cluster. Run
# scripts/setup-env.sh on the login node first. Account and partition come from
# scripts/cluster-info.sh and are given on the command line:
#
#   sbatch --account=<account> --partition=<gpu partition> scripts/cluster-job.sbatch
#   sbatch --account=<account> --partition=<gpu partition> scripts/cluster-job.sbatch 30
#
# A GPU model can be requested with, for example, --gres=gpu:H100:1 if the partition
# names its models. Watch with: squeue -u $USER and tail -f output/full-song/run.log
#SBATCH --job-name=bad-apple-mri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=output/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1   # refinement parallelizes across processes instead
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
exec scripts/match-full-song.sh "$@"
