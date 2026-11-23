#!/usr/bin/env bash
# Benchmark the H200 (or any CUDA GPU) before the full-song run: search speed at 32,
# 48 and 64-pixel thumbnails against all 6,960 frames at 30 fps, refinement speed on
# all CPUs, and estimated full-run times for several densities. Run setup-env.sh on
# the login node first, then:
#
#   sbatch --account=<account> --partition=<gpu partition> scripts/cluster-bench.sbatch
#
# Results print to output/slurm-<job>.out and are saved to output/bench.json.
#SBATCH --job-name=bad-apple-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see a CUDA GPU. Check nvidia-smi and the job request.'
print(torch.cuda.get_device_name(), torch.__version__, flush=True)
PY
.venv/bin/python scripts/bench.py --device cuda "$@"
