#!/usr/bin/env bash
# Check the GPU, then match 0-232 s into output/full-song. Expects scripts/setup-env.sh
# to have run. Rerunning after an interruption reuses the saved retrieval
# (output/full-song/retrieval.json.gz) and repeats only the CPU refinement.
#
#   scripts/match-full-song.sh [fps] [extra match options]
set -euo pipefail
cd "$(dirname "$0")/.."
fps=${1:-10}
shift || true
seconds=232   # The pinned source is 232.13 s long.

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see a CUDA GPU. Check nvidia-smi and the job request.'
free, total = torch.cuda.mem_get_info()
print(f'{torch.cuda.get_device_name()}: {free/1e9:.0f} of {total/1e9:.0f} GB free, torch {torch.__version__}', flush=True)
PY

mkdir -p output/full-song
.venv/bin/python -m mriapple match --device cuda --start 0 --seconds "$seconds" --fps "$fps" \
  --output output/full-song "$@" 2>&1 | tee -a output/full-song/run.log
echo "Done. Locally: scripts/gpu-sync.sh pull <host> <dir>, then render."
