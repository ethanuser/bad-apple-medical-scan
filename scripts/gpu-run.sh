#!/usr/bin/env bash
# Match the full song on a CUDA machine you can run commands on directly, such as a
# lab workstation. For the Campus Cluster, use scripts/cluster-job.sbatch instead.
# Run from the repository root after scripts/gpu-sync.sh push.
#
#   scripts/gpu-run.sh            # 10 fps, 2,320 frames
#   scripts/gpu-run.sh 30         # 30 fps, 6,960 frames
#   scripts/gpu-run.sh 10 --coarse-width 48 --slice-step 2
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/setup-env.sh
exec scripts/match-full-song.sh "$@"
