#!/usr/bin/env bash
# Create .venv, install the requirements and fetch the pinned inputs (about 4.6 GB,
# plus 2.2 GB of CT caches and about 5 GB for PyTorch with CUDA). Needs internet, so on a cluster run it on
# the login node. No sudo needed. Matching does not need FFmpeg.
#
#   scripts/setup-env.sh
#   PYTHON=/path/to/python3.12 scripts/setup-env.sh
set -euo pipefail

ok_python() {
  command -v "$1" >/dev/null &&
    "$1" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null
}

if [ ! -x .venv/bin/python ]; then
  made=
  for candidate in ${PYTHON:-} python3.13 python3.12 python3.11 python3; do
    if ok_python "$candidate" && "$candidate" -m venv .venv 2>/dev/null; then
      made=1
      break
    fi
    rm -rf .venv
  done
  if [ -z "$made" ]; then
    # No suitable interpreter: uv downloads a standalone Python 3.13 into ~/.local.
    if command -v uv >/dev/null; then
      uv venv --python 3.13 .venv
    else
      python3 -m pip install --user -q uv
      python3 -m uv venv --python 3.13 .venv
    fi
  fi
fi

if .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m pip install -q -r requirements.txt
elif command -v uv >/dev/null; then
  uv pip install --python .venv/bin/python -q -r requirements.txt
else
  python3 -m uv pip install --python .venv/bin/python -q -r requirements.txt
fi
.venv/bin/python -m mriapple fetch
mkdir -p output   # Slurm writes its log here before the job starts.
echo "Environment ready: $(.venv/bin/python --version), inputs verified."
