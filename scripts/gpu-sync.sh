#!/usr/bin/env bash
# Copy the code to a GPU machine, or copy its full-song results back. Run on the local machine
# from the repository root. <host> is anything ssh accepts, such as user@1.2.3.4.
#
#   scripts/gpu-sync.sh push <host> [remote dir]
#   scripts/gpu-sync.sh push-ct <host> [remote dir]     # code plus the 2.4 GB of CT
#   scripts/gpu-sync.sh push-data <host> [remote dir]   # code plus all 4.6 GB of inputs
#   scripts/gpu-sync.sh pull <host> [remote dir]
#
# push copies only code; setup-env.sh then fetches the inputs from their pinned URLs.
# push-ct also copies the CT scans, which Zenodo serves slowly (about 45 minutes);
# the brain MRI then downloads quickly from Hugging Face on the GPU machine.
# push-data also copies the verified inputs from this machine, which avoids Zenodo's slow
# per-file CT download; setup-env.sh still checks every hash. Prepared-CT caches are
# not copied, since the GPU machine rebuilds them in a few minutes. After pull, render locally with:
#   .venv/bin/python -m mriapple render --output output/full-song
set -euo pipefail
cmd=${1:?push or pull}
host=${2:?ssh host}
dir=${3:-bad-apple-mri}

# Reuse one authenticated connection for 10 minutes, so a two-factor prompt appears once per sync.
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
rsh="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=10m"

case "$cmd" in
  push)
    rsync -az -e "$rsh" --exclude .git --exclude .venv --exclude assets --exclude output \
      --exclude __pycache__ ./ "$host:$dir/"
    echo "Copied to $host:$dir. Next, there: scripts/setup-env.sh, then sbatch (cluster) or scripts/gpu-run.sh."
    ;;
  push-ct)
    rsync -az -e "$rsh" --exclude .git --exclude .venv --exclude assets --exclude output \
      --exclude __pycache__ ./ "$host:$dir/"
    rsync -az -e "$rsh" --exclude '*.npz' --exclude '*.part' assets/ts_* "$host:$dir/assets/"
    echo "Copied code and CT to $host:$dir. Next, there: scripts/setup-env.sh"
    ;;
  push-data)
    rsync -az -e "$rsh" --exclude .git --exclude .venv --exclude output \
      --exclude __pycache__ --exclude '*.npz' --exclude '*.part' ./ "$host:$dir/"
    echo "Copied code and inputs to $host:$dir. Next, there: scripts/setup-env.sh"
    ;;
  pull)
    mkdir -p output/full-song
    rsync -az -e "$rsh" "$host:$dir/output/full-song/" output/full-song/
    echo "Copied results to output/full-song"
    ;;
  *)
    echo "Use push, push-ct, push-data or pull" >&2
    exit 2
    ;;
esac
