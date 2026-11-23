#!/usr/bin/env bash
# Print what this Slurm account can use: accounts, GPU partitions, storage and
# Python versions. Read-only. Run on the cluster login node.
echo "== Slurm accounts and partitions for $USER (account|partition|qos)"
sacctmgr -nP show assoc user="$USER" format=account,partition,qos 2>/dev/null | sort -u
echo
echo "== GPU partitions (partition, time limit, nodes, GPUs per node)"
sinfo -h -o "%P %l %D %G" 2>/dev/null | grep -i gpu | sort -u
echo
echo "== Storage (run the project from a directory with about 15 GB free)"
quota -s 2>/dev/null || true
df -h "$HOME" 2>/dev/null | tail -1
for d in /scratch/users/"$USER" /scratch/"$USER" /projects; do
  [ -d "$d" ] && df -h "$d" | tail -1 | sed "s|^|$d: |"
done
echo
echo "== Python"
for p in python3.13 python3.12 python3.11 python3; do
  command -v "$p" >/dev/null && echo "$p: $("$p" --version 2>&1)"
done
(module -t avail python 2>&1 || true) | grep -i python | head -10
