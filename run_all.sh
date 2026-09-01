#!/usr/bin/env bash
# One B200 session, everything captured to one log.
set -euo pipefail

ARCH="${1:-sm_100}"          # sm_100 B200/B300, sm_90 H100, sm_89 4090

{
  echo "=== provenance ==="
  git rev-parse HEAD
  nvcc --version
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
  date -u

  echo
  echo "=== build (arch $ARCH) ==="
  for f in bench order layout rl contend timing; do
    echo "--- splitk_$f"
    nvcc -O3 -arch="$ARCH" "splitk_$f.cu" -o "splitk_$f"
  done
  echo "all builds ok"

  for f in order layout bench rl contend timing; do
    echo
    echo "################ splitk_$f ################"
    "./splitk_$f"
  done

  echo
  echo "=== done ==="
  date -u
} 2>&1 | tee b200-final-results.txt
