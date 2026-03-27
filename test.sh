#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-new_oneformer3d}"
CONFIG="${CONFIG:-configs/test_cfg/aaai_nocolor_1024_test_mixedpooling.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/mixedpooling_best.pth}"
WORK_DIR="${WORK_DIR:-./work_dirs/smoke_test_public}"

export PYTHONPATH="$PROJECT_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  echo "Download the public checkpoint first with:"
  echo "  bash download_public_ckpt.sh"
  exit 1
fi

conda run -n "$CONDA_ENV_NAME" env PYTHONPATH="$PYTHONPATH" \
  python tools/test.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --work-dir "$WORK_DIR" \
  "$@"
