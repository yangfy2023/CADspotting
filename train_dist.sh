#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cadspotting}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CONFIG_NAME="${CONFIG_NAME:-aaai_nocolor_1024_nopooling}"
CONFIG="${CONFIG:-configs/${CONFIG_NAME}.py}"
GPUS="${GPUS:-1}"
WORK_DIR="${WORK_DIR:-./work_dirs/${CONFIG_NAME}_gpus${GPUS}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29500}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"

if [ -d "$CUDA_HOME" ]; then
  export PATH="$CUDA_HOME/bin${PATH:+:${PATH}}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="$PROJECT_ROOT"

conda run -n "$CONDA_ENV_NAME" env PYTHONPATH="$PYTHONPATH" \
  python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  tools/train.py \
  "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --launcher pytorch \
  "$@"
