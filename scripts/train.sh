#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTHON="${MAGIC_W0_PYTHON:-python}"
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    exec "$PYTHON" "$ROOT/scripts/train.py" "$@"
  fi
done
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NNODES" == "1" ]]; then
  exec "$PYTHON" -m torch.distributed.run --standalone \
    --nproc_per_node="$NPROC_PER_NODE" "$ROOT/scripts/train.py" "$@"
fi
: "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
: "${NODE_RANK:?Set NODE_RANK for multi-node training}"
exec "$PYTHON" -m torch.distributed.run --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
  "$ROOT/scripts/train.py" "$@"
