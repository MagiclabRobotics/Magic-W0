#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---inference}"
case "$MODE" in
  --inference)
    TORCH_INDEX_URL="${MAGIC_W0_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX_URL"
    python -m pip install -e "${REPO_DIR}[eval]"
    ;;
  --train)
    TORCH_INDEX_URL="${MAGIC_W0_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX_URL"
    python -m pip install -c "$REPO_DIR/constraints/train.txt" -e "${REPO_DIR}[train,teachers,eval]"
    ;;
  *) echo 'Usage: bash scripts/install.sh [--inference|--train]' >&2; exit 2 ;;
esac
python -m pip check
