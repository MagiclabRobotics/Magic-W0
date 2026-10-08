#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX_URL="${MAGIC_W0_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX_URL}"
python -m pip install -e "${REPO_DIR}[eval]"
python -m pip check
