#!/bin/bash
# Preprocess H5 data into tokenized .pt files
#
# Usage:
#   bash scripts/preprocess.sh
#   bash scripts/preprocess.sh --config configs/custom.yaml

source /datafs/users/wujxy/py_venv/my_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "${PROJECT_DIR}"

python3 ${PROJECT_DIR}/python/RunModule.py \
    --config "${PROJECT_DIR}/configs/default.yaml" \
    --Preprocess \
    "$@"
