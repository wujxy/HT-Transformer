#!/bin/bash
# Evaluate HT-Transformer + generate plots
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python3 "${PROJECT_DIR}/python/RunModule.py" \
    --config "${PROJECT_DIR}/configs/default.yaml" \
    --Eval \
    "$@"
