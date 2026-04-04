#!/bin/bash
# Inspect H5 schema
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python "${PROJECT_DIR}/python/RunModule.py" \
    --config "${PROJECT_DIR}/configs/default.yaml" \
    --InspectH5 \
    "$@"
