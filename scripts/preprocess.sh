#!/bin/bash
source /datafs/users/wujxy/py_venv/my_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "${PROJECT_DIR}"
python3 -m cli.run \
    --config "${PROJECT_DIR}/configs/default.yaml" \
    --Preprocess \
    "$@"
