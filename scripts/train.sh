#!/bin/bash
# Train HT-Transformer for endpoint reconstruction
#
# Usage:
#   Single GPU:     bash scripts/train.sh
#   Multi-GPU:      bash scripts/train.sh --accelerate
#   Specify GPUs:   bash scripts/train.sh --accelerate --num_processes 2

source /datafs/users/wujxy/py_venv/my_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USE_ACCELERATE=false
NUM_PROCESSES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --accelerate)
            USE_ACCELERATE=true
            shift
            ;;
        --num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ "$USE_ACCELERATE" = true ]; then
    echo "Launching with accelerate..."
    # Use config file from project directory
    export ACCELERATE_CONFIG_FILE="${PROJECT_DIR}/.accelerate/config.yaml"
    if [ -n "$NUM_PROCESSES" ]; then
        accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" --num_processes "$NUM_PROCESSES" \
            "${PROJECT_DIR}/python/RunModule.py" \
            --config "${PROJECT_DIR}/configs/default.yaml" \
            --TrainModel \
            "$@"
    else
        accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" \
            "${PROJECT_DIR}/python/RunModule.py" \
            --config "${PROJECT_DIR}/configs/default.yaml" \
            --TrainModel \
            "$@"
    fi
else
    python3 "${PROJECT_DIR}/python/RunModule.py" \
        --config "${PROJECT_DIR}/configs/default.yaml" \
        --TrainModel \
        "$@"
fi
