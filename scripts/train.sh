#!/bin/bash
# Train HT-Transformer for endpoint reconstruction

source /datafs/users/wujxy/py_venv/my_env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USE_ACCELERATE=false
NUM_PROCESSES=""

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

cd "${PROJECT_DIR}"

if [ "$USE_ACCELERATE" = true ]; then
    echo "Launching with accelerate..."
    export ACCELERATE_CONFIG_FILE="${PROJECT_DIR}/.accelerate/config.yaml"
    if [ -n "$NUM_PROCESSES" ]; then
        accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" --num_processes "$NUM_PROCESSES" \
            -m cli.run \
            --config "${PROJECT_DIR}/configs/default.yaml" \
            --TrainModel \
            "$@"
    else
        accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" \
            -m cli.run \
            --config "${PROJECT_DIR}/configs/default.yaml" \
            --TrainModel \
            "$@"
    fi
else
    python3 -m cli.run \
        --config "${PROJECT_DIR}/configs/default.yaml" \
        --TrainModel \
        "$@"
fi
