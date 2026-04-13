#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:?Usage: $0 <gpu> <use_laat:0|1> <from_scratch:0|1> <output_dir> [python_bin]}"
USE_LAAT="${2:?Usage: $0 <gpu> <use_laat:0|1> <from_scratch:0|1> <output_dir> [python_bin]}"
FROM_SCRATCH="${3:?Usage: $0 <gpu> <use_laat:0|1> <from_scratch:0|1> <output_dir> [python_bin]}"
OUTPUT_DIR="${4:?Usage: $0 <gpu> <use_laat:0|1> <from_scratch:0|1> <output_dir> [python_bin]}"
PYTHON_BIN="${5:-python}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$PROJECT_ROOT/models/clinical-longformer-base}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/train.csv}"
VALIDATION_FILE="${VALIDATION_FILE:-$PROJECT_ROOT/data/val.csv}"
CODE_FILE="${CODE_FILE:-$PROJECT_ROOT/data/ALL_CODES.txt}"

ARGS="--model_name_or_path $MODEL_NAME_OR_PATH     --train_file $TRAIN_FILE     --validation_file $VALIDATION_FILE     --code_file $CODE_FILE     --max_length 4096     --per_device_train_batch_size 8     --gradient_accumulation_steps 2     --learning_rate 2e-5     --num_train_epochs 3     --output_dir $OUTPUT_DIR"

if [ "$USE_LAAT" = "1" ]; then
    ARGS="$ARGS --use_laat"
fi

if [ "$FROM_SCRATCH" = "1" ]; then
    ARGS="$ARGS --from_scratch"
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "$PROJECT_ROOT/src/train_longformer_icd.py" $ARGS
