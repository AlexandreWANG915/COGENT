#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLM_ICD_ROOT="${PLM_ICD_ROOT:-$(cd "$SCRIPT_DIR/../PLM-ICD" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
ADAPTER_PATH="${ADAPTER_PATH:-$SCRIPT_DIR/saves/qwen3-4b-thinking/lora/icd_multitask_verifier}"
FEATHER_FILE="${FEATHER_FILE:-}"
INPUT_FILE="${INPUT_FILE:-$PLM_ICD_ROOT/data_v4/candidates/section_candidates_val_t0.1_v4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PLM_ICD_ROOT/results}"
RESULTS_CSV="${RESULTS_CSV:-$OUTPUT_DIR/task2_results.csv}"
GPU_DEVICES="${GPU_DEVICES:-4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"

if [ -z "$FEATHER_FILE" ]; then
    echo "Please set FEATHER_FILE to the structured-note feather file." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

THRESHOLDS=(0.0)
CODE_PROBS=(0.3)
CODE_PROB_UPPER=1

for THRESHOLD in "${THRESHOLDS[@]}"; do
    for CODE_PROB in "${CODE_PROBS[@]}"; do
        echo "Running with THRESHOLD=$THRESHOLD, CODE_PROB=$CODE_PROB"

        CMD=(
            "$PYTHON_BIN" "$SCRIPT_DIR/run_verifier_inference.py"
            --model_path "$MODEL_PATH"
            --input "$INPUT_FILE"
            --feather_file "$FEATHER_FILE"
            --output "$OUTPUT_DIR/verifier_multi_task_val_codeprob${CODE_PROB}_att${THRESHOLD}.jsonl"
            --template qwen3
            --num_gpus "$NUM_GPUS"
            --code_prob_threshold "$CODE_PROB"
            --code_prob_upper "$CODE_PROB_UPPER"
            --attention_threshold "$THRESHOLD"
            --mode multi
            --results_csv "$RESULTS_CSV"
        )

        if [ -n "$ADAPTER_PATH" ]; then
            CMD+=(--adapter_path "$ADAPTER_PATH")
        fi

        CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "${CMD[@]}"

        echo "Completed THRESHOLD=$THRESHOLD, CODE_PROB=$CODE_PROB"
        echo "----------------------------------------"
    done
done

echo "All experiments completed!"
