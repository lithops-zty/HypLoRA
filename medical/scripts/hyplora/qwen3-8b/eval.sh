#!/bin/bash
# =============================================================================
# HypLoRA evaluation on MIMIC-IV lab-value prediction — Qwen3-8B
#
# Required environment variables:
#   TEST_DATA_PATH    path to test.jsonl
#   BASE_MODEL        model path or HF Hub ID (default: Qwen/Qwen3-8B)
#   LORA_WEIGHTS      path to fine-tuned LoRA weights directory
#
# Example:
#   TEST_DATA_PATH=/data/test.jsonl \
#   LORA_WEIGHTS=./trained_models/qwen3-8b-hyplora-medical \
#   bash scripts/hyplora/qwen3-8b/eval.sh
# =============================================================================
set -euo pipefail

TEST_DATA_PATH="${TEST_DATA_PATH:?Please set TEST_DATA_PATH}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LORA_WEIGHTS="${LORA_WEIGHTS:-./trained_models/qwen3-8b-hyplora-medical}"
OUTPUT_FILE="${OUTPUT_FILE:-./results/qwen3-8b-hyplora-results.json}"

cd "$(dirname "$0")/../../.."   # cd to medical/

python evaluate_medical.py \
    --base_model      "${BASE_MODEL}" \
    --lora_weights    "${LORA_WEIGHTS}" \
    --lora_type       "hyplora-0.5" \
    --test_data_path  "${TEST_DATA_PATH}" \
    --output_file     "${OUTPUT_FILE}" \
    --rank            32 \
    --lora_alpha      128 \
    --batch_size      1 \
    --max_new_tokens  256
