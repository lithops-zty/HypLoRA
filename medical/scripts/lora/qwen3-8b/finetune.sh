#!/bin/bash
# =============================================================================
# LoRA (baseline) fine-tuning on MIMIC-IV lab-value prediction — Qwen3-8B
#
# Required environment variables:
#   TRAIN_DATA_PATH   path to train.jsonl
#   VAL_DATA_PATH     path to val.jsonl
#   BASE_MODEL        model path or HF Hub ID (default: Qwen/Qwen3-8B)
#
# Example:
#   TRAIN_DATA_PATH=/data/train.jsonl \
#   VAL_DATA_PATH=/data/val.jsonl \
#   bash scripts/lora/qwen3-8b/finetune.sh
# =============================================================================
set -euo pipefail

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Please set TRAIN_DATA_PATH}"
VAL_DATA_PATH="${VAL_DATA_PATH:?Please set VAL_DATA_PATH}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./trained_models/qwen3-8b-lora-medical}"

cd "$(dirname "$0")/../../.."   # cd to medical/

python finetune_medical.py \
    --base_model      "${BASE_MODEL}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --val_data_path   "${VAL_DATA_PATH}" \
    --output_dir      "${OUTPUT_DIR}" \
    --lora_type       "std" \
    --lora_r          32 \
    --lora_alpha      64 \
    --lora_dropout    0.05 \
    --batch_size      16 \
    --micro_batch_size 2 \
    --num_epochs      3 \
    --learning_rate   3e-4 \
    --cutoff_len      2048 \
    --train_on_inputs False \
    --eval_steps      500 \
    --save_steps      500
