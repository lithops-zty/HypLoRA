#!/bin/bash
# =============================================================================
# HypLoRA Table 1 & 2 Reproduction — MIMIC-IV Medical Dataset
# =============================================================================
# Usage:
#   bash medical/analysis/run_analysis.sh \
#       --data_path /path/to/val.jsonl \
#       --base_model Qwen/Qwen3-8B \
#       [--output_dir medical/analysis/results] \
#       [--text_field all] \
#       [--max_samples 0] \
#       [--model_name Qwen3-8B]
#
# Arguments are passed through directly to embedding_analysis.py.
# Run from the HypLoRA repository root directory.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

echo "============================================================"
echo " HypLoRA Embedding Tree-Structure Analysis (Medical Dataset)"
echo "============================================================"
echo "Repository root: ${REPO_ROOT}"
echo ""

python3 medical/analysis/embedding_analysis.py "$@"
