# Medical Dataset Pipeline

This directory contains a **model-agnostic** fine-tuning and evaluation pipeline
for the MIMIC-IV lab-value prediction task, using HypLoRA or vanilla LoRA.

## Directory Structure

```
medical/
├── model_adapter.py        # All model-specific logic lives here
├── finetune_medical.py     # Training entry point (model-agnostic)
├── evaluate_medical.py     # Evaluation entry point (model-agnostic)
├── scripts/
│   ├── hyplora/
│   │   └── qwen3-8b/
│   │       ├── finetune.sh
│   │       └── eval.sh
│   └── lora/
│       └── qwen3-8b/
│           ├── finetune.sh
│           └── eval.sh
└── README.md
```

## Data Format

Each line of the JSONL files must follow the OpenAI messages format:

```json
{
  "messages": [
    {"role": "system",    "content": "<patient info + discharge summary>"},
    {"role": "user",      "content": "<time-ordered clinical events ending with [PREDICTION_TARGET]>"},
    {"role": "assistant", "content": "{\"lab\": [v1, v2, ...]}"}
  ]
}
```

## Setup

Follow the main repository's `setup.sh` to install dependencies.
The scripts use the local modified PEFT (with HypLoRA extensions) automatically.

```bash
bash ../setup.sh
export WANDB_DISABLED=true
```

## Running Experiments

All scripts are run from the `medical/` directory.
Dataset paths are passed via environment variables.

### HypLoRA

```bash
cd medical/

# Training
TRAIN_DATA_PATH=/path/to/train.jsonl \
VAL_DATA_PATH=/path/to/val.jsonl \
bash scripts/hyplora/qwen3-8b/finetune.sh

# Evaluation
TEST_DATA_PATH=/path/to/test.jsonl \
LORA_WEIGHTS=./trained_models/qwen3-8b-hyplora-medical \
bash scripts/hyplora/qwen3-8b/eval.sh
```

### LoRA (baseline)

```bash
cd medical/

# Training
TRAIN_DATA_PATH=/path/to/train.jsonl \
VAL_DATA_PATH=/path/to/val.jsonl \
bash scripts/lora/qwen3-8b/finetune.sh

# Evaluation
TEST_DATA_PATH=/path/to/test.jsonl \
LORA_WEIGHTS=./trained_models/qwen3-8b-lora-medical \
bash scripts/lora/qwen3-8b/eval.sh
```

## Evaluation Metrics

Metrics follow [LabTOP](https://arxiv.org/abs/2502.14259), the SOTA benchmark
on the same MIMIC-IV lab-value prediction task:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (raw scale) |
| **RMSE** | Root Mean Squared Error (raw scale) |
| **NMAE** | Normalized MAE — MAE divided by the per-item value range (99th − 1st percentile). Comparable across lab items with different units. |
| **SMAPE** | Symmetric Mean Absolute Percentage Error — symmetric scaling prevents over-penalisation when ground-truth values are small. |

All metrics are reported both overall (micro-averaged) and per lab item.
Results are saved to a JSON file with the following structure:

```json
{
  "model": "Qwen/Qwen3-8B",
  "lora_type": "hyplora-0.5",
  "overall": {
    "MAE": 1.23, "RMSE": 2.45, "NMAE": 0.087, "SMAPE": 12.3,
    "parse_success_rate": 0.98
  },
  "per_item": {
    "item_0": {"MAE": ..., "RMSE": ..., "NMAE": ..., "SMAPE": ..., "count": 1200},
    ...
  },
  "samples": [...]
}
```

## Adding a New Model

All model-specific logic is isolated in `model_adapter.py`.
To support a new model:

1. Add a subclass of `BaseModelAdapter` implementing `build_prompt()` and
   `extract_response()` (only if the chat template differs from existing ones).

2. Add a pattern → adapter mapping to `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = [
    ("qwen3",   QwenChatAdapter(disable_thinking=True)),
    # Add your model here:
    ("phi-4",   Phi4Adapter()),
    ...
]
```

3. Create new shell scripts under `scripts/<method>/<model-name>/`.

No changes to `finetune_medical.py` or `evaluate_medical.py` are needed.
