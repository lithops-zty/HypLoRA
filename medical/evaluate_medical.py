"""
evaluate_medical.py
===================
Model-agnostic evaluation script for the MIMIC-IV lab-value prediction task.

Metrics (following LabTOP, the SOTA benchmark on the same task):
  - MAE   : Mean Absolute Error (raw scale, for reference)
  - RMSE  : Root Mean Squared Error (raw scale, for reference)
  - NMAE  : Normalized MAE — MAE divided by the per-item value range
             (99th pct − 1st pct), making results comparable across lab items
             with different units and scales.
  - SMAPE : Symmetric Mean Absolute Percentage Error — symmetric scaling
             prevents over-penalisation when ground-truth values are small.

All metrics are reported both overall (micro-averaged across all predictions)
and per lab-item.

Usage
-----
    python evaluate_medical.py \\
        --base_model   "Qwen/Qwen3-8B" \\
        --lora_weights "./trained_models/qwen3-8b-hyplora-medical" \\
        --lora_type    "hyplora-0.5" \\
        --test_data_path /path/to/test.jsonl \\
        --output_file  "./results/hyplora_results.json" \\
        --rank 32 --lora_alpha 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig

# ---------------------------------------------------------------------------
# Path setup (same as finetune_medical.py)
# ---------------------------------------------------------------------------
_MEDICAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_MEDICAL_DIR)
# The local modified PEFT package lives directly at <repo>/peft/
_PEFT_SRC = _REPO_ROOT
if _PEFT_SRC not in sys.path:
    sys.path.insert(0, _PEFT_SRC)

from model_adapter import get_adapter  # noqa


# ===========================================================================
# Data loading
# ===========================================================================

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_messages(record: dict) -> tuple[str, str, str]:
    msgs     = record["messages"]
    role_map = {m["role"]: m["content"] for m in msgs}
    return role_map.get("system", ""), role_map.get("user", ""), role_map.get("assistant", "")


# ===========================================================================
# Answer parsing
# ===========================================================================

def parse_lab_json(text: str) -> Optional[list[float]]:
    """
    Extract the list of predicted lab values from the model's response.

    Accepts:
      - {"lab": [1.0, 2.5, ...]}
      - {"lab": [1.0, 2.5, ...]}  (with surrounding text)
      - plain JSON arrays [1.0, 2.5, ...]

    Returns None if parsing fails.
    """
    # Strategy 1: find a JSON object containing "lab"
    match = re.search(r'\{[^{}]*"lab"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            vals = obj.get("lab", [])
            return [float(v) for v in vals]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: find any JSON array of numbers
    match = re.search(r'\[[\d\s.,\-eE]+\]', text)
    if match:
        try:
            vals = json.loads(match.group())
            return [float(v) for v in vals]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def parse_lab_json_label(text: str) -> Optional[list[float]]:
    """Parse ground-truth label (same format, but must not fail)."""
    result = parse_lab_json(text)
    if result is None:
        raise ValueError(f"Cannot parse ground-truth label: {repr(text)}")
    return result


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(
    preds: list[float],
    labels: list[float],
    item_names: list[str],
) -> dict:
    """
    Compute MAE, RMSE, NMAE, SMAPE — overall and per lab item.

    Parameters
    ----------
    preds, labels : parallel lists of scalar predictions and ground-truths
    item_names    : parallel list of lab-item identifiers (e.g. "item_0")
    """
    assert len(preds) == len(labels) == len(item_names)

    # Group by item
    item_preds  = defaultdict(list)
    item_labels = defaultdict(list)
    for p, l, name in zip(preds, labels, item_names):
        item_preds[name].append(p)
        item_labels[name].append(l)

    per_item: dict[str, dict] = {}
    for name in item_preds:
        ps = np.array(item_preds[name], dtype=float)
        ls = np.array(item_labels[name], dtype=float)
        n  = len(ps)

        mae  = float(np.mean(np.abs(ps - ls)))
        rmse = float(np.sqrt(np.mean((ps - ls) ** 2)))

        # NMAE: normalise by the value range (99th pct - 1st pct) of labels
        v99, v01 = np.percentile(ls, 99), np.percentile(ls, 1)
        scale    = v99 - v01
        nmae     = float(mae / scale) if scale > 1e-9 else float("nan")

        # SMAPE
        denom = (np.abs(ps) + np.abs(ls)) / 2.0
        smape = float(np.mean(
            np.where(denom < 1e-9, 0.0, np.abs(ps - ls) / denom)
        ) * 100.0)

        per_item[name] = {
            "MAE":   mae,
            "RMSE":  rmse,
            "NMAE":  nmae,
            "SMAPE": smape,
            "count": n,
        }

    # Overall (micro-average across all predictions)
    all_p = np.array(preds,  dtype=float)
    all_l = np.array(labels, dtype=float)

    overall_mae  = float(np.mean(np.abs(all_p - all_l)))
    overall_rmse = float(np.sqrt(np.mean((all_p - all_l) ** 2)))

    # Overall NMAE: macro-average of per-item NMAE (ignoring NaN items)
    valid_nmae = [v["NMAE"] for v in per_item.values() if not math.isnan(v["NMAE"])]
    overall_nmae = float(np.mean(valid_nmae)) if valid_nmae else float("nan")

    # Overall SMAPE: micro-average
    denom = (np.abs(all_p) + np.abs(all_l)) / 2.0
    overall_smape = float(np.mean(
        np.where(denom < 1e-9, 0.0, np.abs(all_p - all_l) / denom)
    ) * 100.0)

    return {
        "overall":  {
            "MAE":   overall_mae,
            "RMSE":  overall_rmse,
            "NMAE":  overall_nmae,
            "SMAPE": overall_smape,
        },
        "per_item": per_item,
    }


# ===========================================================================
# Inference
# ===========================================================================

def run_inference(
    tokenizer,
    model,
    adapter,
    dataset: list[dict],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """
    Run batched inference and return raw decoded strings for each sample.
    """
    device = next(model.parameters()).device

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        max_new_tokens=max_new_tokens,
    )

    raw_outputs = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
        batch = dataset[start : start + batch_size]

        # Build inference prompts (no assistant response included)
        prompts = []
        for record in batch:
            system, user, _ = parse_messages(record)
            prompts.append(adapter.build_prompt(system, user, include_response=False))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
            )

        for seq in gen_out.sequences:
            raw_outputs.append(tokenizer.decode(seq, skip_special_tokens=False))

    return raw_outputs


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # ------------------------------------------------------------------
    # Model-agnostic adapter
    # ------------------------------------------------------------------
    adapter = get_adapter(args.base_model)
    print(f"Model adapter : {adapter.__class__.__name__}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    tokenizer, model = adapter.load_model(
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    dataset = load_jsonl(args.test_data_path)
    total   = len(dataset)
    print(f"Test samples  : {total}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    raw_outputs = run_inference(
        tokenizer=tokenizer,
        model=model,
        adapter=adapter,
        dataset=dataset,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Parse predictions and collect (pred, label, item_name) triples
    # ------------------------------------------------------------------
    all_preds:      list[float] = []
    all_labels:     list[float] = []
    all_item_names: list[str]   = []
    sample_records: list[dict]  = []

    parse_success = 0

    for idx, (record, raw_out) in enumerate(zip(dataset, raw_outputs)):
        _, _, assistant_label = parse_messages(record)

        # Extract clean response (model-agnostic)
        clean_response = adapter.extract_response(raw_out)

        # Parse predicted values
        pred_vals  = parse_lab_json(clean_response)
        label_vals = parse_lab_json_label(assistant_label)

        success = pred_vals is not None
        if success:
            parse_success += 1

        # Align lengths: only score positions present in both pred and label
        if success:
            n = min(len(pred_vals), len(label_vals))
            for i, (p, l) in enumerate(zip(pred_vals[:n], label_vals[:n])):
                all_preds.append(p)
                all_labels.append(l)
                all_item_names.append(f"item_{i}")

        sample_records.append({
            "idx":           idx,
            "pred_raw":      clean_response,
            "pred":          pred_vals,
            "label":         label_vals,
            "parse_success": success,
        })

    parse_rate = parse_success / total if total > 0 else 0.0
    print(f"\nParse success rate: {parse_success}/{total} ({parse_rate:.1%})")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    if all_preds:
        metrics = compute_metrics(all_preds, all_labels, all_item_names)
    else:
        metrics = {"overall": {}, "per_item": {}}
        print("WARNING: no parseable predictions — metrics cannot be computed.")

    metrics["overall"]["parse_success_rate"] = parse_rate

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics["overall"].items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: {v}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "model":      args.base_model,
        "lora_type":  args.lora_type,
        "lora_weights": args.lora_weights,
        **metrics,
        "samples":    sample_records,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output_file}")


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the MIMIC-IV lab-value prediction task."
    )
    parser.add_argument("--base_model",      required=True,  help="Base model path or HF Hub ID")
    parser.add_argument("--lora_weights",    required=True,  help="Path to fine-tuned LoRA weights")
    parser.add_argument("--lora_type",       required=True,  help="LoRA type used during training (e.g. 'std', 'hyplora-0.5')")
    parser.add_argument("--test_data_path",  required=True,  help="Path to test JSONL file")
    parser.add_argument("--output_file",     default="results/result.json", help="Path to save evaluation results JSON")
    parser.add_argument("--rank",            type=int, default=32,   help="LoRA rank (informational only)")
    parser.add_argument("--lora_alpha",      type=int, default=128,  help="LoRA alpha (informational only)")
    parser.add_argument("--batch_size",      type=int, default=1,    help="Inference batch size")
    parser.add_argument("--max_new_tokens",  type=int, default=256,  help="Max new tokens to generate")
    return parser.parse_args()


if __name__ == "__main__":
    main()
