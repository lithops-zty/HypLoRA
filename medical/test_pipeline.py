"""
Lightweight functional tests for the medical pipeline.
Does NOT load any LLM weights — only tests logic.
"""
import sys, os, json, math
import numpy as np

# Ensure local peft is on path (needed by model_adapter imports)
_MEDICAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_MEDICAL_DIR)
# The local modified PEFT package lives directly at <repo>/peft/
sys.path.insert(0, _REPO_ROOT)

from model_adapter import get_adapter, MODEL_REGISTRY
from evaluate_medical import parse_lab_json, compute_metrics

# ---- 1. Registry matching ----
print("=== 1. Registry matching ===")
cases = [
    ("Qwen/Qwen3-8B",                         "QwenChatAdapter"),
    ("Qwen/Qwen2.5-7B-Instruct-1M",           "QwenChatAdapter"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",   "LlamaInstructAdapter"),
    ("mistralai/Mistral-7B-Instruct-v0.3",    "MistralInstructAdapter"),
    ("google/gemma-7b-it",                     "GemmaChatAdapter"),
]
for name, expected_cls in cases:
    a = get_adapter(name)
    assert a.__class__.__name__ == expected_cls, \
        f"FAIL: {name} -> {a.__class__.__name__} (expected {expected_cls})"
    print(f"  {name:<50s} -> {a.__class__.__name__}  OK")

# ---- 2. build_prompt: Qwen3 ----
print("\n=== 2. build_prompt (Qwen3) ===")
qwen3   = get_adapter("Qwen/Qwen3-8B")
sys_msg = "Patient Information\n---\nAge: 61"
usr_msg = "Event Group: 2121-03-13\n...[PREDICTION_TARGET]"
asst    = '{"lab": [19.4, 20.5]}'

full   = qwen3.build_prompt(sys_msg, usr_msg, include_response=True,  response=asst)
prefix = qwen3.build_prompt(sys_msg, usr_msg, include_response=False)

assert "<|im_start|>assistant\n" in full,        "missing assistant marker"
assert full.endswith("<|im_end|>\n"),             "full prompt should end with <|im_end|>"
assert prefix.endswith("<|im_start|>assistant\n"),"prefix should end just before response"
assert "/no_think" in full,                       "Qwen3 thinking mode not disabled"
assert full.startswith(prefix),                   "prefix must be strict prefix of full"
print(f"  full length   : {len(full)}")
print(f"  prefix length : {len(prefix)}")
print("  build_prompt OK")

# ---- 3. extract_response ----
print("\n=== 3. extract_response ===")
fake_raw = (
    "<|im_start|>system\nPatient...<|im_end|>\n"
    "<|im_start|>user\nEvents...<|im_end|>\n"
    '<|im_start|>assistant\n{"lab": [19.4, 20.5]}<|im_end|>\n'
)
extracted = qwen3.extract_response(fake_raw)
assert extracted == '{"lab": [19.4, 20.5]}', f"FAIL: {repr(extracted)}"
print(f"  Qwen3 extract_response OK: {extracted}")

# Qwen3 with residual <think> block (safety net)
fake_thinking = (
    "<|im_start|>assistant\n"
    "<think>Let me calculate...</think>\n"
    '{"lab": [1.0, 2.0]}<|im_end|>\n'
)
extracted2 = qwen3.extract_response(fake_thinking)
assert extracted2 == '{"lab": [1.0, 2.0]}', f"FAIL thinking strip: {repr(extracted2)}"
print(f"  Qwen3 thinking-strip OK: {extracted2}")

# ---- 4. label-mask boundary (all adapters) ----
print("\n=== 4. Label-mask boundary (prefix is strict prefix of full) ===")
for model_name, _ in cases:
    a  = get_adapter(model_name)
    fp = a.build_prompt("sys", "usr", include_response=True,  response="ans")
    pp = a.build_prompt("sys", "usr", include_response=False)
    assert fp.startswith(pp), f"FAIL: {model_name}"
    print(f"  {a.__class__.__name__:<30s} OK")

# ---- 5. parse_lab_json ----
print("\n=== 5. parse_lab_json ===")
assert parse_lab_json('{"lab": [19.4, 20.5, 6]}')        == [19.4, 20.5, 6.0]
assert parse_lab_json('text {"lab": [1.0, 2.0]} more')   == [1.0, 2.0]
assert parse_lab_json("[3.14, 2.71]")                     == [3.14, 2.71]
assert parse_lab_json("no numbers here")                  is None
print("  parse_lab_json OK")

# ---- 6. compute_metrics ----
print("\n=== 6. compute_metrics ===")
preds  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
labels = [1.1, 1.9, 3.2, 3.8, 5.1, 5.9]
names  = ["item_0", "item_0", "item_1", "item_1", "item_2", "item_2"]
m = compute_metrics(preds, labels, names)

assert "overall"  in m
assert "per_item" in m
expected_mae = float(np.mean(np.abs(np.array(preds) - np.array(labels))))
assert abs(m["overall"]["MAE"] - expected_mae) < 1e-9, \
    f"MAE mismatch: {m['overall']['MAE']} vs {expected_mae}"
assert not math.isnan(m["overall"]["NMAE"]),  "NMAE is NaN"
assert not math.isnan(m["overall"]["SMAPE"]), "SMAPE is NaN"
print(f"  MAE={m['overall']['MAE']:.4f}  RMSE={m['overall']['RMSE']:.4f}"
      f"  NMAE={m['overall']['NMAE']:.4f}  SMAPE={m['overall']['SMAPE']:.4f}")
print("  compute_metrics OK")

print("\n" + "="*50)
print("All tests passed.")
print("="*50)
