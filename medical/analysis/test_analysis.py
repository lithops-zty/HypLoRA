"""
Lightweight end-to-end test for embedding_analysis.py
Mocks the model with random embeddings so no GPU / large model download is needed.
"""
import json, os, sys, math, tempfile
import numpy as np
import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from medical.analysis.embedding_analysis import (
    load_jsonl,
    extract_text,
    clean_token,
    compute_norms,
    accumulate_token_stats,
    compute_group_stats,
    delta_hyp,
    get_delta,
    MEDICAL_GROUPS_FALLBACK,
    print_table1,
    print_table2,
    plot_frequency_vs_norm,
    plot_delta_distribution,
    save_csv,
    save_results_json,
)
from collections import defaultdict

# ── 1. Load real val.jsonl ──────────────────────────────────────────────────
DATA_PATH = "/home/ubuntu/upload/val.jsonl"
samples = load_jsonl(DATA_PATH)
assert len(samples) == 39, f"Expected 39 samples, got {len(samples)}"
print(f"[OK] Loaded {len(samples)} samples from {DATA_PATH}")

# ── 2. extract_text ─────────────────────────────────────────────────────────
for field in ("user", "system", "all"):
    t = extract_text(samples[0], field)
    assert len(t) > 0, f"extract_text('{field}') returned empty string"
print("[OK] extract_text for user / system / all")

# ── 3. Mock tokenizer & embeddings ──────────────────────────────────────────
class MockTokenizer:
    def __call__(self, text, return_tensors=None, truncation=False):
        # Simulate ~50 tokens per sample
        n = min(50, len(text.split()))
        ids = torch.randint(0, 32000, (1, n))
        return {"input_ids": ids}
    def convert_ids_to_tokens(self, ids):
        vocab = list(MEDICAL_GROUPS_FALLBACK["Group 1 (function words)"]) + \
                list(MEDICAL_GROUPS_FALLBACK["Group 2 (clinical common)"]) + \
                list(MEDICAL_GROUPS_FALLBACK["Group 3 (general medical)"]) + \
                list(MEDICAL_GROUPS_FALLBACK["Group 4 (specific clinical)"]) + \
                ["xyz", "abc", "123"]
        return [vocab[i % len(vocab)] for i in ids]

tokenizer = MockTokenizer()
HIDDEN = 64

token_frequency = defaultdict(int)
token_norms     = defaultdict(list)
delta_ratios    = []

for sample in samples:
    text = extract_text(sample, "all")
    inp  = tokenizer(text)
    ids  = inp["input_ids"]
    n    = ids.shape[1]
    emb  = torch.randn(1, n, HIDDEN)
    norms = compute_norms(emb)
    accumulate_token_stats(tokenizer, ids, norms, token_frequency, token_norms)
    delta, diam = get_delta(emb, max_points=200)
    if diam > 0:
        delta_ratios.append(2.0 * delta / diam)

print(f"[OK] Accumulated stats for {len(samples)} samples")
print(f"     Unique tokens: {len(token_frequency)}")
print(f"     Delta samples: {len(delta_ratios)}")

# ── 4. Group stats ───────────────────────────────────────────────────────────
group_stats = compute_group_stats(MEDICAL_GROUPS_FALLBACK, token_frequency, token_norms)
found_any = any(len(s["tokens_found"]) > 0 for s in group_stats.values())
assert found_any, "No tokens found in any group — mock tokenizer may be broken"
print_table1(group_stats, "MockModel")
print("[OK] Table 1 group stats computed")

# ── 5. Delta hyperbolicity ───────────────────────────────────────────────────
assert len(delta_ratios) > 0
assert all(0.0 <= d <= 1.0 for d in delta_ratios), "delta_rel out of [0,1]"
print_table2(delta_ratios, "MockModel", "val")
print("[OK] Table 2 delta-hyperbolicity computed")

# ── 6. Outputs ───────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    plot_frequency_vs_norm(token_frequency, token_norms,
                           os.path.join(tmpdir, "freq_vs_norm.png"), "val")
    plot_delta_distribution(delta_ratios,
                            os.path.join(tmpdir, "delta_hist.png"), "val")
    save_csv(token_frequency, token_norms, os.path.join(tmpdir, "stats.csv"))
    save_results_json(group_stats, delta_ratios, "MockModel", "val",
                      os.path.join(tmpdir, "results.json"))
    # Verify JSON is valid
    with open(os.path.join(tmpdir, "results.json")) as f:
        r = json.load(f)
    assert "table1_group_stats" in r and "table2_delta_hyperbolicity" in r
    print(f"[OK] All output files generated successfully")

# ── 7. LLM grouping mock test ────────────────────────────────────────────────
print("\n--- Testing LLM-based Table 1 grouping (mock LMCompletion) ---")

from medical.analysis.embedding_analysis import select_groups_via_llm

# Patch LMCompletion inside embedding_analysis to intercept the API call
import medical.analysis.embedding_analysis as _ea
import json as _json
import unittest.mock as _mock

def _mock_lm_init(self, model=None, api_key=None, request_url=None,
                  log_mode="none", track_usage=False):
    self.model = model
    self.usage = {"prompt_tokens": 0, "total_tokens": 0}
    self._history = []

def _mock_lm_call(self, prompt, role="user", temperature=1.0, **kwargs):
    if role == "system":
        self._history.append({"role": "system", "content": prompt})
        return ""
    # Build a minimal 4-group response using the first tokens seen
    all_tokens = list(token_frequency.keys())
    n = len(all_tokens)
    chunk = max(1, n // 4)
    groups = {
        "group1": all_tokens[:chunk],
        "group2": all_tokens[chunk:2*chunk],
        "group3": all_tokens[2*chunk:3*chunk],
        "group4": all_tokens[3*chunk:],
    }
    self.usage["total_tokens"] += 100
    return _json.dumps(groups)

MockLMClass = type("MockLMCompletion", (), {
    "__init__": _mock_lm_init,
    "__call__": _mock_lm_call,
})

with _mock.patch.object(_ea, "LMCompletion", MockLMClass):
    llm_groups = select_groups_via_llm(
        token_frequency, "mock-llm-model", vocab_sample_k=50
    )
assert isinstance(llm_groups, dict), "select_groups_via_llm should return a dict"
assert len(llm_groups) == 4, f"Expected 4 groups, got {len(llm_groups)}"
llm_stats = compute_group_stats(llm_groups, token_frequency, token_norms)
assert len(llm_stats) == 4
print(f"[OK] select_groups_via_llm: {len(llm_groups)} groups, "
      f"tokens covered: {sum(len(v) for v in llm_groups.values())}")

print("\n=== All tests passed ===")
