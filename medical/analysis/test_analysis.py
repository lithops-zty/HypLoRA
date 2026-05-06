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
    compute_group_stats,
    MEDICAL_GROUPS,
    print_table1,
    print_table2,
    plot_frequency_vs_norm,
    plot_delta_distribution,
    save_csv,
    save_results_json,
    build_embedding_cache,
    embeddings_from_cache,
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
        vocab = list(MEDICAL_GROUPS["Group 1 (function words)"]) + \
                list(MEDICAL_GROUPS["Group 2 (clinical common)"]) + \
                list(MEDICAL_GROUPS["Group 3 (general medical)"]) + \
                list(MEDICAL_GROUPS["Group 4 (specific clinical)"]) + \
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
group_stats = compute_group_stats(MEDICAL_GROUPS, token_frequency, token_norms)
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

# ── 7. API mode mock test ────────────────────────────────────────────────────
print("\n--- Testing API mode (mock LMCompletion.embed) ---")

HIDDEN_API = 1536  # typical OpenAI embedding dim

class MockLMCompletion:
    """Minimal mock that satisfies LMCompletion.embed() interface."""
    def __init__(self):
        self.usage = {"prompt_tokens": 0, "total_tokens": 0}

    def embed(self, texts, model=None, **kwargs):
        # Return a random unit vector for each text
        vecs = []
        for _ in texts:
            v = np.random.randn(HIDDEN_API).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            vecs.append(v.tolist())
        self.usage["prompt_tokens"] += len(texts)
        self.usage["total_tokens"]  += len(texts)
        return vecs

mock_lm = MockLMCompletion()

# build_embedding_cache should collect unique tokens and call embed()
api_cache = build_embedding_cache(
    samples, tokenizer, "all", mock_lm, "mock-embed-model", embed_batch_size=100
)
assert len(api_cache) > 0, "API cache is empty"
first_vec = next(iter(api_cache.values()))
assert len(first_vec) == HIDDEN_API, f"Expected dim {HIDDEN_API}, got {len(first_vec)}"
print(f"[OK] build_embedding_cache: {len(api_cache)} unique tokens, dim={HIDDEN_API}")

# embeddings_from_cache should reconstruct a tensor from the cache
api_token_frequency = defaultdict(int)
api_token_norms     = defaultdict(list)
api_delta_ratios    = []

for sample in samples:
    text = extract_text(sample, "all")
    if not text.strip():
        continue
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    emb = embeddings_from_cache(tokenizer, ids, api_cache)
    if emb is None:
        continue
    assert emb.dim() == 3, "Expected (1, seq_len, hidden) tensor"
    assert emb.shape[2] == HIDDEN_API
    norms = compute_norms(emb)
    accumulate_token_stats(tokenizer, ids, norms, api_token_frequency, api_token_norms)
    delta, diam = get_delta(emb, max_points=200)
    if diam > 0:
        api_delta_ratios.append(2.0 * delta / diam)

assert len(api_delta_ratios) > 0, "No delta values computed in API mode"
assert all(0.0 <= d <= 1.0 for d in api_delta_ratios)
print(f"[OK] embeddings_from_cache: {len(api_delta_ratios)} delta values, "
      f"mean={np.mean(api_delta_ratios):.4f}")

print("\n=== All tests passed ===")
