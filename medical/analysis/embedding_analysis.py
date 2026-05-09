"""
Medical Dataset Embedding Tree-Structure Analysis
==================================================
Reproduces the investigation from HypLoRA (NeurIPS 2025) Tables 1 & 2
on the MIMIC-IV clinical lab-value prediction dataset.

Table 1 equivalent: Token frequency vs. embedding norm analysis.
  - Four token groups are defined based on medical-domain specificity:
      Group 1 (high-freq function words): the, is, of, and, in, to, a
      Group 2 (clinical common terms):    patient, blood, mg, day, history
      Group 3 (general medical concepts): glucose, sodium, hemoglobin, creatinine, platelet
      Group 4 (specific clinical values): administered, hematocrit, fibrinogen, aripiprazole, phytonadione
  - For each group, reports: Frequency (Mean [Min~Max]) and Norm (Mean [Min~Max])

Table 2 equivalent: delta-hyperbolicity of per-sample token embeddings.
  - For each sample, tokenizes the full prompt (system + user content),
    extracts input embeddings, computes Gromov delta-hyperbolicity,
    normalises by diameter: delta_rel = 2*delta / diam
  - Reports mean ± std of delta_rel across all samples.

Usage:
    python medical/analysis/embedding_analysis.py \
        --data_path path/to/val.jsonl \
        --base_model Qwen/Qwen3-8B \
        --output_dir medical/analysis/results \
        [--text_field all]           # user | system | all
        [--max_samples 0]            # 0 = all samples
        [--max_points 1500]          # max tokens per sample for delta computation
        [--model_name Qwen3-8B]      # display name used in output filenames
        [--group_llm_model gpt-4o-mini]  # LLM for Table 1 token grouping (API)
        [--api_key sk-...]           # API key for --group_llm_model
        [--api_base https://...]     # API base URL for --group_llm_model

    Embeddings are always extracted locally via the model's input embedding
    matrix (no GPU forward pass needed beyond loading the embedding layer).
    OPENAI_API_KEY and OPENAI_BASE_URL environment variables are respected
    for the optional LLM-based token grouping call.
"""

import argparse
import json
import os
import re
import sys
import csv
import math
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import distance_matrix as scipy_distance_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from transformers import AutoModelForCausalLM, AutoTokenizer

# LMCompletion is in the parent package; add it to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from lm_completion import LMCompletion
except ImportError:
    LMCompletion = None  # gracefully degrade if openai not installed

# ---------------------------------------------------------------------------
# Medical-domain token groups (Table 1 equivalent)
# ---------------------------------------------------------------------------
# Groups are selected at runtime by an LLM from a stratified random sample of
# the actual tokenizer vocabulary observed in the dataset.  The hard-coded
# fallback below is used only when --group_llm_model is not provided.
MEDICAL_GROUPS_FALLBACK = {
    "Group 1 (function words)": [
        "the", "of", "and", "to", "for", "by", "with", "is", "are", "been",
    ],
    "Group 2 (clinical common)": [
        "blood", "mg", "insulin", "sodium", "hematocrit",
        "glucose", "chloride", "administered", "laboratory", "emar",
    ],
    "Group 3 (general medical)": [
        "phosphate", "hemoglobin", "calcium", "bicarbonate",
        "magnesium", "platelet", "bilirubin", "potassium", "albumin",
    ],
    "Group 4 (specific clinical)": [
        "phytonadione", "enoxaparin", "creatinine", "furosemide",
        "gabapentin", "prednisone", "budesonide", "tramadol", "loratadine",
    ],
}


_GROUP_SYSTEM_PROMPT = """\
You are a medical NLP expert. You will be given a list of tokens extracted from \
a BPE tokenizer applied to MIMIC-IV clinical EHR notes. The tokens have been \
cleaned (BPE prefix symbols removed, lowercased), so some may be subword \
fragments rather than complete words.

Your task is to select tokens for four groups based on clinical specificity:

- Group 1 (Function words): Common non-medical function words (articles, \
prepositions, conjunctions, auxiliaries). E.g., "the", "of", "is", "with".
- Group 2 (Clinical common): General clinical terms understandable to \
non-specialists. E.g., "blood", "mg", "glucose", "administered".
- Group 3 (General medical): Standard medical terminology requiring basic \
medical training. E.g., "hemoglobin", "platelet", "bilirubin", "creatinine".
- Group 4 (Specific clinical): Highly specialized terms — rare drug names, \
specific lab assays, or complex procedures unfamiliar to non-specialists. \
E.g., "phytonadione", "enoxaparin", "esophagogastroduodenoscopy".

Rules:
1. Prefer tokens that are semantically complete words. Avoid selecting tokens \
that are clearly subword fragments (e.g., single syllables like "ine", "tion", \
"ing" that carry no standalone meaning).
2. Select exactly 20 tokens per group.
3. Each token must come from the provided list.
4. No token may appear in more than one group.
5. Return ONLY a valid JSON object with keys "group1", "group2", "group3", \
"group4", each a list of exactly 20 strings. No explanation or markdown.
"""


def _stratified_sample(
    token_frequency: dict[str, int],
    sample_k: int,
    seed: int = 42,
) -> dict[str, int]:
    """
    Draw a stratified random sample from token_frequency.

    sample_k=0 returns the full vocabulary (no sampling).

    Tokens are split into three frequency bands:
      - High  : freq > 500
      - Mid   : freq 21–500
      - Low   : freq <= 20

    Each band contributes sample_k // 3 tokens (remainder goes to mid).
    All tokens in a band are included if the band is smaller than its quota.
    """
    if sample_k == 0:
        return {str(k): v for k, v in token_frequency.items()}

    rng = np.random.default_rng(seed)
    high = {t: f for t, f in token_frequency.items() if f > 500}
    mid  = {t: f for t, f in token_frequency.items() if 21 <= f <= 500}
    low  = {t: f for t, f in token_frequency.items() if f <= 20}

    per_band = sample_k // 3

    def _sample_band(band: dict, n: int) -> dict:
        keys = list(band.keys())
        chosen = rng.choice(keys, size=min(n, len(keys)), replace=False)
        return {str(k): band[k] for k in chosen}  # str() ensures JSON-serialisable keys

    sampled = {}
    sampled.update(_sample_band(high, per_band))
    sampled.update(_sample_band(mid,  per_band + (sample_k % 3)))  # remainder to mid
    sampled.update(_sample_band(low,  per_band))
    return sampled


def select_groups_via_llm(
    token_frequency: dict[str, int],
    group_llm_model: str,
    vocab_sample_k: int = 600,
    api_key: str = "",
    api_base: str = "",
    log_dir: str = ".",
) -> dict[str, list[str]]:
    """
    Ask an LLM to select 20 representative tokens per group from a stratified
    random sample of the observed tokenizer vocabulary.

    Returns a dict with keys matching MEDICAL_GROUPS_FALLBACK:
      "Group 1 (function words)", "Group 2 (clinical common)",
      "Group 3 (general medical)", "Group 4 (specific clinical)"

    Falls back to MEDICAL_GROUPS_FALLBACK on any error.
    """
    print(f"Selecting Table 1 token groups via LLM ({group_llm_model})...")

    # Build stratified sample; pass only token strings (no frequencies) to the LLM
    sample = _stratified_sample(token_frequency, vocab_sample_k)
    tokens_sorted = sorted(sample.keys(), key=lambda t: -sample[t])  # high-freq first
    token_list_str = json.dumps(tokens_sorted, ensure_ascii=False)

    user_prompt = (
        f"Token list:\n{token_list_str}\n\n"
        "Select exactly 20 tokens per group."
    )

    grouping_lm = LMCompletion(
        model=group_llm_model,
        api_key=api_key or None,
        request_url=api_base or None,
        log_mode="file",
        log_dir=log_dir,
        track_usage=True,
    )

    try:
        grouping_lm(_GROUP_SYSTEM_PROMPT, role="system", temperature=0.0)
        raw = grouping_lm(user_prompt, temperature=0.0)

        # Strip optional markdown code fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw.strip())
        parsed = json.loads(raw)

        groups = {
            "Group 1 (function words)":    parsed["group1"],
            "Group 2 (clinical common)":   parsed["group2"],
            "Group 3 (general medical)":   parsed["group3"],
            "Group 4 (specific clinical)": parsed["group4"],
        }

        if grouping_lm.usage:
            print(f"  LLM grouping API usage: {grouping_lm.usage['total_tokens']} tokens")
        print(f"  Groups selected: {', '.join(f'{k}({len(v)})' for k, v in groups.items())}")
        return groups

    except Exception as exc:
        print(f"  WARNING: LLM group selection failed ({exc}). "
              "Falling back to hard-coded MEDICAL_GROUPS_FALLBACK.")
        return MEDICAL_GROUPS_FALLBACK

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_text(sample: dict, text_field: str) -> str:
    """
    Extract the text to be tokenised from a messages-format sample.

    text_field options:
      "user"   – only the user turn (clinical event sequence)
      "system" – only the system turn (patient info + discharge summary)
      "all"    – system + user concatenated (full prompt, excluding assistant)
    """
    messages = sample.get("messages", [])
    role_map = {m["role"]: m["content"] for m in messages}
    if text_field == "user":
        return role_map.get("user", "")
    elif text_field == "system":
        return role_map.get("system", "")
    else:  # "all"
        return (role_map.get("system", "") + "\n" + role_map.get("user", "")).strip()


# ---------------------------------------------------------------------------
# Token norm utilities
# ---------------------------------------------------------------------------

def clean_token(token: str) -> str:
    """Strip leading/trailing punctuation, keep alphanumeric and medical symbols."""
    token = re.sub(r'^[^a-zA-Z0-9]+', '', token)
    token = re.sub(r'[^a-zA-Z0-9]+$', '', token)
    return token.strip().lower()


def compute_norms(embeddings: torch.Tensor) -> np.ndarray:
    """Return L2 norms of token embeddings, shape (seq_len,)."""
    return torch.norm(embeddings.squeeze(0), dim=-1).detach().cpu().float().numpy()


def accumulate_token_stats(
    tokenizer,
    input_ids: torch.Tensor,
    norms: np.ndarray,
    token_frequency: dict,
    token_norms: dict,
) -> None:
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().numpy())
    for token, norm in zip(tokens, norms):
        cleaned = clean_token(token)
        if cleaned:
            token_frequency[cleaned] += 1
            token_norms[cleaned].append(float(norm))


# ---------------------------------------------------------------------------
# Table 1: group statistics
# ---------------------------------------------------------------------------

def compute_group_stats(
    groups: dict[str, list[str]],
    token_frequency: dict,
    token_norms: dict,
) -> dict:
    """
    For each group, collect frequency and norm values of all matching tokens,
    then report mean/min/max.
    """
    results = {}
    for group_name, tokens in groups.items():
        freqs, norms = [], []
        for t in tokens:
            t_lower = t.lower()
            if t_lower in token_frequency:
                freqs.append(token_frequency[t_lower])
                norms.append(np.mean(token_norms[t_lower]))
        if freqs:
            results[group_name] = {
                "tokens_found": [t for t in tokens if t.lower() in token_frequency],
                "freq_mean": float(np.mean(freqs)),
                "freq_min":  float(np.min(freqs)),
                "freq_max":  float(np.max(freqs)),
                "norm_mean": float(np.mean(norms)),
                "norm_min":  float(np.min(norms)),
                "norm_max":  float(np.max(norms)),
            }
        else:
            results[group_name] = {
                "tokens_found": [],
                "freq_mean": float("nan"), "freq_min": float("nan"), "freq_max": float("nan"),
                "norm_mean": float("nan"), "norm_min": float("nan"), "norm_max": float("nan"),
            }
    return results


# ---------------------------------------------------------------------------
# Table 2: delta-hyperbolicity
# ---------------------------------------------------------------------------

def delta_hyp(dismat: np.ndarray) -> float:
    """
    Gromov delta-hyperbolicity via the 4-point condition with base point p=0.
    Implements the efficient algorithm of Fournier et al. (2015).
    """
    p = 0
    row = dismat[p, :][np.newaxis, :]   # (1, n)
    col = dismat[:, p][:, np.newaxis]   # (n, 1)
    XY_p = 0.5 * (row + col - dismat)   # Gromov products (n, n)
    # For each pair (i,j), compute max_k min(GP(i,k), GP(k,j))
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return float(np.max(maxmin - XY_p))


def get_delta(embeddings: torch.Tensor, max_points: int = 1500) -> tuple[float, float]:
    """
    Compute (delta, diameter) from a (1, seq_len, hidden) or (seq_len, hidden) tensor.
    Subsamples to max_points if seq_len > max_points.
    """
    feats = embeddings.squeeze(0).detach().cpu().float().numpy()  # (seq_len, hidden)
    n = len(feats)
    if n <= 1:
        return 0.0, 0.0
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        feats = feats[idx]
    dists = scipy_distance_matrix(feats, feats)
    delta = delta_hyp(dists)
    diam  = float(np.max(dists))
    return delta, diam


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_frequency_vs_norm(
    token_frequency: dict,
    token_norms: dict,
    output_path: str,
    dataset_name: str,
    bin_count: int = 50,
) -> None:
    """Reproduce Figure 1 (right): norm-binned average frequency histogram."""
    avg_norms = {t: np.mean(v) for t, v in token_norms.items()}
    freqs = np.array([token_frequency[t] for t in avg_norms])
    norms = np.array([avg_norms[t] for t in avg_norms])

    bins = np.linspace(norms.min(), norms.max(), bin_count)
    binned_freqs, binned_norm_centers = [], []
    for i in range(1, len(bins)):
        mask = (norms >= bins[i - 1]) & (norms < bins[i])
        if mask.sum() > 0:
            binned_freqs.append(np.mean(freqs[mask]))
            binned_norm_centers.append((bins[i - 1] + bins[i]) / 2)

    bin_widths = np.diff(bins[:len(binned_norm_centers) + 1])
    norm_arr = np.array(binned_norm_centers)
    colors = cm.coolwarm((norm_arr - norm_arr.min()) / (np.ptp(norm_arr) + 1e-9))

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(binned_norm_centers, binned_freqs, width=bin_widths, color=colors, edgecolor="gray")
    ax.set_yscale("log")
    ax.set_xlabel("Token Norm", fontsize=17)
    ax.set_ylabel("Token Frequency (log scale)", fontsize=17)
    ax.set_title(dataset_name, fontsize=17)
    ax.tick_params(labelsize=14)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_delta_distribution(
    delta_ratios: list[float],
    output_path: str,
    dataset_name: str,
) -> None:
    """Reproduce Table 2 histogram of per-sample delta values."""
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(5, 4))
    n, bins, patches = ax.hist(delta_ratios, bins=30, alpha=0.7)
    cmap = plt.cm.coolwarm
    norm_obj = plt.Normalize(0, 0.6)
    for center, patch in zip(0.5 * (bins[:-1] + bins[1:]), patches):
        patch.set_facecolor(cmap(norm_obj(center)))
    mean_v, std_v = np.mean(delta_ratios), np.std(delta_ratios)
    ax.text(
        0.95, 0.95,
        f"Mean: {mean_v:.3f}$\\pm${std_v:.3f}",
        transform=ax.transAxes, fontsize=14,
        va="top", ha="right",
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", alpha=0.5),
    )
    ax.set_xlabel("Hyperbolicity ($\\delta$)", fontsize=17)
    ax.set_ylabel("Frequency", fontsize=17)
    ax.set_title(dataset_name, fontsize=17)
    ax.tick_params(labelsize=14)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table1(group_stats: dict, model_name: str) -> None:
    header = f"\n{'='*80}\nTable 1 Equivalent — Token Frequency vs. Embedding Norm\nModel: {model_name}\n{'='*80}"
    print(header)
    print(f"{'Group':<38} {'Frequency (Mean [Min~Max])':<30} {'Norm (Mean [Min~Max])'}")
    print("-" * 80)
    for group, s in group_stats.items():
        if math.isnan(s["freq_mean"]):
            freq_str = "N/A (no tokens found)"
            norm_str = "N/A"
        else:
            freq_str = f"{s['freq_mean']:.1f} [{s['freq_min']:.0f}~{s['freq_max']:.0f}]"
            norm_str = f"{s['norm_mean']:.3f} [{s['norm_min']:.3f}~{s['norm_max']:.3f}]"
        print(f"{group:<38} {freq_str:<30} {norm_str}")
        if s["tokens_found"]:
            print(f"  (tokens found: {', '.join(s['tokens_found'])})")
    print()


def print_table2(delta_ratios: list[float], model_name: str, dataset_name: str) -> None:
    mean_v = np.mean(delta_ratios)
    std_v  = np.std(delta_ratios)
    print(f"\n{'='*80}")
    print(f"Table 2 Equivalent — delta-Hyperbolicity of Token Embeddings")
    print(f"Model: {model_name}  |  Dataset: {dataset_name}  |  Samples: {len(delta_ratios)}")
    print(f"{'='*80}")
    print(f"  delta_rel (2*delta/diam) = {mean_v:.4f} ± {std_v:.4f}")
    print(f"\n  Reference values from paper:")
    print(f"    Tree Graph:    0.00")
    print(f"    Scale-free:    0.00")
    print(f"    PubMed Graph:  0.40 ± 0.45")
    print(f"    Random Graph:  0.62 ± 0.34")
    print(f"    Sphere Space:  0.99 ± 0.01")
    print(f"\n  LLM baselines from paper (arithmetic reasoning datasets):")
    print(f"    LLaMA-7B:   0.08~0.10")
    print(f"    LLaMA-13B:  0.08~0.09")
    print(f"    LLaMA3-8B:  0.09~0.11")
    print(f"    Gemma-7B:   0.08~0.10")
    print()


def save_csv(
    token_frequency: dict,
    token_norms: dict,
    output_path: str,
) -> None:
    rows = []
    for token, freq in token_frequency.items():
        rows.append((token, freq, float(np.mean(token_norms[token]))))
    rows.sort(key=lambda x: x[1], reverse=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Token", "Frequency", "Average Norm"])
        writer.writerows(rows)
    print(f"Saved token statistics CSV: {output_path}")


def save_results_json(
    group_stats: dict,
    delta_ratios: list[float],
    model_name: str,
    dataset_name: str,
    output_path: str,
) -> None:
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "table1_group_stats": group_stats,
        "table2_delta_hyperbolicity": {
            "mean": float(np.mean(delta_ratios)),
            "std":  float(np.std(delta_ratios)),
            "n_samples": len(delta_ratios),
            "per_sample": delta_ratios,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce HypLoRA Table 1 & 2 on MIMIC-IV medical dataset"
    )
    # ── data / output ──────────────────────────────────────────────────────
    parser.add_argument("--data_path",   required=True,  help="Path to JSONL dataset file")
    parser.add_argument("--output_dir",  default="medical/analysis/results",
                        help="Directory to save figures and result files")
    parser.add_argument("--text_field",  default="all",
                        choices=["user", "system", "all"],
                        help="Which message role(s) to tokenise for analysis")
    parser.add_argument("--max_samples", default=0, type=int,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--max_points",  default=1500, type=int,
                        help="Max tokens per sample for delta computation (subsampled if exceeded)")
    parser.add_argument("--model_name",  default="",
                        help="Display name for the model (defaults to base_model basename)")
    # ── local model ──────────────────────────────────────────────────────────────────────────────────
    parser.add_argument("--base_model",  required=True,
                        help="HuggingFace model name or local path")
    # ── LLM grouping API (for --group_llm_model) ───────────────────────────────────────────────
    parser.add_argument("--api_key",     default="",
                        help="API key for --group_llm_model (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--api_base",    default="",
                        help="API base URL for --group_llm_model (falls back to OPENAI_BASE_URL env var)")
    parser.add_argument("--tables", default="all",
                        choices=["1", "2", "all"],
                        help="Which tables to reproduce: '1' = token norm analysis only, "
                             "'2' = delta-hyperbolicity only, 'all' = both (default: all). "
                             "Selecting '1' skips the O(n^3) delta computation; "
                             "selecting '2' still collects token stats but skips Table 1 output.")
    # ── Table 1 LLM grouping ──────────────────────────────────────────────────────────────────────────────────
    parser.add_argument("--group_llm_model", default="",
                        help="LLM model for Table 1 token group selection via API. "
                             "Uses same --api_base/--api_key. "
                             "If empty, falls back to hard-coded MEDICAL_GROUPS_FALLBACK.")
    parser.add_argument("--vocab_sample_k", default=600, type=int,
                        help="Number of tokens to sample for LLM group selection (default: 600)")
    args = parser.parse_args()

    if LMCompletion is None and args.group_llm_model:
        parser.error("--group_llm_model requires the 'openai' package: pip install openai")

    # Determine display name
    model_name = args.model_name or os.path.basename(args.base_model.rstrip("/"))

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from: {args.data_path}")
    samples = load_jsonl(args.data_path)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Samples to process: {len(samples)}")

    # ------------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.base_model}  (device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"":  int(os.environ.get("LOCAL_RANK") or 0)} if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    run_table1 = args.tables in ("1", "all")
    run_table2 = args.tables in ("2", "all")

    # ------------------------------------------------------------------
    # Main loop: token statistics (Table 1) + delta-hyperbolicity (Table 2)
    # ------------------------------------------------------------------
    token_frequency: dict[str, int]        = defaultdict(int)
    token_norms:     dict[str, list[float]] = defaultdict(list)
    delta_ratios:    list[float]            = []

    skip_delta_msg = "" if run_table2 else " (delta skipped: --tables=1)"
    print(f"\nAnalysing {len(samples)} samples (text_field='{args.text_field}'){skip_delta_msg}...")
    for sample in tqdm(samples, desc="Embedding analysis"):
        text = extract_text(sample, args.text_field)
        if not text.strip():
            continue

        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )["input_ids"]  # keep on CPU; moved to device below if needed

        input_ids = input_ids.to(device)
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(input_ids)  # (1, seq_len, hidden)

        # Table 1: accumulate token statistics (always needed for CSV/JSON)
        norms = compute_norms(embeddings)
        accumulate_token_stats(tokenizer, input_ids, norms, token_frequency, token_norms)

        # Table 2: delta-hyperbolicity (skip if --tables=1)
        if run_table2:
            delta, diam = get_delta(embeddings, max_points=args.max_points)
            if diam > 0:
                delta_ratios.append(2.0 * delta / diam)

    # ------------------------------------------------------------------
    # Table 1: group statistics
    # ------------------------------------------------------------------
    if run_table1:
        if args.group_llm_model:
            medical_groups = select_groups_via_llm(
                token_frequency,
                args.group_llm_model,
                vocab_sample_k=args.vocab_sample_k,
                api_key=args.api_key,
                api_base=args.api_base,
                log_dir=args.output_dir,
            )
        else:
            medical_groups = MEDICAL_GROUPS_FALLBACK
        group_stats = compute_group_stats(medical_groups, token_frequency, token_norms)
        print_table1(group_stats, model_name)
    else:
        medical_groups = MEDICAL_GROUPS_FALLBACK
        group_stats = compute_group_stats(medical_groups, token_frequency, token_norms)

    # ------------------------------------------------------------------
    # Table 2: delta-hyperbolicity summary
    # ------------------------------------------------------------------
    if run_table2:
        print_table2(delta_ratios, model_name, dataset_name)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    prefix = f"{model_name}_{dataset_name}_{args.text_field}"

    # Figure 1 equivalent (only if Table 1 was run)
    if run_table1:
        plot_frequency_vs_norm(
            token_frequency, token_norms,
            output_path=os.path.join(args.output_dir, f"{prefix}_freq_vs_norm.png"),
            dataset_name=f"{dataset_name} ({args.text_field})",
        )

    # Delta distribution histogram (only if Table 2 was run)
    if run_table2 and delta_ratios:
        plot_delta_distribution(
            delta_ratios,
            output_path=os.path.join(args.output_dir, f"{prefix}_delta_hist.png"),
            dataset_name=f"{dataset_name} ({args.text_field})",
        )

    # Token statistics CSV (always saved; useful for both tables)
    save_csv(
        token_frequency, token_norms,
        output_path=os.path.join(args.output_dir, f"{prefix}_token_stats.csv"),
    )

    # Full results JSON
    save_results_json(
        group_stats, delta_ratios, model_name, dataset_name,
        output_path=os.path.join(args.output_dir, f"{prefix}_results.json"),
    )

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
