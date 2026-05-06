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

Usage — local model:
    python medical/analysis/embedding_analysis.py \
        --data_path path/to/val.jsonl \
        --base_model Qwen/Qwen3-8B \
        --output_dir medical/analysis/results \
        [--text_field all]           # user | system | all
        [--max_samples 0]            # 0 = all samples
        [--max_points 1500]          # max tokens per sample for delta computation
        [--model_name Qwen3-8B]      # display name used in output filenames

Usage — API mode (OpenAI-compatible /v1/embeddings):
    python medical/analysis/embedding_analysis.py \
        --data_path path/to/val.jsonl \
        --base_model Qwen/Qwen3-8B \
        --api_mode \
        --embed_model text-embedding-3-small \
        [--api_key sk-...] \
        [--api_base https://api.openai.com/v1] \
        [--embed_batch_size 512] \
        [--output_dir medical/analysis/results]

    In API mode, --base_model is used ONLY to load the tokenizer (no model
    weights are loaded). The actual embedding vectors are fetched from the
    /v1/embeddings endpoint via LMCompletion.embed(). Each unique token string
    is embedded once and cached; the cache is reused across all samples.
    OPENAI_API_KEY and OPENAI_BASE_URL environment variables are respected.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MEDICAL_GROUPS = {
    "Group 1 (function words)": [
        "the", "is", "of", "and", "in", "to", "a",
    ],
    "Group 2 (clinical common)": [
        "patient", "blood", "mg", "day", "history",
    ],
    "Group 3 (general medical)": [
        "glucose", "sodium", "hemoglobin", "creatinine", "platelet",
    ],
    "Group 4 (specific clinical)": [
        "administered", "hematocrit", "fibrinogen", "aripiprazole", "phytonadione",
    ],
}

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
# API mode: build token embedding cache via LMCompletion.embed()
# ---------------------------------------------------------------------------

def _collect_unique_tokens(
    samples: list[dict],
    tokenizer,
    text_field: str,
) -> list[str]:
    """Tokenise all samples and return a sorted list of unique token strings."""
    unique_tokens: set[str] = set()
    for sample in tqdm(samples, desc="Tokenising for cache"):
        text = extract_text(sample, text_field)
        if not text.strip():
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze(0).numpy())
        for t in tokens:
            cleaned = clean_token(t)
            if cleaned:
                unique_tokens.add(cleaned)
    return sorted(unique_tokens)


def build_embedding_cache(
    samples: list[dict],
    tokenizer,
    text_field: str,
    lm: "LMCompletion",
    embed_model: str,
    embed_batch_size: int,
    parallel_workers: int = 1,
) -> dict[str, list[float]]:
    """
    Collect all unique token strings across all samples, embed them in batches
    via the /v1/embeddings endpoint, and return a {token_str: vector} cache.

    Each token is embedded in isolation (single-token input), which closely
    approximates the static input-embedding-matrix lookup used in the original
    HypLoRA paper.

    Args:
        parallel_workers: Number of concurrent API requests. 1 = sequential
                          (default). >1 enables parallel mode via
                          ThreadPoolExecutor, which significantly reduces
                          wall-clock time when the API has high per-request
                          latency.
    """
    print("Collecting unique tokens across all samples...")
    token_list = _collect_unique_tokens(samples, tokenizer, text_field)
    print(f"Unique tokens to embed: {len(token_list)}")

    # Split into batches
    batches = [
        (i, token_list[i: i + embed_batch_size])
        for i in range(0, len(token_list), embed_batch_size)
    ]
    n_batches = len(batches)

    if parallel_workers <= 1:
        # Sequential path (original behaviour)
        print(f"Fetching embeddings from API "
              f"(model={embed_model}, batch_size={embed_batch_size}, sequential)...")
        results: dict[int, list[list[float]]] = {}
        for i, batch in tqdm(batches, desc="Embedding batches", total=n_batches):
            results[i] = lm.embed(batch, model=embed_model)
    else:
        # Parallel path
        print(f"Fetching embeddings from API "
              f"(model={embed_model}, batch_size={embed_batch_size}, "
              f"workers={parallel_workers})...")

        def _fetch(idx_batch):
            idx, batch = idx_batch
            return idx, lm.embed(batch, model=embed_model)

        results: dict[int, list[list[float]]] = {}
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_fetch, b): b[0] for b in batches}
            with tqdm(total=n_batches, desc="Embedding batches") as pbar:
                for future in as_completed(futures):
                    idx, vecs = future.result()  # re-raises exceptions from threads
                    results[idx] = vecs
                    pbar.update(1)

    # Reassemble in original order
    vectors: list[list[float]] = []
    for i, _ in batches:
        vectors.extend(results[i])

    cache = {token_list[i]: vectors[i] for i in range(len(token_list))}
    print(f"Embedding cache built: {len(cache)} entries, dim={len(next(iter(cache.values())))}")
    return cache


def embeddings_from_cache(
    tokenizer,
    input_ids: torch.Tensor,
    cache: dict[str, list[float]],
) -> torch.Tensor | None:
    """
    Reconstruct a (1, seq_len, hidden) float tensor from the embedding cache.
    Tokens not found in the cache are skipped; returns None if no tokens match.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).numpy())
    rows = []
    for t in tokens:
        cleaned = clean_token(t)
        if cleaned and cleaned in cache:
            rows.append(cache[cleaned])
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float32)          # (n_found, hidden)
    return torch.from_numpy(arr).unsqueeze(0)        # (1, n_found, hidden)


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
                        help="Display name for the model (defaults to base_model / embed_model basename)")
    # ── local model ────────────────────────────────────────────────────────
    parser.add_argument("--base_model",  default="",
                        help="HuggingFace model name or local path. "
                             "In API mode, used only for tokenizer loading.")
    # ── API mode ───────────────────────────────────────────────────────────
    parser.add_argument("--api_mode",    action="store_true",
                        help="Use OpenAI-compatible /v1/embeddings API instead of local model")
    parser.add_argument("--embed_model", default="text-embedding-3-small",
                        help="Embedding model name for API mode (default: text-embedding-3-small)")
    parser.add_argument("--api_key",     default="",
                        help="API key (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--api_base",    default="",
                        help="API base URL (falls back to OPENAI_BASE_URL env var)")
    parser.add_argument("--embed_batch_size", default=512, type=int,
                        help="Number of tokens per /v1/embeddings request (default: 512)")
    parser.add_argument("--parallel_workers", default=1, type=int,
                        help="Number of concurrent /v1/embeddings requests in API mode "
                             "(default: 1 = sequential). Increase to e.g. 16 to reduce "
                             "wall-clock time when API latency is the bottleneck.")
    args = parser.parse_args()

    if not args.api_mode and not args.base_model:
        parser.error("--base_model is required in local mode")
    if args.api_mode and LMCompletion is None:
        parser.error("API mode requires the 'openai' package: pip install openai")

    # Determine display name
    if args.model_name:
        model_name = args.model_name
    elif args.api_mode:
        model_name = os.path.basename(args.embed_model.rstrip("/"))
    else:
        model_name = os.path.basename(args.base_model.rstrip("/"))

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
    # Load tokenizer (always needed) and optionally the model
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # API mode: build embedding cache; skip model loading entirely
    if args.api_mode:
        print(f"API mode: embedding model = {args.embed_model}")
        lm = LMCompletion(
            model=args.embed_model,
            api_key=args.api_key or None,
            request_url=args.api_base or None,
            log_mode="none",
            track_usage=True,
        )
        embedding_cache = build_embedding_cache(
            samples, tokenizer, args.text_field, lm, args.embed_model,
            args.embed_batch_size, args.parallel_workers,
        )
        model = None
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from: {args.base_model}  (device={device})")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map={"":  int(os.environ.get("LOCAL_RANK") or 0)} if device == "cuda" else "cpu",
            trust_remote_code=True,
        )
        model.eval()
        embedding_cache = None

    # ------------------------------------------------------------------
    # Main loop: token statistics (Table 1) + delta-hyperbolicity (Table 2)
    # ------------------------------------------------------------------
    token_frequency: dict[str, int]        = defaultdict(int)
    token_norms:     dict[str, list[float]] = defaultdict(list)
    delta_ratios:    list[float]            = []

    print(f"\nAnalysing {len(samples)} samples (text_field='{args.text_field}')...")
    for sample in tqdm(samples, desc="Embedding analysis"):
        text = extract_text(sample, args.text_field)
        if not text.strip():
            continue

        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )["input_ids"]  # keep on CPU; moved to device below if needed

        if args.api_mode:
            embeddings = embeddings_from_cache(tokenizer, input_ids, embedding_cache)
            if embeddings is None:
                continue
        else:
            input_ids = input_ids.to(device)
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(input_ids)  # (1, seq_len, hidden)

        # Table 1: accumulate token statistics
        norms = compute_norms(embeddings)
        accumulate_token_stats(tokenizer, input_ids, norms, token_frequency, token_norms)

        # Table 2: delta-hyperbolicity
        delta, diam = get_delta(embeddings, max_points=args.max_points)
        if diam > 0:
            delta_ratios.append(2.0 * delta / diam)

    # Print API usage summary if applicable
    if args.api_mode and lm.usage:
        print(f"\nAPI usage: {lm.usage['total_tokens']} total tokens")

    # ------------------------------------------------------------------
    # Table 1: group statistics
    # ------------------------------------------------------------------
    group_stats = compute_group_stats(MEDICAL_GROUPS, token_frequency, token_norms)
    print_table1(group_stats, model_name)

    # ------------------------------------------------------------------
    # Table 2: delta-hyperbolicity summary
    # ------------------------------------------------------------------
    print_table2(delta_ratios, model_name, dataset_name)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    prefix = f"{model_name}_{dataset_name}_{args.text_field}"

    # Figure 1 equivalent
    plot_frequency_vs_norm(
        token_frequency, token_norms,
        output_path=os.path.join(args.output_dir, f"{prefix}_freq_vs_norm.png"),
        dataset_name=f"{dataset_name} ({args.text_field})",
    )

    # Delta distribution histogram
    if delta_ratios:
        plot_delta_distribution(
            delta_ratios,
            output_path=os.path.join(args.output_dir, f"{prefix}_delta_hist.png"),
            dataset_name=f"{dataset_name} ({args.text_field})",
        )

    # Token statistics CSV
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
