"""
Normative Framework Analysis: Need / Utility / Affordability
Analyzes entity hallucination detection results under the normative framework.

Search impact is measured via a 3x3 bucket confusion matrix:
  Rows    = no-search score bucket    (Low / Mid / High)
  Columns = force-search score bucket (Low / Mid / High)

  Buckets:
    Low  [0.0, 0.5)
    Mid  [0.5, 0.9)
    High [0.9, 1.0]

  Diagonal   = score stayed in same bucket (no meaningful change)
  Upper tri  = search pushed score into a higher bucket (improved)
  Lower tri  = search pushed score into a lower bucket  (degraded)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

MODEL_IN_LEGEND = {
    "google_gemma-3-27b-it": "Gemma3-27B",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "Qwen_Qwen3-30B-A3B": "Qwen3-30B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen-3-30B-IT",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral3.1-24B-IT",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama3.2-3B",
}

# FOR BFCL
# TASK = "bfcl"
# BASE_ALL = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/"
# BASE_DIR   = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main"
# OUTPUT_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl"
# PERCEIVED_NEED_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main-perceived-need"

# BUDGET     = 10000        # dollars
# NEED_THRESHOLD     = 0.90   # score > this  → model does NOT need help
# UTILITY_POS_THRESH = 0.0    # delta > 0     → utility = +1
# UTILITY_NEG_THRESH = -0.05  # delta < -0.05 → utility = -1; in between → 0
# MAX_LEN = 314

# # Score buckets — shared for both no-search and force-search
# SCORE_BINS   = [-0.01,  0.99, 1.001]   # 1.001 so score==1.0 falls in High
# SCORE_LABELS = ["Incorrect",  "Correct"]

# # # Perceived-need analysis
# ACTUAL_NEED_SCORE_THRESH = 0.9    # no_search_score < this → actual_need = 1


# ── Config for InVivoGPT real user query ────────────────────────────────────────────────────────────────────
# TASK = 'invivo'
# BASE_ALL = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0"
# BASE_DIR   = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main"
# OUTPUT_DIR = f"/NS/chatgpt/work/qwu/hallucinations_detection/results/{TASK}"
# PERCEIVED_NEED_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main-perceived-need"

# BUDGET     = 10000        # dollars
# NEED_THRESHOLD     = 0.9   # score > this  → model does NOT need help
# UTILITY_POS_THRESH = 0.0    # delta > 0     → utility = +1
# UTILITY_NEG_THRESH = -0.05  # delta < -0.05 → utility = -1; in between → 0

# # Score buckets — shared for both no-search and force-search
# SCORE_BINS = [-np.inf, 0.1, 0.9, np.inf]  # 1.001 so score==1.0 falls in High
# SCORE_LABELS = ["Low", "Mid", "High"]
# MAX_LEN = 500

# # Perceived-need analysis
# ACTUAL_NEED_SCORE_THRESH = 0.9    # no_search_score < this → actual_need = 1

# # ── Config ────────────────────────────────────────────────────────────────────
# TASK = 'entity'
# BASE_ALL = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0"
# BASE_DIR   = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main"
# OUTPUT_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/normative"
# PERCEIVED_NEED_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need"

# BUDGET     = 10000        # dollars
# NEED_THRESHOLD     = 0.9   # score > this  → model does NOT need help
# UTILITY_POS_THRESH = 0.0    # delta > 0     → utility = +1
# UTILITY_NEG_THRESH = -0.05  # delta < -0.05 → utility = -1; in between → 0

# # Score buckets — shared for both no-search and force-search
# SCORE_BINS = [-np.inf, 0.1, 0.9, np.inf]  # 1.001 so score==1.0 falls in High
# SCORE_LABELS = ["Low", "Mid", "High"]
# MAX_LEN = 500

# Perceived-need analysis
# ACTUAL_NEED_SCORE_THRESH = 0.9    # no_search_score < this → actual_need = 1
BUDGET=10000
# ── All-task configs (for --task all) ─────────────────────────────────────────
TASK_CONFIGS = {
    "entity": dict(
        task                      = "entity",
        base_all                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0",
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/normative",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need",
        perceived_need_dir_v1     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need-v1",
        perceived_need_dir_v2     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need-v2",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        max_len                   = 500,
        need_threshold            = 0.9,
        actual_need_score_thresh  = 0.9,
        csv_suffix                = "",   # suffix for main/ files
        perceived_need_csv_suffix = "",   # suffix for main-perceived-need/ files
    ),
    "bfcl": dict(
        task                      = "bfcl",
        base_all                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result",
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main-perceived-need",
        perceived_need_dir_v1     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main-perceived-need-v1",  # bfcl has no v1/v2
        perceived_need_dir_v2     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main-perceived-need-v2",
        score_bins                = [-0.01, 0.99, 1.001],
        score_labels              = ["Incorrect", "Correct"],
        max_len                   = 314,
        need_threshold            = 0.90,
        actual_need_score_thresh  = 0.9,
        csv_suffix                = "", # main/ files use llm_judge
        perceived_need_csv_suffix = "",           # main-perceived-need/ files have no special suffix
    ),
    "invivo": dict(
        task                      = "invivo",
        base_all                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0",
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/invivo",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main-perceived-need",
        perceived_need_dir_v1     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main-perceived-need-v1",
        perceived_need_dir_v2     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main-perceived-need-v2",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        max_len                   = 500,
        need_threshold            = 0.9,
        actual_need_score_thresh  = 0.9,
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
    "perplexity": dict(
        task                      = "entity",
        base_all                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0",
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity_",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main-perceived-need",
        perceived_need_dir_v1     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main-perceived-need-v1",
        perceived_need_dir_v2     = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main-perceived-need-v2",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        max_len                   = 500,
        need_threshold            = 0.9,
        actual_need_score_thresh  = 0.9,
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
}

CSV_SUFFIX                = ""   # suffix for main/ files (no_search, force_search, with_search)
PERCEIVED_NEED_CSV_SUFFIX = ""   # suffix for main-perceived-need/ files
PERCEIVED_NEED_DIR_V1     = None # main-perceived-need-v1 dir (None if not applicable)
PERCEIVED_NEED_DIR_V2     = None # main-perceived-need-v2 dir (None if not applicable)

# ── Default globals (initialised from entity config; overridden by _apply_task_config) ──
_default_cfg             = TASK_CONFIGS["entity"]
TASK                     = _default_cfg["task"]
BASE_ALL                 = _default_cfg["base_all"]
BASE_DIR                 = _default_cfg["base_dir"]
OUTPUT_DIR               = _default_cfg["output_dir"]
PERCEIVED_NEED_DIR       = _default_cfg["perceived_need_dir"]
SCORE_BINS               = _default_cfg["score_bins"]
SCORE_LABELS             = _default_cfg["score_labels"]
MAX_LEN                  = _default_cfg["max_len"]
NEED_THRESHOLD           = _default_cfg["need_threshold"]
ACTUAL_NEED_SCORE_THRESH = _default_cfg["actual_need_score_thresh"]

BUDGET             = 10000   # total budget in USD
UTILITY_POS_THRESH = 0.0     # delta > 0     → utility = +1
UTILITY_NEG_THRESH = -0.05   # delta < -0.05 → utility = -1; in between → 0


def _apply_task_config(task: str):
    """Update all module-level globals for the given task."""
    global TASK, BASE_ALL, BASE_DIR, OUTPUT_DIR, PERCEIVED_NEED_DIR
    global PERCEIVED_NEED_DIR_V1, PERCEIVED_NEED_DIR_V2
    global SCORE_BINS, SCORE_LABELS, MAX_LEN, NEED_THRESHOLD, ACTUAL_NEED_SCORE_THRESH
    global CSV_SUFFIX, PERCEIVED_NEED_CSV_SUFFIX
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
    cfg = TASK_CONFIGS[task]
    TASK                     = cfg["task"]
    BASE_ALL                 = cfg["base_all"]
    BASE_DIR                 = cfg["base_dir"]
    OUTPUT_DIR               = cfg["output_dir"]
    PERCEIVED_NEED_DIR       = cfg["perceived_need_dir"]
    PERCEIVED_NEED_DIR_V1    = cfg.get("perceived_need_dir_v1")
    PERCEIVED_NEED_DIR_V2    = cfg.get("perceived_need_dir_v2")
    SCORE_BINS               = cfg["score_bins"]
    SCORE_LABELS             = cfg["score_labels"]
    MAX_LEN                  = cfg["max_len"]
    NEED_THRESHOLD           = cfg["need_threshold"]
    ACTUAL_NEED_SCORE_THRESH = cfg["actual_need_score_thresh"]
    CSV_SUFFIX               = cfg["csv_suffix"]
    PERCEIVED_NEED_CSV_SUFFIX = cfg["perceived_need_csv_suffix"]
    print(f"\n{'#'*60}")
    print(f"# Applying task config: {task}")
    print(f"#   BASE_DIR              = {BASE_DIR}")
    print(f"#   OUTPUT_DIR            = {OUTPUT_DIR}")
    print(f"#   csv_suffix            = '{CSV_SUFFIX}'")
    print(f"#   perceived_need_suffix = '{PERCEIVED_NEED_CSV_SUFFIX}'")
    print(f"{'#'*60}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def score_to_bucket(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        series,
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        right=False,
        include_lowest=True,
    )


def _extract_needs_tool(tool_calls: list) -> bool | None:
    """
    Extract the needs_tool boolean from a tool_calls list.

    Priority:
    1. decision.needs_tool — valid only when it was NOT set by a JSON parse
       error fallback (detected by checking decision.reasoning).
    2. raw_response — Mistral uses {{ / }} double-brace escaping which causes
       the live JSON parser to fail; fix the escaping and re-parse.
    Returns True/False, or None if no decision can be determined.
    """
    import json as _json
    import re as _re

    for tc in tool_calls:
        if tc.get("type") != "tool_selection":
            continue

        decision = tc.get("decision") or {}
        reasoning = str(decision.get("reasoning", ""))

        # If the decision was set by the error handler it says "JSON parse error"
        is_parse_error = "JSON parse error" in reasoning

        if not is_parse_error and "needs_tool" in decision:
            return bool(decision["needs_tool"])

        # Try raw_response (fixes Mistral {{ }} escaping)
        raw = tc.get("raw_response", "")
        if raw:
            fixed = raw.replace("{{", "{").replace("}}", "}")
            try:
                parsed = _json.loads(fixed)
                if "needs_tool" in parsed:
                    return bool(parsed["needs_tool"])
            except _json.JSONDecodeError:
                pass
            # Last resort: regex scan
            m = _re.search(r'"needs_tool"\s*:\s*(true|false)', raw, _re.IGNORECASE)
            if m:
                return m.group(1).lower() == "true"

    return None


def _patch_yes_no_from_jsonl(df: pd.DataFrame, jsonl_path: Path) -> pd.DataFrame:
    """
    Reconstruct yes_no_decision from the companion JSONL when the CSV column
    is entirely NaN (e.g. Mistral, whose JSON parser fails on {{ }} escaping).
    """
    import json as _json

    if not jsonl_path.exists():
        return df

    decisions: list = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                decisions.append(None)
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                decisions.append(None)
                continue

            # Already populated in some runs
            ynd = rec.get("yes_no_decision")
            if ynd is not None:
                decisions.append("yes" if str(ynd).strip().lower() in ("yes", "true", "1") else "no")
                continue

            needs_tool = _extract_needs_tool(rec.get("tool_calls") or [])
            if needs_tool is None:
                decisions.append(None)
            else:
                decisions.append("yes" if needs_tool else "no")

    if len(decisions) != len(df):
        print(f"  WARNING: JSONL row count ({len(decisions)}) != CSV row count "
              f"({len(df)}) in {jsonl_path.name}; skipping yes_no_decision patch.")
        return df

    df = df.copy()
    df["yes_no_decision"] = decisions
    n_yes = sum(1 for d in decisions if d == "yes")
    n_no  = sum(1 for d in decisions if d == "no")
    n_nan = sum(1 for d in decisions if d is None)
    print(f"  Patched yes_no_decision from JSONL: yes={n_yes}, no={n_no}, nan={n_nan}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from {path.name}")

    # If yes_no_decision is entirely missing, try to recover it from the
    # companion JSONL (e.g. Mistral stores the decision only in tool_calls).
    if "yes_no_decision" in df.columns and df["yes_no_decision"].isna().all():
        # _summary.csv  →  .jsonl  (drop the _summary suffix)
        jsonl_path = path.with_name(path.stem.replace("_summary", "") + ".jsonl")
        if jsonl_path.exists():
            df = _patch_yes_no_from_jsonl(df, jsonl_path)

    return df


# ── Pipeline steps ────────────────────────────────────────────────────────────

def compute_need(no_search_df: pd.DataFrame) -> pd.DataFrame:
    """Actual need: score <= NEED_THRESHOLD → model needs external tools (actual_need=1)."""
    df = no_search_df.copy()
    if "score" not in df.columns:
        raise ValueError("'score' column not found in no_search file.")
    df["actual_need"] = (df["score"] <= NEED_THRESHOLD).astype(int)
    print(f"  Need ratio (model needs help): {df['actual_need'].mean():.2%}")
    return df


def compute_utility(force_df: pd.DataFrame, no_search_df: pd.DataFrame) -> pd.DataFrame:
    """Actual utility based on delta = force_search_score - no_search_score."""
    if "score" not in force_df.columns or "score" not in no_search_df.columns:
        raise ValueError("'score' column missing in one of the files.")

    min_len = min(len(force_df), len(no_search_df))
    if len(force_df) != len(no_search_df):
        print(f"  WARNING: length mismatch ({len(force_df)} vs {len(no_search_df)}). "
              f"Using first {min_len} rows.")

    merged = force_df.iloc[:min_len].copy().reset_index(drop=True)
    merged["no_search_score"]    = no_search_df["score"].iloc[:min_len].values
    merged["force_search_score"] = force_df["score"].iloc[:min_len].values
    merged["delta"]              = merged["force_search_score"] - merged["no_search_score"]

    def utility_label(d):
        if pd.isna(d):              return np.nan
        if d > UTILITY_POS_THRESH:  return  1
        if d >= UTILITY_NEG_THRESH: return  0
        return -1

    merged["actual_utility"] = merged["delta"].apply(utility_label)
    print(f"  Utility distribution:\n"
          f"{merged['actual_utility'].value_counts().sort_index().to_string()}")
    return merged


def compute_bucket_confusion_matrix(utility_df: pd.DataFrame):
    """
    3x3 confusion matrix comparing score buckets between no-search and force-search.

    Rows    = no_search score bucket    (Low / Mid / High)
    Columns = force_search score bucket (Low / Mid / High)

    Returns
    -------
    cm_counts : pd.DataFrame  raw counts  (3x3)
    cm_pct    : pd.DataFrame  row-normalised percentages (3x3)
    summary   : pd.DataFrame  flat summary row with aggregate stats
    """
    df = utility_df.dropna(subset=["no_search_score", "force_search_score"]).copy()
    # df = utility_df
    n  = len(df)
    print(f'Total samples in the dataset:{n}')

    df["no_search_bucket"]    = score_to_bucket(df["no_search_score"])
    df["force_search_bucket"] = score_to_bucket(df["force_search_score"])

    # Raw counts
    cm_counts = (
        df.groupby(["no_search_bucket", "force_search_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SCORE_LABELS, columns=SCORE_LABELS, fill_value=0)
    )

    col_totals = cm_counts.sum(axis=0)
    cm_pct = cm_counts.div(col_totals, axis=1) * 100

    # Aggregate transition stats
    stayed   = int(np.diag(cm_counts.values).sum())
    improved = int(np.triu(cm_counts.values, k=1).sum())   # upper tri
    degraded = int(np.tril(cm_counts.values, k=-1).sum())  # lower tri

    print(f"\n  ── Bucket Confusion Matrix (no_search rows x force_search cols) ──")
    header = f"  {'':25s}" + "".join(f"  {c:>22s}" for c in SCORE_LABELS)
    print(header)
    print(f"  {'-' * (25 + 24 * 3)}")
    for row_label in SCORE_LABELS:
        row_str = f"  {row_label:25s}"
        for col_label in SCORE_LABELS:
            cnt = cm_counts.loc[row_label, col_label]
            pct = cm_pct.loc[row_label, col_label]
            row_str += f"  {cnt:>8,} ({pct:5.0f}%)"
        print(row_str)

    print(f"\n  Bucket transition summary (n={n:,}):")
    print(f"    Stayed   (same bucket)  : {stayed:,}   ({stayed/n:.1%})")
    print(f"    Improved (higher bucket): {improved:,}   ({improved/n:.1%})")
    print(f"    Degraded (lower bucket) : {degraded:,}   ({degraded/n:.1%})")
    print(f"\n    Mean no-search score    : {df['no_search_score'].mean():.4f}")
    print(f"    Mean force-search score : {df['force_search_score'].mean():.4f}")
    print(f"    Mean delta              : {df['delta'].mean():+.4f}")

    summary = pd.DataFrame([{
        "total_samples":           n,
        "stayed_same_bucket":      stayed,
        "improved_bucket":         improved,
        "degraded_bucket":         degraded,
        "stayed_pct":              round(stayed   / n, 6),
        "improved_pct":            round(improved / n, 6),
        "degraded_pct":            round(degraded / n, 6),
        "mean_no_search_score":    round(df["no_search_score"].mean(),    6),
        "mean_force_search_score": round(df["force_search_score"].mean(), 6),
        "overall_mean_delta":      round(df["delta"].mean(),              6),
    }])

    return cm_counts, cm_pct, summary

def plot_bucket_confusion_matrix(
    cm_counts: pd.DataFrame,
    cm_pct: pd.DataFrame,   # kept for compatibility, but recomputed below
    output_path: str,
    model_name: str
):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    counts = cm_counts.values.astype(int)

    # --- infer size dynamically ---
    nrows, ncols = counts.shape
    if nrows != ncols:
        raise ValueError(f"Confusion matrix must be square, got shape {counts.shape}")

    # Use SCORE_LABELS if available, otherwise fall back to DataFrame labels / generic names
    if "SCORE_LABELS" in globals() and len(SCORE_LABELS) == ncols:
        score_labels = SCORE_LABELS
    elif hasattr(cm_counts, "columns") and len(cm_counts.columns) == ncols:
        score_labels = [str(x) for x in cm_counts.columns]
    else:
        score_labels = [f"Bucket {i+1}" for i in range(ncols)]

    # Recompute exact column percentages from counts
    col_totals = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_exact = np.divide(
            counts,
            col_totals,
            out=np.zeros_like(counts, dtype=float),
            where=col_totals != 0
        ) * 100

    # Fix rounding so each column sums to exactly 100
    pct_rounded = _round_to_100_columnwise(pct_exact)

    # ---- Build a custom RGBA image based on direction + intensity ----
    COLOR_NEUTRAL = np.array([0.6, 0.6, 0.6])      # gray
    COLOR_HELPS   = np.array([0.18, 0.80, 0.44])   # green
    COLOR_HURTS   = np.array([0.91, 0.30, 0.24])   # red
    WHITE         = np.array([1.0, 1.0, 1.0])

    rgba_image = np.zeros((nrows, ncols, 4))

    # Full-strength cells:
    # - diagonal
    # - max-distance off-diagonal cells (e.g. corners for 3x3, both off-diagonals for 2x2)
    max_dist = ncols - 1
    full_cells = {
        (r, c)
        for r in range(nrows)
        for c in range(ncols)
        if (r == c) or (abs(r - c) == max_dist)
    }

    for r in range(nrows):
        for c in range(ncols):
            alpha = 1.0 if (r, c) in full_cells else 0.35

            if r == c:
                base = COLOR_NEUTRAL
            elif c > r:
                base = COLOR_HELPS
            else:
                base = COLOR_HURTS

            rgb = WHITE * (1 - alpha) + base * alpha
            rgba_image[r, c] = [*rgb, 1.0]

    ax.imshow(
        rgba_image,
        aspect="equal",
        origin="upper",
        extent=[-0.5, ncols - 0.5, nrows - 0.5, -0.5],
        zorder=0
    )

    # ---- Grid lines ----
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color="#EEEEEE", lw=0.5, zorder=1)
    for j in range(ncols + 1):
        ax.axvline(j - 0.5, color="#EEEEEE", lw=0.5, zorder=1)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # ---- annotations ----
    # Make font sizes adapt a bit for 2 vs 3 buckets
    value_fontsize = 48 if ncols == 2 else 43

    for r in range(nrows):
        for c in range(ncols):
            cnt = counts[r, c]
            pct = pct_rounded[r, c]
            text = f"{cnt}"

            ax.text(
                c,
                r + 0.1,
                text,
                ha="center",
                va="center",
                fontsize=value_fontsize,
                weight="semibold",
                color="black",
                zorder=10
            )

    # ---- axis labels ----
    ax.set_xlabel("Performance with tool", fontsize=35, fontweight="bold")
    ax.set_ylabel("Performance\nwithout tool", fontsize=35, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(score_labels, rotation=0, ha="center", fontsize=28)
    ax.set_yticklabels(score_labels, rotation=0, ha="right", fontsize=28)

    # ---- legend ----
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='Neutral',
               markerfacecolor='#444444', markersize=18),
        Line2D([0], [0], marker='o', color='w', label='Positive',
               markerfacecolor='#2ECC71', markersize=18),
        Line2D([0], [0], marker='o', color='w', label='Negative',
               markerfacecolor='#E74C3C', markersize=18),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(legend_items),
        frameon=True,
        borderaxespad=0,
        fontsize=35
    )
    # ---- "Actual Need" bracket ----

    if nrows == 3:
        # Low + Mid rows
        y_top = -0.5
        y_bottom = 1.5
        y_text = 0.5
        x_bracket = ncols - 0.4

    elif nrows == 2:
        # Incorrect rows (off-diagonal cells)
        y_top = -0.5
        y_bottom = 0.5
        y_text = 0
        x_bracket = ncols - 0.4

    else:
        y_top = None

    if y_top is not None:
        ax.plot([x_bracket, x_bracket], [y_top, y_bottom], color="#FF8C00", lw=4, clip_on=False)
        ax.plot([x_bracket, x_bracket - 0.1], [y_top, y_top], color="#FF8C00", lw=4, clip_on=False)
        ax.plot([x_bracket, x_bracket - 0.1], [y_bottom, y_bottom], color="#FF8C00", lw=4, clip_on=False)

        ax.text(
            x_bracket + 0.35,
            y_text,
            "True Need",
            rotation=90,
            va="center",
            ha="center",
            fontsize=35,
            color="#FF8C00",
            weight="bold",
            clip_on=False
        )

    plt.subplots_adjust(left=0.22)
    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Bucket confusion matrix saved -> {output_path}")

# ── Remaining pipeline (unchanged) ───────────────────────────────────────────

def compute_rank_list(utility_df: pd.DataFrame) -> pd.DataFrame:
    df       = utility_df.copy()
    nan_mask = df["delta"].isna()
    n_nan    = nan_mask.sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN delta values — placed at end of rank list.")

    ranked   = df[~nan_mask].copy().reset_index(drop=True)
    unranked = df[nan_mask].copy().reset_index(drop=True)
    ranked["rank"] = ranked["delta"].rank(ascending=False, method="first").astype(int)
    ranked = ranked.sort_values("rank").reset_index(drop=True)

    if not unranked.empty:
        unranked["rank"] = pd.NA
        ranked = pd.concat([ranked, unranked], ignore_index=True)

    priority_cols = ["rank", "delta", "force_search_score", "no_search_score",
                     "actual_utility", "actual_need"]
    front = [c for c in priority_cols if c in ranked.columns]
    rest  = [c for c in ranked.columns if c not in front]
    ranked = ranked[front + rest]

    print(f"  Rank list: {len(ranked):,} samples  "
          f"(top delta={ranked['delta'].iloc[0]:.4f}, "
          f"bottom delta={ranked['delta'].dropna().iloc[-1]:.4f})")
    return ranked

# ── Affordability (new) ───────────────────────────────────────────────────────

def compute_affordability(ranked_df: pd.DataFrame, budget: float = BUDGET) -> pd.DataFrame:
    """
    X-axis : k = number of top-gain tool calls (0 … min(n_helpful, 500))
    Y-axis : net Performance gain over baseline

    Only utility=+1 samples (delta > 0 AND no_search_bucket != High) are eligible.
    "Actual Need" bucket condition: no_search_score < 0.9 (Low or Mid bucket).
    """
    valid = ranked_df.dropna(subset=["delta"]).reset_index(drop=True)
    n_total = len(valid)

    # Bucket condition: only samples in Low or Mid no-search bucket (score < 0.9)
    valid["no_search_bucket"] = score_to_bucket(valid["no_search_score"])
    if TASK == 'bfcl':
        in_need_bucket = valid["no_search_bucket"].isin(
            ["Incorrect", "Correct"]
        )
    else:
        in_need_bucket = valid["no_search_bucket"].isin(
            ["Low", "Mid"]
        )

    # Eligible: delta > 0 AND in Low/Mid bucket (actual need + search helps)
    # IMPORTANT: preserve the positional index into `valid` (before reset) so we
    # can correctly index back into the running_scores array.
    eligible_mask = in_need_bucket & (valid["delta"] > 0)
    helpful = (
        valid[eligible_mask]
        .assign(_valid_pos=valid[eligible_mask].index)   # positional index in `valid`
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )
    n_helpful = len(helpful)
    k_max = min(n_helpful, 501)

    baseline_score = valid["no_search_score"].mean()
    oracle_score   = valid["force_search_score"].mean()
    oracle_gain    = oracle_score - baseline_score

    print(f"  Baseline score       (no search) : {baseline_score:.4f}")
    print(f"  Oracle score         (full swap)  : {oracle_score:.4f}")
    print(f"  Oracle gain                       : {oracle_gain:+.4f}")
    print(f"  Eligible (need bucket + delta>0)  : {n_helpful:,} / {n_total:,} "
          f"({n_helpful/n_total:.1%})")
    print(f"  Plotting up to k={k_max}")

    # Incrementally swap in top-k helpful entities using their original positions
    scores = valid["no_search_score"].copy().values.astype(float)

    afford_rows = [{"k": 0,
                    "net_gain": 0.0,
                    "optimized_score": round(float(scores.mean()), 8)}]

    running_scores = scores.copy()
    for k in range(1, k_max + 1):
        pos = int(helpful["_valid_pos"].iloc[k - 1])   # correct position in valid/scores
        running_scores[pos] = helpful["force_search_score"].iloc[k - 1]
        gain = float(running_scores.mean()) - baseline_score
        afford_rows.append({
            "k": k,
            "net_gain": round(gain, 8),
            "optimized_score": round(float(running_scores.mean()), 8),
        })

    afford_df = pd.DataFrame(afford_rows)
    afford_df["baseline_score"] = round(baseline_score, 6)
    afford_df["oracle_gain"]    = round(oracle_gain, 6)
    afford_df["pct_of_oracle"]  = (
        (afford_df["net_gain"] / oracle_gain * 100).round(2)
        if oracle_gain != 0 else 0.0
    )

    milestones = sorted(set(m for m in [0, 1, 10, 50, 100, 200, 500] if m <= k_max))
    print(f"\n  {'k':>7}  {'Optimized':>10}  {'Net Gain':>10}  {'% Oracle':>9}")
    print(f"  {'-'*42}")
    for m in milestones:
        row = afford_df.iloc[m]
        print(f"  {int(row['k']):>7,}  {row['optimized_score']:>10.4f}  "
              f"{row['net_gain']:>+10.4f}  {row['pct_of_oracle']:>8.0f}%")

    return afford_df

def plot_affordability(
    afford_dfs: dict,
    output_path: str,
    results_base_dir: str = BASE_ALL,
    no_search_dfs: dict = None,
    budget_aware_suffix: str = "",
):
    """
    Plot utility gain vs tool cost for multiple models.

    Notes:
    - Solid line: ideal affordability curve
    - Dashed line + dots: experimental points
    - X-axis is tool cost in dollars on a reversed log scale
    - The ideal curve is extended to the left using the y-value of the leftmost
      valid point (largest x value after sorting).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BUDGET_TOTAL = 10_000
    LEFT_X = BUDGET_TOTAL
    LEFT_X_PAD = BUDGET_TOTAL * 1.15
    RIGHT_X = 1.0  # proxy for cost=0 on log axis
    cost_setups = [10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0]

    palette = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#7f7f7f",  # gray
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")

    model_items = list(afford_dfs.items())
    actual_calls_data = {}   # model_name -> list of (cost, k_budget, k_actual)
    no_cost_desc_calls = {}  # model_name -> k_ws (tool calls without cost description)

    for model_idx, (model_name, afford_df) in enumerate(model_items):
        color = palette[model_idx % len(palette)]

        no_search_df = no_search_dfs[model_name]
        baseline_score = no_search_df["score"].mean()
        total_n = len(no_search_df)

        # ── Ideal affordability curve: convert k -> cost = budget / k ─────────
        x_k = afford_df["k"].to_numpy(dtype=float)
        y = afford_df["net_gain"].to_numpy(dtype=float)

        if len(x_k) == 0 or len(y) == 0:
            continue

        # Keep only meaningful k values for interior curve
        valid = (x_k > 0) & (x_k < total_n)

        x_plot = x_k[valid] / total_n * 100  # convert to % of queries
        y_plot = y[valid]

        # Sort by x ascending for plotting
        if len(x_plot) > 0:
            order = np.argsort(x_plot)
            x_plot = x_plot[order]
            y_plot = y_plot[order]

        # Determine right-edge y value (100% = all calls)
        search_all = (x_k >= total_n)
        if np.any(search_all):
            right_y = y[search_all][-1]
        elif len(y_plot) > 0:
            right_y = y_plot[-1]
        else:
            continue

        # Determine left-edge y value (0% = no calls)
        left_y = y_plot[0] if len(y_plot) > 0 else right_y

        # Explicitly extend both ends
        x_plot = np.concatenate(([0.0], x_plot, [100.0]))
        y_plot = np.concatenate(([left_y], y_plot, [right_y]))

        ax.plot(
            x_plot,
            y_plot,
            color=color,
            linewidth=5,
            linestyle="-",
            solid_capstyle="round",
            alpha=0.95,
            label=MODEL_IN_LEGEND[model_name],
        )
        # ── Experimental points ──────────────────────────────────────────────
        exp_points = []
        actual_calls = []   # (cost, k_budget, k_actual) for the second figure

        for cost in cost_setups:
            cost_tag = str(int(cost))
            csv_path = (
                f"{results_base_dir}/tool-cost-{cost_tag}-budget-aware{budget_aware_suffix}/"
                f"vllm_{model_name}_with_search_summary{CSV_SUFFIX}.csv"
            )
            force_search_path = (
                f"{results_base_dir}/main/"
                f"vllm_{model_name}_force_search_summary{CSV_SUFFIX}.csv"
            )

            try:
                exp_df = load_csv(csv_path)
                force_search_df = load_csv(force_search_path)

                if "search_called" not in exp_df.columns or "score" not in exp_df.columns:
                    print(f"  WARNING: missing columns in {csv_path}, skipping.")
                    continue

                k_actual = int(exp_df["search_called"].sum())
                k_budget = total_n if cost == 0 else int(BUDGET_TOTAL / cost)
                actual_calls.append((cost, k_budget, k_actual))

                base_scores = no_search_df["score"].values.copy().astype(float)
                search_mask = exp_df["search_called"].astype(bool).values

                force_scores = force_search_df["score"].values.copy().astype(float)
                nan_force = np.isnan(force_scores)
                if nan_force.any():
                    print(
                        f"  WARNING: {nan_force.sum()} NaN force-search scores, "
                        f"falling back to no-search score."
                    )
                    force_scores[nan_force] = no_search_df["score"].values[nan_force]

                merged_scores = base_scores.copy()

                if search_mask.sum() > k_budget:
                    true_indices = np.where(search_mask)[0]
                    capped_mask = np.zeros_like(search_mask, dtype=bool)
                    capped_mask[true_indices[:k_budget]] = True
                    search_mask = capped_mask

                merged_scores[search_mask] = force_scores[search_mask]
                net_gain = float(merged_scores.mean()) - baseline_score

                exp_points.append((cost, net_gain))
                print(
                    f"  [{model_name}] Cost ${cost:>6}: "
                    f"k_actual={k_actual:>6,}  net_gain={net_gain:+.4f}"
                )

            except FileNotFoundError:
                print(f"  WARNING: file not found — {csv_path}")
            except Exception as e:
                print(f"  WARNING: error reading {csv_path}: {e}")

        processed_points = []
        for cost, net_gain in exp_points:
            plot_pct = 100.0 if cost == 0 else min(BUDGET_TOTAL / cost / total_n * 100, 100.0)
            processed_points.append((plot_pct, net_gain))

        processed_points.sort(key=lambda x: x[0])

        if processed_points:
            xs, ys = zip(*processed_points)

            ax.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=4,
                color=color,
                alpha=0.8,
                zorder=4,
            )

            ax.scatter(
                xs,
                ys,
                s=250,
                color=color,
                edgecolor="white",
                linewidth=2,
                zorder=5,
                alpha=0.95,
            )

        # ── Force-search gain: horizontal line at avg(force_search_score) ─────
        force_search_path = (
            f"{results_base_dir}/main/"
            f"vllm_{model_name}_force_search_summary{CSV_SUFFIX}.csv"
        )
        try:
            force_search_df = load_csv(force_search_path)
            force_scores = force_search_df["score"].values.copy().astype(float)
            nan_force = np.isnan(force_scores)
            if nan_force.any():
                force_scores[nan_force] = no_search_df["score"].values[nan_force]
            force_gain = float(force_scores.mean()) - baseline_score
            ax.axhline(
                force_gain,
                color=color,
                linewidth=3,
                linestyle=":",
                alpha=0.6,
                zorder=3,
            )
        except FileNotFoundError:
            print(f"  WARNING: file not found — {force_search_path}")

        # ── With-search gain: single point from main with_search CSV ──────────
        # x = k_actual (search_called.sum()), y = merged gain
        with_search_path = (
            f"{results_base_dir}/main/"
            f"vllm_{model_name}_with_search_summary{CSV_SUFFIX}.csv"
        )
        try:
            with_search_df = load_csv(with_search_path)
            if "search_called" in with_search_df.columns:
                ws_mask = with_search_df["search_called"].astype(bool).values
                k_ws = int(ws_mask.sum())
                base_scores = no_search_df["score"].values.copy().astype(float)
                ws_merged = base_scores.copy()
                ws_merged[ws_mask] = force_scores[ws_mask]
                ws_gain = float(ws_merged.mean()) - baseline_score
                # x-position: k_ws on the cost axis → cost = BUDGET_TOTAL / k_ws
                ws_plot_x = k_ws / total_n * 100
                no_cost_desc_calls[model_name] = k_ws
                ax.scatter(
                    [ws_plot_x],
                    [ws_gain],
                    s=250,
                    color=color,
                    edgecolor="black",
                    linewidth=2,
                    zorder=6,
                    marker="s",
                )
                print(
                    f"  [{model_name}] With-search (main): "
                    f"k_actual={k_ws}  ws_gain={ws_gain:+.4f}  plot_x={ws_plot_x:.2f}"
                )
        except FileNotFoundError:
            print(f"  WARNING: file not found — {with_search_path}")

        actual_calls_data[model_name] = actual_calls

    handles, labels = ax.get_legend_handles_labels()
    point_handle = Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=10,
        linestyle="None",
    )
    force_handle = Line2D([0], [0], color="black", linewidth=2.5, linestyle="-", alpha=0.6)
    with_handle  = Line2D(
        [0], [0],
        marker="s",
        color="none",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=16,
        linestyle="None",
    )
    handles.append(point_handle)
    # handles.append(force_handle)
    handles.append(with_handle)
    # labels.append("Force-search gain")
    labels.append("Cost Description")
    labels.append("No Cost Description")

    ax.set_xlim(-2, 105)

    tick_positions = [0, 20, 40, 60, 80, 100]
    tick_labels = [f"{p}%" for p in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=20, rotation=0)

    ax.set_xlabel("% Queries with Tool Calls", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Utility Gain over No-Tool", fontsize=25, fontweight="bold")
    ax.tick_params(axis="y", labelsize=22)

    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
        fontsize=25, 
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.subplots_adjust(top=0.82)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Affordability curve saved -> {output_path}")
    return actual_calls_data, no_cost_desc_calls


def plot_actual_tool_calls(
    actual_calls_data: dict,
    output_path: str,
    no_cost_desc_calls: dict = None,
):
    """
    Figure: actual tool-call count vs budget limit per cost setup.

    X-axis : tool cost ($), same log scale as the affordability figure,
             with $0 proxy at RIGHT_X=1.0. Each tick annotated with
             (k_budget calls, pct%).
    Y-axis : number of times the model actually called the tool (k_actual
             from the budget-aware with_search CSV), without any budget cap.
    Ideal line : y = k_budget — the budget-imposed limit at each cost.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BUDGET_TOTAL = 10_000
    TOTAL_NUM    = MAX_LEN        # 500
    max_ideal_y  = 314 if TASK == "bfcl" else TOTAL_NUM

    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")

    RIGHT_X    = 1.0              # proxy for $0 on log axis
    LEFT_X_PAD = BUDGET_TOTAL * 1.15

    for model_idx, (model_name, calls) in enumerate(actual_calls_data.items()):
        if not calls:
            continue
        color = palette[model_idx % len(palette)]

        calls_sorted = sorted(calls, key=lambda p: p[0], reverse=True)
        xs = [RIGHT_X if c == 0 else float(c) for c, kb, ka in calls_sorted]
        ys = [ka for c, kb, ka in calls_sorted]

        ax.plot(xs, ys, color=color, linewidth=4, linestyle="-",
                solid_capstyle="round", alpha=0.9,
                # label=MODEL_IN_LEGEND.get(model_name, model_name)
                )
        ax.scatter(xs, ys, s=200, color=color, edgecolor="white",
                   linewidth=2, zorder=5, alpha=0.95)

    # No-cost-description lines: horizontal line per model at k_ws
    if no_cost_desc_calls:
        for model_idx, (model_name, k_ws) in enumerate(no_cost_desc_calls.items()):
            color = palette[model_idx % len(palette)]
            ax.axhline(k_ws, color=color, linewidth=2.5, linestyle=":",
                       alpha=0.8, zorder=3)

    # Ideal line: y = k_budget at each cost
    tick_costs = [10, 20, 25, 29, 33, 40, 50, 67, 100, 200, 222, 250, 500, 1000, 10000]
    ideal_xs = [float(c) for c in tick_costs] + [RIGHT_X]
    ideal_ys = [min(int(BUDGET_TOTAL / c), max_ideal_y) for c in tick_costs] + [max_ideal_y]
    ideal_pairs = sorted(zip(ideal_xs, ideal_ys), key=lambda p: p[0], reverse=True)
    ideal_xs = [p[0] for p in ideal_pairs]
    ideal_ys = [p[1] for p in ideal_pairs]
    ax.plot(ideal_xs, ideal_ys, color="black", linewidth=3, linestyle="--",
            alpha=0.7, zorder=3, label="Maximum Tool Calls")
    ax.scatter(ideal_xs, ideal_ys, s=120, color="black",
               edgecolor="white", linewidth=1.5, zorder=4, alpha=0.8)

    # ── X-axis: log scale, reversed (high cost left, low cost right) ──────────
    labelled = {10, 25, 50, 100, 250, 500, 1000, 10000, RIGHT_X}
    all_ticks = [10, 20, 25, 29, 33, 40, 50, 67, 100, 200, 222, 250, 500, 1000, 10000, RIGHT_X]
    tick_labels = []
    for pos in all_ticks:
        if pos not in labelled:
            tick_labels.append("")
        elif pos == RIGHT_X:
            tick_labels.append("$0")
        else:
            tick_labels.append(f"${int(pos):,}")

    ax.set_xscale("log")
    ax.set_xlim(LEFT_X_PAD, 0.8)
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(tick_labels, fontsize=20, rotation=45, ha="right")
    ax.set_xlabel("Tool Cost ($)", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Actual Tool Calls", fontsize=25, fontweight="bold")
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=22)

    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="lower center", bbox_to_anchor=(0.5, 1),
              ncol=2, frameon=False, fontsize=25,
              columnspacing=1.0, handletextpad=0.4)

    plt.subplots_adjust(top=0.82)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Actual tool-calls figure saved -> {output_path}")


def plot_ndcg_rank_correlation(
    model_data: dict,
    output_path: str,
    results_base_dir: str,
    budget_aware_suffix: str = "",
):
    """
    Plot NDCG@k rank correlation for budget-aware tool-call decisions.

    For each model, iterates over tool-cost-*-budget-aware{suffix}/ directories.
    At each cost level the model produces a specific set of search_called decisions
    (k_actual calls).  NDCG@k_actual is computed using the ground-truth delta
    ranking, then plotted at x = k_actual / total_n * 100 (same x-axis as the
    affordability plot).

    model_data        : model_name -> {"utility_df": ..., "total_n": int}
    results_base_dir  : root dir containing tool-cost-* sub-directories
    budget_aware_suffix: "" for original, "-v2" for v2
    """
    from sklearn.metrics import ndcg_score as sklearn_ndcg

    cost_setups = [10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0]

    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")

    for model_idx, (model_name, data) in enumerate(model_data.items()):
        color   = palette[model_idx % len(palette)]

        utility_df = data["utility_df"]
        total_n    = data["total_n"]

        # ── delta ranked (true utility of calling search for each instance) ────
        raw_delta = utility_df["delta"].fillna(0).values.astype(float)
        delta     = pd.Series(raw_delta).rank(method="average", ascending=True).values
        y_true    = delta.reshape(1, -1)

        BUDGET_TOTAL = 10_000
        points = []   # (x_pct, ndcg_val)

        for cost in cost_setups:
            cost_tag  = str(int(cost))
            k_budget  = total_n if cost == 0 else int(BUDGET_TOTAL / cost)
            csv_path  = (
                f"{results_base_dir}/tool-cost-{cost_tag}-budget-aware{budget_aware_suffix}/"
                f"vllm_{model_name}_with_search_summary{CSV_SUFFIX}.csv"
            )
            try:
                exp_df = load_csv(csv_path)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  WARNING: error reading {csv_path}: {e}")
                continue

            if "search_called" not in exp_df.columns:
                print(f"  WARNING: 'search_called' missing in {csv_path}, skipping.")
                continue

            raw = exp_df["search_called"]
            if pd.api.types.is_bool_dtype(raw):
                search_called = raw.astype(int).values
            elif pd.api.types.is_numeric_dtype(raw):
                search_called = (raw != 0).astype(int).values
            else:
                search_called = (
                    raw.astype(str).str.strip().str.lower()
                       .isin(["1", "true", "yes", "y"])
                ).astype(int).values

            min_len = min(len(delta), len(search_called))
            sc      = search_called[:min_len]

            # ── cap at budget (mirror plot_affordability logic) ───────────────
            k_actual = int(sc.sum())
            if k_actual > k_budget:
                true_indices = np.where(sc)[0]
                capped = np.zeros_like(sc)
                capped[true_indices[:k_budget]] = 1
                sc = capped
                k_actual = k_budget

            if k_actual == 0:
                continue

            # x-axis follows budget (same as affordability plot)
            x_pct    = min(k_budget, total_n) / total_n * 100
            y_score  = sc.astype(float).reshape(1, -1)
            ndcg_val = sklearn_ndcg(y_true[:, :min_len], y_score, k=k_actual)
            points.append((x_pct, ndcg_val))
            print(
                f"  [{model_name}] cost=${cost:>6}  k_budget={k_budget:>5}  "
                f"k_actual={k_actual:>5}  x={x_pct:.1f}%  NDCG@{k_actual}={ndcg_val:.4f}"
            )

        if not points:
            print(f"  [{model_name}] No budget-aware data found — skipping.")
            continue

        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(
            xs, ys,
            color=color, linewidth=5, linestyle="-",
            solid_capstyle="round", alpha=0.95,
            label=MODEL_IN_LEGEND.get(model_name, model_name),
        )
        ax.scatter(
            xs, ys,
            s=120, color=color, edgecolor="white",
            linewidth=1.5, zorder=5, alpha=0.95,
        )

    ax.set_xlim(-2, 105)
    tick_positions = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{p}%" for p in tick_positions], fontsize=20, rotation=0)

    ax.set_xlabel("% Queries with Tool Calls", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("NDCG@k", fontsize=25, fontweight="bold")
    ax.tick_params(axis="y", labelsize=22)

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
        fontsize=25,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    plt.subplots_adjust(top=0.82)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  NDCG rank-correlation figure saved -> {output_path}")


def compute_perceived_need_matrix(no_search_df: pd.DataFrame,
                                   perceived_need_df: pd.DataFrame):
    """
    2×2 confusion matrix: Actual Need (rows) vs Perceived Need (cols).

    Actual need   : no_search_score < ACTUAL_NEED_SCORE_THRESH  → 1 (needs help)
    Perceived need: yes_no_decision == 'yes'                     → 1 (model thinks it needs help)

    Matrix layout (standard confusion matrix orientation):
                        Perceived Need=1    Perceived Need=0
      Actual Need=1     True Need (TP)      Missed Need (FN)
      Actual Need=0     False Alarm (FP)    No Need (TN)

    Returns
    -------
    cm_counts : pd.DataFrame  raw counts  (2×2)
    cm_pct    : pd.DataFrame  row-normalised percentages (2×2)
    stats     : dict          precision, recall, F1, Performance
    """
    col = "yes_no_decision"
    if col not in perceived_need_df.columns:
        # Try stripping whitespace from column names
        perceived_need_df.columns = perceived_need_df.columns.str.strip()
        if col not in perceived_need_df.columns:
            raise ValueError(f"Column '{col}' not found. "
                             f"Available: {list(perceived_need_df.columns)}")

    min_len = min(len(no_search_df), len(perceived_need_df))
    if len(no_search_df) != len(perceived_need_df):
        print(f"  WARNING: length mismatch (no_search={len(no_search_df)}, "
              f"perceived={len(perceived_need_df)}). Using first {min_len} rows.")

    scores = no_search_df["score"].iloc[:min_len].values
    raw    = perceived_need_df[col].iloc[:min_len]

    # Handle boolean, numeric (0/1), or string ("yes"/"no") column gracefully
    if pd.api.types.is_bool_dtype(raw):
        perceived_need = raw.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw):
        perceived_need = (raw != 0).astype(int).values
    else:
        perceived_need = (raw.astype(str).str.strip().str.lower() == "yes").astype(int).values

    actual_need = (scores < ACTUAL_NEED_SCORE_THRESH).astype(int)   # 1 = needs help

    LABELS     = ["Need (score<0.9)", "No Need (score≥0.9)"]
    LABEL_VALS = [1, 0]   # rows/cols in this order

    cm_data = {}
    for col_val, col_lbl in zip(LABEL_VALS, ["Perceived Need\n(yes)", "Perceived No Need\n(no)"]):
        cm_data[col_lbl] = [
            int(((actual_need == rv) & (perceived_need == col_val)).sum())
            for rv in LABEL_VALS
        ]

    cm_counts = pd.DataFrame(cm_data, index=LABELS)
    row_totals = cm_counts.sum(axis=0)
    cm_pct     = cm_counts.div(row_totals, axis=1) * 100

    tp = cm_counts.iloc[0, 0]   # actual=need,    perceived=need
    fn = cm_counts.iloc[0, 1]   # actual=need,    perceived=no need  (missed)
    fp = cm_counts.iloc[1, 0]   # actual=no need, perceived=need     (false alarm)
    tn = cm_counts.iloc[1, 1]   # actual=no need, perceived=no need

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))
    Performance  = (tp + tn) / (tp + fn + fp + tn)

    print(f"\n  ── Perceived Need vs Actual Need (2×2 confusion matrix) ──")
    print(f"  {'':30s}  {'Perceived Need (yes)':>22s}  {'Perceived No Need (no)':>24s}")
    print(f"  {'-' * 82}")
    for lbl in LABELS:
        row_str = f"  {lbl:30s}"
        for c in cm_counts.columns:
            row_str += f"  {int(cm_counts.loc[lbl, c]):>10,} ({cm_pct.loc[lbl, c]:5.0f}%)"
        print(row_str)
    print(f"\n  Precision (of perceived need calls) : {precision:.4f}")
    print(f"  Recall    (actual need captured)    : {recall:.4f}")
    print(f"  F1 score                            : {f1:.4f}")
    print(f"  Performance                            : {Performance:.4f}")

    stats = dict(precision=round(precision, 6), recall=round(recall, 6),
                 f1=round(f1, 6), Performance=round(Performance, 6),
                 tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
                 n=int(min_len))
    return cm_counts, cm_pct, stats

def compute_perceived_utility_matrix(
    no_search_df: pd.DataFrame,
    force_search_df: pd.DataFrame,
    auto_search_df: pd.DataFrame,
    thresholds: list = [0.1, 0.9],
):
    """
    3×2 confusion-style matrix:
        Actual Utility (rows) vs Perceived Utility (cols)

    Actual utility is derived by comparing no-search vs force-search score buckets:
        low:  [0, t_low)
        mid:  [t_low, t_high)
        high: [t_high, 1]

        bucket_force > bucket_no  -> 2 (search helps)
        bucket_force == bucket_no -> 1 (no change)
        bucket_force < bucket_no  -> 0 (search hurts)

    Perceived utility is derived from auto_search_df["search_called"]:
        search_called == 1 / True / "yes" -> 1 (model thinks search is useful)
        otherwise                         -> 0 (model thinks no search needed)

    Matrix layout:
                              Perceived Useful    Perceived Not Useful
      Actual Helps                 TP-like              Missed Help
      Actual No Change             False Alarm          Correct No-Search
      Actual Hurts                 Bad Call             Correct Avoid

    Returns
    -------
    cm_counts : pd.DataFrame   raw counts
    cm_pct    : pd.DataFrame   row-normalized percentages
    stats     : dict           summary metrics
    """

    # ---------- validate inputs ----------
    if "score" not in no_search_df.columns:
        raise ValueError(f"'score' column not found in no_search_df. Available: {list(no_search_df.columns)}")
    if "score" not in force_search_df.columns:
        raise ValueError(f"'score' column not found in force_search_df. Available: {list(force_search_df.columns)}")

    auto_search_df.columns = auto_search_df.columns.str.strip()
    if "search_called" not in auto_search_df.columns:
        raise ValueError(f"'search_called' column not found in auto_search_df. Available: {list(auto_search_df.columns)}")

    if len(thresholds) != 2:
        raise ValueError(f"thresholds must have length 2, got {thresholds}")

    t_low, t_high = sorted(thresholds)
    if not (0 <= t_low <= t_high <= 1):
        raise ValueError(f"thresholds must satisfy 0 <= low <= high <= 1, got {thresholds}")

    # ---------- align lengths ----------
    min_len = min(len(no_search_df), len(force_search_df), len(auto_search_df))
    if len({len(no_search_df), len(force_search_df), len(auto_search_df)}) != 1:
        print(
            f"  WARNING: length mismatch "
            f"(no_search={len(no_search_df)}, force_search={len(force_search_df)}, auto_search={len(auto_search_df)}). "
            f"Using first {min_len} rows."
        )

    scores_no = pd.to_numeric(no_search_df["score"].iloc[:min_len], errors="coerce").values
    scores_force = pd.to_numeric(force_search_df["score"].iloc[:min_len], errors="coerce").values
    raw_search_called = auto_search_df["search_called"].iloc[:min_len]

    # ---------- perceived utility ----------
    if pd.api.types.is_bool_dtype(raw_search_called):
        perceived_useful = raw_search_called.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw_search_called):
        perceived_useful = (raw_search_called != 0).astype(int).values
    else:
        perceived_useful = (
            raw_search_called.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        ).astype(int).values

    # ---------- actual utility ----------
    def bucketize(score: float) -> float:
        if pd.isna(score):
            return np.nan
        if score < t_low:
            return 0   # low
        if score < t_high:
            return 1   # mid
        return 2       # high

    bucket_no = np.array([bucketize(x) for x in scores_no], dtype=float)
    bucket_force = np.array([bucketize(x) for x in scores_force], dtype=float)

    valid_mask = ~np.isnan(bucket_no) & ~np.isnan(bucket_force)
    if valid_mask.sum() < min_len:
        print(f"  WARNING: skipped {min_len - valid_mask.sum()} rows due to NaN score values.")

    bucket_no = bucket_no[valid_mask].astype(int)
    bucket_force = bucket_force[valid_mask].astype(int)
    perceived_useful = perceived_useful[valid_mask]

    # 2 = helps, 1 = no change, 0 = hurts
    actual_utility = np.where(bucket_force > bucket_no, 2, 0)

    # ---------- build matrix ----------
    row_labels = ["Positive", "Negative \n/ Neutral"]
    row_vals = [2, 0]
    col_labels = ["Perceived Useful\n(search_called=1)", "Perceived Not Useful\n(search_called=0)"]
    col_vals = [1, 0]

    cm_data = {}
    for col_val, col_lbl in zip(col_vals, col_labels):
        cm_data[col_lbl] = [
            int(((actual_utility == rv) & (perceived_useful == col_val)).sum())
            for rv in row_vals
        ]

    cm_counts = pd.DataFrame(cm_data, index=row_labels)

    row_totals = cm_counts.sum(axis=0)
    cm_pct = cm_counts.div(row_totals.replace(0, np.nan), axis=1) * 100

    # ---------- summary stats ----------
    helps_useful = cm_counts.loc["Positive", "Perceived Useful\n(search_called=1)"]
    helps_not_useful = cm_counts.loc["Positive", "Perceived Not Useful\n(search_called=0)"]

    total_pred_useful = cm_counts["Perceived Useful\n(search_called=1)"].sum()
    total_actual_helps = cm_counts.loc["Positive"].sum()
    total = cm_counts.values.sum()

    precision_for_help = helps_useful / total_pred_useful if total_pred_useful > 0 else float("nan")
    recall_for_help = helps_useful / total_actual_helps if total_actual_helps > 0 else float("nan")
    f1_for_help = (
        2 * precision_for_help * recall_for_help / (precision_for_help + recall_for_help)
        if pd.notna(precision_for_help) and pd.notna(recall_for_help) and (precision_for_help + recall_for_help) > 0
        else float("nan")
    )

    directional_correct = (
        cm_counts.loc["Positive", "Perceived Useful\n(search_called=1)"] +
        cm_counts.loc["Negative \n/ Neutral", "Perceived Not Useful\n(search_called=0)"] +
        cm_counts.loc["Negative \n/ Neutral", "Perceived Not Useful\n(search_called=0)"]
    )
    directional_Performance = directional_correct / total if total > 0 else float("nan")

    print(f"\n  ── Perceived Utility vs Actual Utility ──")
    print(f"  {'':18s}  {'Perceived Useful':>18s}  {'Perceived Not Useful':>24s}")
    print(f"  {'-' * 70}")
    for lbl in row_labels:
        row_str = f"  {lbl:18s}"
        for c in cm_counts.columns:
            row_str += f"  {int(cm_counts.loc[lbl, c]):>10,} ({cm_pct.loc[lbl, c]:5.0f}%)"
        print(row_str)

    print(f"\n  Precision (predicted useful that truly help) : {precision_for_help:.4f}")
    print(f"  Recall    (actual helps captured)            : {recall_for_help:.4f}")
    print(f"  F1 score                                     : {f1_for_help:.4f}")
    print(f"  Directional Performance                         : {directional_Performance:.4f}")

    stats = {
        "precision_help": round(precision_for_help, 6) if pd.notna(precision_for_help) else np.nan,
        "recall_help": round(recall_for_help, 6) if pd.notna(recall_for_help) else np.nan,
        "f1_help": round(f1_for_help, 6) if pd.notna(f1_for_help) else np.nan,
        "directional_Performance": round(directional_Performance, 6) if pd.notna(directional_Performance) else np.nan,
        "actual_helps_pred_useful": int(helps_useful),
        "actual_helps_pred_not_useful": int(helps_not_useful),
        "pred_useful_total": int(total_pred_useful),
        "actual_helps_total": int(total_actual_helps),
        "n": int(total),
        "threshold_low": float(t_low),
        "threshold_high": float(t_high),
    }

    return cm_counts, cm_pct, stats


def compute_perceived_need_vs_utility_matrix(
    perceived_need_df: pd.DataFrame,
    auto_search_df: pd.DataFrame,
):
    """
    2×2 matrix:
        Perceived Need (rows) vs Perceived Utility (cols)

    Perceived need:
        yes_no_decision == "yes" / True / 1  -> 1
        otherwise                            -> 0

    Perceived utility:
        search_called == "yes" / True / 1    -> 1
        otherwise                            -> 0

    Matrix layout:
                                  Perceived Utility=1   Perceived Utility=0
      Perceived Need=1                 Need+Useful          Need+NotUseful
      Perceived Need=0                 NoNeed+Useful        NoNeed+NotUseful

    Returns
    -------
    cm_counts : pd.DataFrame   raw counts
    cm_pct    : pd.DataFrame   row-normalized percentages
    stats     : dict           simple agreement stats
    """

    # ---------- validate inputs ----------
    need_col = "yes_no_decision"
    util_col = "search_called"

    perceived_need_df.columns = perceived_need_df.columns.str.strip()
    auto_search_df.columns = auto_search_df.columns.str.strip()

    if need_col not in perceived_need_df.columns:
        raise ValueError(
            f"Column '{need_col}' not found in perceived_need_df. "
            f"Available: {list(perceived_need_df.columns)}"
        )
    if util_col not in auto_search_df.columns:
        raise ValueError(
            f"Column '{util_col}' not found in auto_search_df. "
            f"Available: {list(auto_search_df.columns)}"
        )

    # ---------- align lengths ----------
    min_len = min(len(perceived_need_df), len(auto_search_df))
    if len(perceived_need_df) != len(auto_search_df):
        print(
            f"  WARNING: length mismatch "
            f"(perceived_need={len(perceived_need_df)}, auto_search={len(auto_search_df)}). "
            f"Using first {min_len} rows."
        )

    raw_need = perceived_need_df[need_col].iloc[:min_len]
    raw_util = auto_search_df[util_col].iloc[:min_len]

    # ---------- normalize perceived need ----------
    if pd.api.types.is_bool_dtype(raw_need):
        perceived_need = raw_need.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw_need):
        perceived_need = (raw_need != 0).astype(int).values
    else:
        perceived_need = (
            raw_need.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        ).astype(int).values

    # ---------- normalize perceived utility ----------
    if pd.api.types.is_bool_dtype(raw_util):
        perceived_utility = raw_util.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw_util):
        perceived_utility = (raw_util != 0).astype(int).values
    else:
        perceived_utility = (
            raw_util.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        ).astype(int).values

    # ---------- build matrix ----------
    row_labels = ["Need", "No Need"]
    row_vals = [1, 0]
    col_labels = ["Useful", "Not Useful"]
    col_vals = [1, 0]

    cm_data = {}
    for col_val, col_lbl in zip(col_vals, col_labels):
        cm_data[col_lbl] = [
            int(((perceived_need == rv) & (perceived_utility == col_val)).sum())
            for rv in row_vals
        ]

    cm_counts = pd.DataFrame(cm_data, index=row_labels)

    # row-normalized percentages
    row_totals = cm_counts.sum(axis=0)
    cm_pct = cm_counts.div(row_totals.replace(1, np.nan), axis=0) * 100

    # ---------- simple agreement stats ----------
    both_yes = cm_counts.loc["Need", "Useful"]
    need_not_useful = cm_counts.loc["Need", "Not Useful"]
    no_need_useful = cm_counts.loc["No Need", "Useful"]
    both_no = cm_counts.loc["No Need", "Not Useful"]

    total = cm_counts.values.sum()
    agreement = (both_yes + both_no) / total if total > 0 else float("nan")

    # utility given need
    p_useful_given_need = (
        both_yes / (both_yes + need_not_useful)
        if (both_yes + need_not_useful) > 0 else float("nan")
    )

    # need given utility
    p_need_given_useful = (
        both_yes / (both_yes + no_need_useful)
        if (both_yes + no_need_useful) > 0 else float("nan")
    )

    print(f"\n  ── Perceived Need vs Perceived Utility ──")
    print(f"  {'':18s}  {'Useful':>12s}  {'Not Useful':>14s}")
    print(f"  {'-' * 50}")
    for lbl in row_labels:
        row_str = f"  {lbl:18s}"
        for c in cm_counts.columns:
            row_str += f"  {int(cm_counts.loc[lbl, c]):>8,} ({cm_pct.loc[lbl, c]:5.0f}%)"
        print(row_str)

    print(f"\n  Agreement                           : {agreement:.4f}")
    print(f"  P(Useful | Need)                   : {p_useful_given_need:.4f}")
    print(f"  P(Need | Useful)                   : {p_need_given_useful:.4f}")

    stats = {
        "agreement": round(agreement, 6) if pd.notna(agreement) else np.nan,
        "p_useful_given_need": round(p_useful_given_need, 6) if pd.notna(p_useful_given_need) else np.nan,
        "p_need_given_useful": round(p_need_given_useful, 6) if pd.notna(p_need_given_useful) else np.nan,
        "both_yes": int(both_yes),
        "need_not_useful": int(need_not_useful),
        "no_need_useful": int(no_need_useful),
        "both_no": int(both_no),
        "n": int(total),
    }

    return cm_counts, cm_pct, stats

def _round_to_100_columnwise(pct_matrix):
    """
    Round percentages so each column sums exactly to 100.
    Uses largest remainder method.
    """
    pct = pct_matrix.copy()
    rounded = np.zeros_like(pct)

    for j in range(pct.shape[1]):
        col = pct[:, j]
        if np.any(np.isnan(col)):
            # empty column — leave as zeros
            rounded[:, j] = 0
            continue

        floored = np.floor(col)
        remainder = 100 - int(floored.sum())

        fractions = col - floored
        order = np.argsort(fractions)[::-1]

        floored = floored.astype(int)
        for i in order[:max(0, remainder)]:
            floored[i] += 1

        rounded[:, j] = floored

    return rounded

def plot_confusion_matrix(
    cm_counts: pd.DataFrame,
    cm_pct: pd.DataFrame,
    stats: dict,
    output_path: str,
    model_name: str,
    COLS: list = ["Need", "No Need"],
    ROWS: list = ["Need", "No Need"],
    x_label: str = "True",
    y_label: str = "Actual"
):
    """
    Plot a 2×2 confusion matrix in the same style as predictor.py.

    Style changes vs the original version:
    - Uses a blue heatmap ("Blues")
    - Uses simple count + row percentage annotations
    - Puts metrics in the title
    - Removes semantic cell borders, footer stats, and custom legend
    """
    cm = cm_counts.values.astype(int)
    # pct = cm_pct.values.astype(float)
    # Handle the round up error:

    # Compute column percentages
    col_totals = cm.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_exact = np.where(col_totals == 0, 0.0, (cm / col_totals) * 100)

    # Round while keeping each column = 100
    pct = _round_to_100_columnwise(pct_exact)

    # Build annotation strings: count only
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}"

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        xticklabels=COLS,
        yticklabels=ROWS,
        ax=ax,
        annot_kws={"size": 48, "weight": "bold"},
        cbar=False,
        vmin=0,
        vmax=MAX_LEN
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(x_label, fontsize=30, fontweight="bold", labelpad=16)
    ax.set_ylabel(y_label, fontsize=30, fontweight="bold", labelpad=16)
    ax.tick_params(axis='both', labelsize=30)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Perceived need matrix plot saved -> {output_path}")

def run_single(model_name: str):
    """
    Run the full pipeline for one model.

    Returns a dict with all per-model artefacts needed by the combined
    affordability plot.
    """
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    no_search_path      = f"{BASE_DIR}/vllm_{model_name}_no_search_summary{CSV_SUFFIX}.csv"
    auto_search_path    = f"{BASE_DIR}/vllm_{model_name}_with_search_summary{CSV_SUFFIX}.csv"
    force_search_path   = f"{BASE_DIR}/vllm_{model_name}_force_search_summary{CSV_SUFFIX}.csv"
    perceived_need_path = (f"{PERCEIVED_NEED_DIR}/"
                           f"vllm_{model_name}_with_search_summary{PERCEIVED_NEED_CSV_SUFFIX}.csv")

    no_search_df      = load_csv(no_search_path)
    auto_search_df    = load_csv(auto_search_path)

    force_search_df   = load_csv(force_search_path)
    perceived_need_df = load_csv(perceived_need_path)

    no_search_df["score"] = no_search_df["score"].fillna(0)
    auto_search_df["score"] = auto_search_df["score"].fillna(0)
    force_search_df["score"] = force_search_df["score"].fillna(0)

    # no_search_df['score'] = pd.cut(no_search_df['score'], bins=SCORE_BINS, labels=[0.0, 0.5, 1.0], right=False).astype(float)
    # auto_search_df['score'] = pd.cut(force_search_df['score'], bins=SCORE_BINS, labels=[0.0, 0.5, 1.0], right=False).astype(float)
    # force_search_df['score'] = pd.cut(force_search_df['score'], bins=SCORE_BINS, labels=[0.0, 0.5, 1.0], right=False).astype(float)

    print("\n[0] Accuracy summary ...")
    print(f"  No-search    accuracy : {no_search_df['score'].mean():.4f}  (n={len(no_search_df):,})")
    print(f"  Force-search accuracy : {force_search_df['score'].mean():.4f}  (n={len(force_search_df):,})")
    print(f"  With-search  accuracy : {auto_search_df['score'].mean():.4f}  (n={len(auto_search_df):,})")

    print("\n[1] Computing actual need ...")
    need_df = compute_need(no_search_df)

    print("\n[2] Computing actual utility ...")
    utility_df = compute_utility(force_search_df, no_search_df)
    utility_df["actual_need"] = need_df["actual_need"].values[:len(utility_df)]

    print("\n[3] Computing 3x3 bucket confusion matrix ...")
    cm_counts, cm_pct, cm_summary = compute_bucket_confusion_matrix(utility_df)

    print("\n[5] Computing perceived need matrix ...")
    pn_counts, pn_pct, pn_stats = compute_perceived_need_matrix(
        no_search_df, perceived_need_df
    )

    print("\n[6] Computing perceived utility matrix ...")
    pu_counts, pu_pct, pu_stats = compute_perceived_utility_matrix(
        no_search_df, force_search_df, auto_search_df
    )

    print("\n[7] Computing perceived need vs perceived utility matrix ...")
    pnu_counts, pnu_pct, pnu_stats = compute_perceived_need_vs_utility_matrix(
        perceived_need_df, auto_search_df
    )

    print("\n[4] Computing rank list & affordability ...")
    rank_df   = compute_rank_list(utility_df)
    afford_df = compute_affordability(rank_df)


    # ── Save per-model outputs ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if TASK == 'bfcl':
        prefix = f"{OUTPUT_DIR}/{TASK}_{model_name}"
    elif TASK == 'invivo':
        prefix = f"{OUTPUT_DIR}/{TASK}_{model_name}"
    else: 
        prefix =  f"{OUTPUT_DIR}/{model_name}"

    # Per-model plots (one figure each)
    plot_bucket_confusion_matrix(
        cm_counts, cm_pct,
        f"{prefix}_bucket_confusion_matrix.pdf", model_name
    )

    plot_confusion_matrix(
        pn_counts, pn_pct, pn_stats,
        f"{prefix}_perceived_need_matrix.pdf", model_name,
        COLS = ["Need", "No Need"],
        ROWS =  ["Need", "No Need"],
        x_label = "Perceived",
        y_label = "True"
    )

    plot_confusion_matrix(
        pu_counts, pu_pct, pu_stats,
        f"{prefix}_perceived_utility_matrix.pdf", model_name,
        COLS =   ["Call", "No Call"],
        ROWS =  ["Positive", "Other"],
        x_label= "Tool Calling",
        y_label = "True Utility"
    )

    plot_confusion_matrix(
        pnu_counts.T, pnu_pct.T, pnu_stats,
        f"{prefix}_perceived_need_vs_utility_matrix.pdf", model_name,
        COLS =  ["Need", "No Need"],
        ROWS = ["Call", "No Call"],
        x_label = "Perceived",
        y_label= "Tool Calling"
    )

    # ── v1 / v2 perceived-need plots ─────────────────────────────────────────
    for ver_label, ver_dir in [("v1", PERCEIVED_NEED_DIR_V1), ("v2", PERCEIVED_NEED_DIR_V2)]:
        if ver_dir is None:
            continue
        ver_path = f"{ver_dir}/vllm_{model_name}_with_search_summary{PERCEIVED_NEED_CSV_SUFFIX}.csv"
        if not Path(ver_path).exists():
            print(f"\n  [{ver_label}] SKIP — file not found: {ver_path}")
            continue
        print(f"\n[{ver_label}] Computing perceived need matrix ...")
        ver_df = load_csv(ver_path)
        pn_v_counts, pn_v_pct, pn_v_stats = compute_perceived_need_matrix(no_search_df, ver_df)
        plot_confusion_matrix(
            pn_v_counts, pn_v_pct, pn_v_stats,
            f"{prefix}_perceived_need_matrix_{ver_label}.pdf", model_name,
            COLS = ["Need", "No Need"],
            ROWS =  ["Need", "No Need"],
            x_label = "Perceived",
            y_label = "True"
        )
        pnu_v_counts, pnu_v_pct, pnu_v_stats = compute_perceived_need_vs_utility_matrix(
            ver_df, auto_search_df
        )
        plot_confusion_matrix(
            pnu_v_counts.T, pnu_v_pct.T, pnu_v_stats,
            f"{prefix}_perceived_need_vs_utility_matrix_{ver_label}.pdf", model_name,
            COLS =  ["Need", "No Need"],
            ROWS = ["Call", "No Call"],
            x_label = "Perceived",
            y_label= "Tool Calling"
        )

    print(f"\nAll outputs for {model_name} written to {OUTPUT_DIR}")

    return dict(
        utility_df=utility_df,
        rank_df=rank_df,
        afford_df=afford_df,
        no_search_df=no_search_df,
        auto_search_df=auto_search_df,
        cm_counts=cm_counts,
        cm_pct=cm_pct,
    )


def run(model_names: list, budget_aware_suffix: str = ""):
    """
    Run the full pipeline for every model in *model_names*.

    Per-model figures (bucket confusion matrix, perceived need matrix) are saved
    individually.  The affordability curve is saved once as a combined figure
    with all models overlaid in different colours.
    """
    afford_dfs    = {}   # model_name -> afford_df  (for combined affordability plot)
    no_search_dfs = {}   # model_name -> no_search_df
    ndcg_data     = {}   # model_name -> {"utility_df": ..., "auto_search_df": ...}

    for model_name in model_names:
        results = run_single(model_name)
        afford_dfs[model_name]    = results["afford_df"]
        no_search_dfs[model_name] = results["no_search_df"]
        ndcg_data[model_name]     = {
            "utility_df": results["utility_df"],
            "total_n":    len(results["no_search_df"]),
        }

    # ── Combined affordability plot (all models, no legend) ───────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix_tag = budget_aware_suffix.lstrip("-").replace("-", "_")  # e.g. "v2" or ""
    file_suffix = f"_{suffix_tag}" if suffix_tag else ""
    if TASK == 'bfcl' or TASK == 'invivo':
        a_output_dir = f"{OUTPUT_DIR}/{TASK}_total_affordability{file_suffix}.pdf"
    else:
        a_output_dir = f"{OUTPUT_DIR}/total_affordability{file_suffix}.pdf"

    actual_calls_data, no_cost_desc_calls = plot_affordability(
        afford_dfs=afford_dfs,
        output_path=a_output_dir,
        no_search_dfs=no_search_dfs,
        results_base_dir=BASE_ALL,
        budget_aware_suffix=budget_aware_suffix,
    )
    if TASK == 'bfcl' or TASK == 'invivo':
        ac_output_dir = f"{OUTPUT_DIR}/{TASK}_actual_tool_calls{file_suffix}.pdf"
    else:
        ac_output_dir = f"{OUTPUT_DIR}/actual_tool_calls{file_suffix}.pdf"
    plot_actual_tool_calls(
        actual_calls_data=actual_calls_data,
        output_path=ac_output_dir,
        no_cost_desc_calls=no_cost_desc_calls,
    )

    # ── NDCG rank-correlation figure (budget-aware original) ─────────────────
    if TASK == 'bfcl' or TASK == 'invivo':
        ndcg_output_dir    = f"{OUTPUT_DIR}/{TASK}_ndcg_rank_correlation{file_suffix}.pdf"
        ndcg_v2_output_dir = f"{OUTPUT_DIR}/{TASK}_ndcg_rank_correlation_v2{file_suffix}.pdf"
    else:
        ndcg_output_dir    = f"{OUTPUT_DIR}/ndcg_rank_correlation{file_suffix}.pdf"
        ndcg_v2_output_dir = f"{OUTPUT_DIR}/ndcg_rank_correlation_v2{file_suffix}.pdf"
    plot_ndcg_rank_correlation(
        model_data=ndcg_data,
        output_path=ndcg_output_dir,
        results_base_dir=BASE_ALL,
        budget_aware_suffix="",
    )

    # ── NDCG rank-correlation figure (budget-aware-v2) ────────────────────────
    import glob as _glob
    if _glob.glob(f"{BASE_ALL}/tool-cost-*-budget-aware-v2"):
        plot_ndcg_rank_correlation(
            model_data=ndcg_data,
            output_path=ndcg_v2_output_dir,
            results_base_dir=BASE_ALL,
            budget_aware_suffix="-v2",
        )

    # ── v2 budget-aware figures ────────────────────────────────────────────────
    if not budget_aware_suffix:
        import glob as _glob
        v2_dirs = _glob.glob(f"{BASE_ALL}/tool-cost-*-budget-aware-v2")
        if v2_dirs:
            print(f"\n  Generating cost-budget-awareness-v2 figures ...")
            if TASK == 'bfcl' or TASK == 'invivo':
                a_v2_path  = f"{OUTPUT_DIR}/{TASK}_actual_affordability_v2.pdf"
                ac_v2_path = f"{OUTPUT_DIR}/{TASK}_actual_tool_calls_v2.pdf"
            else:
                a_v2_path  = f"{OUTPUT_DIR}/actual_affordability_v2.pdf"
                ac_v2_path = f"{OUTPUT_DIR}/actual_tool_calls_v2.pdf"
            actual_calls_data_v2, no_cost_desc_calls_v2 = plot_affordability(
                afford_dfs=afford_dfs,
                output_path=a_v2_path,
                no_search_dfs=no_search_dfs,
                results_base_dir=BASE_ALL,
                budget_aware_suffix="-v2",
            )
            plot_actual_tool_calls(
                actual_calls_data=actual_calls_data_v2,
                output_path=ac_v2_path,
                no_cost_desc_calls=no_cost_desc_calls_v2,
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normative framework analysis (need / utility / affordability)"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=list(TASK_CONFIGS) + ["all"],
        help="Which task to analyse (entity / bfcl / invivo / all).  Default: all.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=[
    # "google_gemma-3-27b-it",
    "openai_gpt-oss-120b",
    # "Qwen_Qwen3-30B-A3B",
    # "Qwen_Qwen3-30B-A3B-Instruct-2507",
    # "mistralai_Mistral-Small-3.1-24B-Instruct-2503",
    # "meta-llama_Llama-3.2-3B-Instruct",
],
        help="One or more model name suffixes used in filenames (e.g. gpt-oss-120b gpt-4o)"
    )
    parser.add_argument(
        "--budget", type=float, default=BUDGET,
        help=f"Total budget in USD (default {BUDGET})"
    )
    parser.add_argument(
        "--need-threshold", type=float, default=None,
        help="Score threshold above which model is self-sufficient (overrides per-task default)."
    )
    parser.add_argument(
        "--budget-aware-suffix", type=str, default="",
        help=(
            "Suffix appended to the 'budget-aware' directory name when looking up "
            "experimental tool-call CSV files.  E.g. '-v2' will read from "
            "'tool-cost-<N>-budget-aware-v2/' instead of 'tool-cost-<N>-budget-aware/'. "
            "The suffix is also appended to the output PDF filenames so existing figures "
            "are not overwritten.  Default: '' (original budget-aware directories)."
        ),
    )
    args = parser.parse_args()

    tasks_to_run = list(TASK_CONFIGS) if args.task == "all" else [args.task]
    for _task in tasks_to_run:
        _apply_task_config(_task)
        BUDGET = args.budget
        if args.need_threshold is not None:
            NEED_THRESHOLD = args.need_threshold
        print(f"  Need threshold: score > {NEED_THRESHOLD} -> no need (actual_need=0)")
        print(f"  Models: {args.models}")
        run(args.models, budget_aware_suffix=args.budget_aware_suffix)