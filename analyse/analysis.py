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
    "google_gemma-3-27b-it": "Gemma3-27B-IT",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "Qwen_Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen-3-30B-IT",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral3.1-24B-IT",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama3.2-3B-IT",
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


# ── Config ────────────────────────────────────────────────────────────────────
TASK = 'entity'
BASE_ALL = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0"
BASE_DIR   = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main"
OUTPUT_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/normative"
PERCEIVED_NEED_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need"

BUDGET     = 10000        # dollars
NEED_THRESHOLD     = 0.9   # score > this  → model does NOT need help
UTILITY_POS_THRESH = 0.0    # delta > 0     → utility = +1
UTILITY_NEG_THRESH = -0.05  # delta < -0.05 → utility = -1; in between → 0

# Score buckets — shared for both no-search and force-search
SCORE_BINS = [-np.inf, 0.1, 0.9, np.inf]  # 1.001 so score==1.0 falls in High
SCORE_LABELS = ["Low", "Mid", "High"]
MAX_LEN = 500

# Perceived-need analysis
ACTUAL_NEED_SCORE_THRESH = 0.9    # no_search_score < this → actual_need = 1

# ── Helpers ───────────────────────────────────────────────────────────────────

def score_to_bucket(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        series,
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        right=False,
        include_lowest=True,
    )


def load_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from {path.name}")
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
    value_fontsize = 30 if ncols == 2 else 25

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
    ax.set_xlabel("Accuracy with tool", fontsize=25, fontweight="bold")
    ax.set_ylabel("Accuracy without tool", fontsize=25, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(score_labels, rotation=0, ha="right", fontsize=25, fontweight="bold")
    ax.set_yticklabels(score_labels, rotation=0, ha="right", fontsize=25, fontweight="bold")

    # ---- legend ----
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker='o', color='w', label='No Change',
               markerfacecolor='#444444', markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Tool Helps',
               markerfacecolor='#2ECC71', markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Tool Hurts',
               markerfacecolor='#E74C3C', markersize=14),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(legend_items),
        frameon=True,
        borderaxespad=0,
        fontsize=25
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
            x_bracket + 0.15,
            y_text,
            "Actual Need",
            rotation=90,
            va="center",
            ha="center",
            fontsize=25, 
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
    Y-axis : net accuracy gain over baseline

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
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BUDGET_TOTAL = 10_000
    LEFT_X = BUDGET_TOTAL
    LEFT_X_PAD = BUDGET_TOTAL * 1.15
    RIGHT_X = 1.0  # proxy for cost=0 on log axis
    cost_setups = [10000, 1000, 100, 10, 0]

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

        RIGHT_X = 1.0                 # proxy for cost = 0 on log axis
        LEFT_X = BUDGET_TOTAL * 1.15  # extend slightly beyond max budget

        # Keep only meaningful k values for interior curve
        valid = (x_k > 0) & (x_k < total_n)

        x_plot = BUDGET_TOTAL / x_k[valid]
        y_plot = y[valid]

        # Sort by x ascending for plotting
        if len(x_plot) > 0:
            order = np.argsort(x_plot)
            x_plot = x_plot[order]
            y_plot = y_plot[order]

        # Determine right-edge y value
        search_all = (x_k >= total_n)
        if np.any(search_all):
            # If there is an explicit "search all" point, use that at RIGHT_X
            right_y = y[search_all][-1]
        elif len(y_plot) > 0:
            # Otherwise extend using the smallest-cost visible point
            right_y = y_plot[0]
        else:
            continue

        # Determine left-edge y value
        left_y = y_plot[-1] if len(y_plot) > 0 else right_y

        # Explicitly extend both ends
        x_plot = np.concatenate(([RIGHT_X], x_plot, [LEFT_X]))
        y_plot = np.concatenate(([right_y], y_plot, [left_y]))

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

        for cost in cost_setups:
            cost_tag = str(int(cost))
            csv_path = (
                f"{results_base_dir}/tool-cost-{cost_tag}-budget-aware/"
                f"vllm_{model_name}_with_search_summary.csv"
            )
            force_search_path = (
                f"{results_base_dir}/main/"
                f"vllm_{model_name}_force_search_summary.csv"
            )

            try:
                exp_df = pd.read_csv(csv_path)
                force_search_df = pd.read_csv(force_search_path)

                if "search_called" not in exp_df.columns or "score" not in exp_df.columns:
                    print(f"  WARNING: missing columns in {csv_path}, skipping.")
                    continue

                k_actual = int(exp_df["search_called"].sum())
                k_budget = total_n if cost == 0 else int(BUDGET_TOTAL / cost)

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
            plot_cost = RIGHT_X if cost == 0 else cost
            processed_points.append((plot_cost, net_gain))

        processed_points.sort(key=lambda x: x[0], reverse=True)

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

    handles, labels = ax.get_legend_handles_labels()
    point_handle = Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=9,
        linestyle="None",
    )
    handles.append(point_handle)
    # labels.append("Perceived Affordability")

    ax.set_xscale("log")
    ax.set_xlim(LEFT_X_PAD, 0.8)
    ax.set_xlabel("Tool Cost ($)", fontsize=25, fontweight="bold")
    ax.set_ylabel("Utility Gain over No-Tool", fontsize=25, fontweight="bold")
    ax.tick_params(axis="both", labelsize=22)

    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
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


# ── Perceived Need vs Actual Need (2×2 confusion matrix) ─────────────────────

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
    stats     : dict          precision, recall, F1, accuracy
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
    accuracy  = (tp + tn) / (tp + fn + fp + tn)

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
    print(f"  Accuracy                            : {accuracy:.4f}")

    stats = dict(precision=round(precision, 6), recall=round(recall, 6),
                 f1=round(f1, 6), accuracy=round(accuracy, 6),
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
    row_labels = ["Helps", "No Help"]
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
    helps_useful = cm_counts.loc["Helps", "Perceived Useful\n(search_called=1)"]
    helps_not_useful = cm_counts.loc["Helps", "Perceived Not Useful\n(search_called=0)"]

    total_pred_useful = cm_counts["Perceived Useful\n(search_called=1)"].sum()
    total_actual_helps = cm_counts.loc["Helps"].sum()
    total = cm_counts.values.sum()

    precision_for_help = helps_useful / total_pred_useful if total_pred_useful > 0 else float("nan")
    recall_for_help = helps_useful / total_actual_helps if total_actual_helps > 0 else float("nan")
    f1_for_help = (
        2 * precision_for_help * recall_for_help / (precision_for_help + recall_for_help)
        if pd.notna(precision_for_help) and pd.notna(recall_for_help) and (precision_for_help + recall_for_help) > 0
        else float("nan")
    )

    directional_correct = (
        cm_counts.loc["Helps", "Perceived Useful\n(search_called=1)"] +
        cm_counts.loc["No Help", "Perceived Not Useful\n(search_called=0)"] +
        cm_counts.loc["No Help", "Perceived Not Useful\n(search_called=0)"]
    )
    directional_accuracy = directional_correct / total if total > 0 else float("nan")

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
    print(f"  Directional accuracy                         : {directional_accuracy:.4f}")

    stats = {
        "precision_help": round(precision_for_help, 6) if pd.notna(precision_for_help) else np.nan,
        "recall_help": round(recall_for_help, 6) if pd.notna(recall_for_help) else np.nan,
        "f1_help": round(f1_for_help, 6) if pd.notna(f1_for_help) else np.nan,
        "directional_accuracy": round(directional_accuracy, 6) if pd.notna(directional_accuracy) else np.nan,
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

        floored = np.floor(col)
        remainder = 100 - int(floored.sum())

        fractions = col - floored
        order = np.argsort(fractions)[::-1]

        floored = floored.astype(int)
        for i in order[:remainder]:
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
    pct_exact = (cm / col_totals) * 100

    # Round while keeping each column = 100
    pct = _round_to_100_columnwise(pct_exact)

    # Build annotation strings like predictor.py: "count\n(xx.x%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({pct[i, j]:.0f}%)"

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
        annot_kws={"size": 25, "weight": "bold"},
        cbar=False,
        vmin=0,
        vmax=MAX_LEN
    )

    ax.set_xlabel(x_label, fontsize=25, fontweight="bold", labelpad=16)
    ax.set_ylabel(y_label, fontsize=25, fontweight="bold", labelpad=16)
    ax.tick_params(axis='both', labelsize=20)

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

    no_search_path      = f"{BASE_DIR}/vllm_{model_name}_no_search_summary.csv"
    auto_search_path    = f"{BASE_DIR}/vllm_{model_name}_with_search_summary.csv"
    force_search_path   = f"{BASE_DIR}/vllm_{model_name}_force_search_summary.csv"
    perceived_need_path = (f"{PERCEIVED_NEED_DIR}/"
                           f"vllm_{model_name}_with_search_summary.csv")

    no_search_df      = load_csv(no_search_path)
    auto_search_df    = load_csv(auto_search_path)
    force_search_df   = load_csv(force_search_path)
    perceived_need_df = load_csv(perceived_need_path)

    no_search_df["score"] = no_search_df["score"].fillna(0)
    force_search_df["score"] = force_search_df["score"].fillna(0)

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
        ROWS =  ["Help", "No Help"],
        x_label= "Tool Calling",
        y_label = "True"
    )

    plot_confusion_matrix(
        pnu_counts.T, pnu_pct.T, pnu_stats,
        f"{prefix}_perceived_need_vs_utility_matrix.pdf", model_name,
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
        cm_counts=cm_counts,
        cm_pct=cm_pct,
    )


def run(model_names: list):
    """
    Run the full pipeline for every model in *model_names*.

    Per-model figures (bucket confusion matrix, perceived need matrix) are saved
    individually.  The affordability curve is saved once as a combined figure
    with all models overlaid in different colours.
    """
    afford_dfs    = {}   # model_name -> afford_df  (for combined affordability plot)
    no_search_dfs = {}   # model_name -> no_search_df

    for model_name in model_names:
        results = run_single(model_name)
        afford_dfs[model_name]    = results["afford_df"]
        no_search_dfs[model_name] = results["no_search_df"]

    # ── Combined affordability plot (all models, no legend) ───────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if TASK == 'bfcl':
        a_output_dir = f"{OUTPUT_DIR}/{TASK}_total_affordability.pdf"
    else:
        a_output_dir = f"{OUTPUT_DIR}/total_affordability.pdf"
    plot_affordability(
        afford_dfs=afford_dfs,
        output_path=a_output_dir,
        no_search_dfs=no_search_dfs,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normative framework analysis (need / utility / affordability)"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=[
    "google_gemma-3-27b-it",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct"

],
        help="One or more model name suffixes used in filenames (e.g. gpt-oss-120b gpt-4o)"
    )
    parser.add_argument(
        "--budget", type=float, default=BUDGET,
        help=f"Total budget in USD (default {BUDGET})"
    )
    parser.add_argument(
        "--need-threshold", type=float, default=NEED_THRESHOLD,
        help=f"Score threshold above which model is self-sufficient (default {NEED_THRESHOLD})."
    )
    args = parser.parse_args()
    BUDGET = args.budget
    NEED_THRESHOLD = args.need_threshold
    print(f"  Need threshold: score > {NEED_THRESHOLD} -> no need (actual_need=0)")
    print(f"  Models: {args.models}")
    run(args.models)