"""
Venn diagram: True Positive Utility / Perceived Need / Perceived Utility.

One subplot per model (2×3 grid) using real per-query set membership.
Plus a separate "Oracle" reference panel.

Sets (per query):
  True Positive Utility: bucket_force > bucket_no  (upper triangle of bucket
                         confusion matrix, SCORE_BINS=[-inf, 0.1, 0.9, inf])
  Perceived Need       : yes_no_decision == "yes"  (from perceived-need CSV)
  Perceived Utility    : search_called == 1        (from with_search CSV)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3, venn3_circles

# ── Task configs (mirrors new_analysis.py) ────────────────────────────────────
TASK_CONFIGS = {
    "entity": dict(
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/normative",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
    "bfcl": dict(
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main-perceived-need",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl",
        score_bins                = [-0.01, 0.99, 1.001],
        score_labels              = ["Incorrect", "Correct"],
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
    "invivo": dict(
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main-perceived-need",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/invivo",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
    "perplexity": dict(
        base_dir                  = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main",
        perceived_need_dir        = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity/temp=0/main-perceived-need",
        output_dir                = "/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity_",
        score_bins                = [-np.inf, 0.1, 0.9, np.inf],
        score_labels              = ["Low", "Mid", "High"],
        csv_suffix                = "",
        perceived_need_csv_suffix = "",
    ),
}

# ── Active config (set by _run_task() via --task) ─────────────────────────────
BASE_DIR                  = TASK_CONFIGS["entity"]["base_dir"]
PERCEIVED_NEED_DIR        = TASK_CONFIGS["entity"]["perceived_need_dir"]
OUTPUT_DIR                = TASK_CONFIGS["entity"]["output_dir"]
SCORE_BINS                = TASK_CONFIGS["entity"]["score_bins"]
SCORE_LABELS              = TASK_CONFIGS["entity"]["score_labels"]
CSV_SUFFIX                = TASK_CONFIGS["entity"]["csv_suffix"]
PERCEIVED_NEED_CSV_SUFFIX = TASK_CONFIGS["entity"]["perceived_need_csv_suffix"]

MODEL_IN_LEGEND = {
    # "google_gemma-3-27b-it":                       "Gemma3-27B-IT",
    "openai_gpt-oss-120b":                         "GPT-OSS-120B",
    # "Qwen_Qwen3-30B-A3B":                          "Qwen3-30B-A3B",
    # "Qwen_Qwen3-30B-A3B-Instruct-2507":            "Qwen-3-30B-IT",
    # "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral3.1-24B-IT",
    # "meta-llama_Llama-3.2-3B-Instruct":            "Llama3.2-3B-IT",
}

# ── Colours ───────────────────────────────────────────────────────────────────
C_TRUE = "#4C72B0"   # blue   – True Positive Utility
C_NEED = "#DD8452"   # orange – Perceived Need
C_UTIL = "#55A868"   # green  – Perceived Utility


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _parse_yes_no(series: pd.Series) -> np.ndarray:
    """Return boolean array: True where model perceived need (yes_no_decision='yes')."""
    if pd.api.types.is_bool_dtype(series):
        return series.values
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool).values
    return (series.str.strip().str.lower() == "yes").values


def _to_bucket(scores: np.ndarray) -> np.ndarray:
    """Map scores to integer bucket indices (0=Low, 1=Mid, 2=High)."""
    labels = pd.cut(scores, bins=SCORE_BINS, labels=False,
                    right=False, include_lowest=True)
    return labels.astype(int)


def compute_sets(model_name: str):
    """
    Returns three Python sets of row-indices (0-based) — one per concept.

    True Positive Utility: bucket_force > bucket_no  (upper triangle of
                           bucket confusion matrix)
    Perceived Need       : yes_no_decision == 'yes'
    Perceived Utility    : search_called == 1
    """
    no_search_df    = _load(f"{BASE_DIR}/vllm_{model_name}_no_search_summary{CSV_SUFFIX}.csv")
    force_search_df = _load(f"{BASE_DIR}/vllm_{model_name}_force_search_summary{CSV_SUFFIX}.csv")
    auto_search_df  = _load(f"{BASE_DIR}/vllm_{model_name}_with_search_summary{CSV_SUFFIX}.csv")
    perceived_need_df = _load(
        f"{PERCEIVED_NEED_DIR}/vllm_{model_name}_with_search_summary{PERCEIVED_NEED_CSV_SUFFIX}.csv"
    )

    n = min(len(no_search_df), len(force_search_df),
            len(auto_search_df), len(perceived_need_df))

    no_scores    = no_search_df["score"].iloc[:n].fillna(0).values.astype(float)
    force_scores = force_search_df["score"].iloc[:n].fillna(0).values.astype(float)

    bucket_no    = _to_bucket(no_scores)
    bucket_force = _to_bucket(force_scores)

    true_util_mask  = bucket_force > bucket_no   # upper triangle = improved bucket
    perc_need_mask  = _parse_yes_no(perceived_need_df["yes_no_decision"].iloc[:n])
    perc_util_mask  = auto_search_df["search_called"].iloc[:n].astype(bool).values

    indices = np.arange(n)
    return (
        set(indices[true_util_mask]),
        set(indices[perc_need_mask]),
        set(indices[perc_util_mask]),
        n,
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_venn(v, c):
    """Apply fill colours and edge styling to a venn3 object."""
    id_color = {
        "100": C_TRUE,
        "010": C_NEED,
        "001": C_UTIL,
        "110": _blend(C_TRUE, C_NEED),
        "101": _blend(C_TRUE, C_UTIL),
        "011": _blend(C_NEED, C_UTIL),
        "111": _blend(_blend(C_TRUE, C_NEED), C_UTIL),
    }
    for patch_id, color in id_color.items():
        patch = v.get_patch_by_id(patch_id)
        if patch:
            patch.set_facecolor(color)
            patch.set_alpha(0.60)
            patch.set_edgecolor("none")

    for circle in c:
        circle.set_linewidth(2.5)
        circle.set_edgecolor("#222222")
        circle.set_fill(False)

    # Count labels — large, white with dark outline for visibility
    for patch_id in ["100", "010", "001", "110", "101", "011", "111"]:
        lbl = v.get_label_by_id(patch_id)
        if lbl:
            lbl.set_fontsize(30)
            lbl.set_fontweight("bold")
            lbl.set_color("black")

    # Hide set labels (shown in legend instead)
    for lbl in v.set_labels:
        if lbl:
            lbl.set_text("")


def _blend(hex1: str, hex2: str) -> str:
    """Average two hex colours."""
    def h2r(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    avg = (h2r(hex1) + h2r(hex2)) / 2
    return "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in avg))


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _draw_venn_panel(ax, true_set, need_set, util_set, title,
                     hide_counts=False, show_title=True):
    """Draw a styled venn3 panel onto ax."""
    # Compute actual region counts from sets
    only_A   = len(true_set - need_set - util_set)
    only_B   = len(need_set - true_set - util_set)
    only_C   = len(util_set - true_set - need_set)
    AB       = len((true_set & need_set) - util_set)
    AC       = len((true_set & util_set) - need_set)
    BC       = len((need_set & util_set) - true_set)
    ABC      = len(true_set & need_set & util_set)
    actual_counts = {"100": only_A, "010": only_B, "001": only_C,
                     "110": AB,     "101": AC,      "011": BC,     "111": ABC}

    # Draw with equal-sized circles
    v = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=("True\nUtility", "Perceived\nNeed", "Perceived\nUtility"),
        ax=ax,
    )
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax=ax)

    # Restore actual counts as labels
    for pid, cnt in actual_counts.items():
        lbl = v.get_label_by_id(pid)
        if lbl:
            lbl.set_text(str(cnt))
    _style_venn(v, c)

    if hide_counts:
        for pid in ["100", "010", "001", "110", "101", "011", "111"]:
            lbl = v.get_label_by_id(pid)
            if lbl:
                lbl.set_text("")

    if show_title:
        ax.set_title(title, fontsize=28, fontweight="bold", pad=12)


def draw_normative(ax):
    """Oracle panel: ideally nested sets rendered via venn3."""
    # Construct perfectly nested index sets:
    #   Perceived Utility ⊆ Perceived Need ⊆ True Utility
    n_true, n_need, n_util = 500, 300, 100
    true_set = set(range(n_true))
    need_set = set(range(n_need))
    util_set = set(range(n_util))
    _draw_venn_panel(ax, true_set, need_set, util_set,
                     "Oracle", hide_counts=True)


def draw_model_panel(ax, model_name, show_model_name=True, show_title=True):
    """Data-driven venn3 panel for one model."""
    true_set, need_set, util_set, _ = compute_sets(model_name)
    display = MODEL_IN_LEGEND[model_name]
    title = f"Descriptive Lens\n{display}" if show_model_name else "Descriptive Lens"
    _draw_venn_panel(ax, true_set, need_set, util_set, title, show_title=show_title)


# ── Shared legend helper ──────────────────────────────────────────────────────

def _add_legend(fig):
    legend_patches = [
        mpatches.Patch(facecolor=C_TRUE, alpha=0.75, label="True Positive Utility"),
        mpatches.Patch(facecolor=C_NEED, alpha=0.75, label="Perceived Need"),
        mpatches.Patch(facecolor=C_UTIL, alpha=0.75, label="Perceived Utility"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=1,
               fontsize=30, frameon=False, bbox_to_anchor=(1.50, 0.35),
               handlelength=2.0, handleheight=1.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_path: str = None, task: str = "entity"):
    """Run venn diagram for one task (or call _run_task directly for 'all')."""
    if task == "all":
        for t in TASK_CONFIGS:
            _run_task(t, output_path=None)
        return
    _run_task(task, output_path=output_path)


def _run_task(task: str, output_path: str = None):
    global BASE_DIR, PERCEIVED_NEED_DIR, OUTPUT_DIR, SCORE_BINS, SCORE_LABELS
    global CSV_SUFFIX, PERCEIVED_NEED_CSV_SUFFIX

    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS)}")
    cfg = TASK_CONFIGS[task]
    BASE_DIR                  = cfg["base_dir"]
    PERCEIVED_NEED_DIR        = cfg["perceived_need_dir"]
    OUTPUT_DIR                = cfg["output_dir"]
    SCORE_BINS                = cfg["score_bins"]
    SCORE_LABELS              = cfg["score_labels"]
    CSV_SUFFIX                = cfg["csv_suffix"]
    PERCEIVED_NEED_CSV_SUFFIX = cfg["perceived_need_csv_suffix"]

    print(f"\n{'#'*60}")
    print(f"# venn_diagram  task={task}")
    print(f"#   BASE_DIR   = {BASE_DIR}")
    print(f"#   OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"{'#'*60}")

    if output_path is None:
        task_tag = f"_{task}" if task != "entity" else ""
        output_path = f"{OUTPUT_DIR}/venn_diagram{task_tag}.pdf"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(MODEL_IN_LEGEND.keys())

    # ── Combined grid figure ──────────────────────────────────────────────────
    # 2 rows × 3 cols: one model per slot
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    fig.patch.set_facecolor("white")

    ax_flat = iter(axes.flat)

    for model_name, ax in zip(models, ax_flat):
        display = MODEL_IN_LEGEND[model_name]
        try:
            draw_model_panel(ax, model_name)
        except FileNotFoundError as e:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{display}\n(data not found)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            print(f"  WARNING: {e}")

    for ax in ax_flat:   # hide unused slots
        ax.axis("off")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Combined venn saved → {output_path}")

    # ── Per-model PDFs ────────────────────────────────────────────
    base, ext = os.path.splitext(output_path)
    for model_name in models:
        display = MODEL_IN_LEGEND[model_name]
        per_path = f"{base}_{display.replace(' ', '_')}{ext}"
        fig2, ax_model = plt.subplots(1, 1, figsize=(5, 5))
        fig2.patch.set_facecolor("white")

        try:
            draw_model_panel(ax_model, model_name, show_model_name=True, show_title=False)
        except FileNotFoundError as e:
            ax_model.axis("off")
            ax_model.text(0.5, 0.5, f"{display}\n(data not found)",
                          ha="center", va="center", transform=ax_model.transAxes,
                          fontsize=11, color="gray")
            print(f"  WARNING: {e}")

        _add_legend(fig2)
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        fig2.savefig(per_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"  Per-model venn saved → {per_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="all",
        choices=list(TASK_CONFIGS) + ["all"],
        help="Which task to analyse (entity / bfcl / invivo / all).  Default: all.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path (default: <output_dir>/venn_diagram.pdf)",
    )
    args = parser.parse_args()
    main(args.output, task=args.task)
