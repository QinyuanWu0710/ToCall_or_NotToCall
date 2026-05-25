
"""
bar_plots.py
============
Produces grouped bar plots comparing balanced accuracy, accuracy, and macro
precision per model:

  Plot 1  – Actual Need
      Bar A : Perceived Need vs Actual Need   (from perceived-need confusion matrix)
      Bar B : Predictor 1 vs Actual Need      (best layer & classifier per model)

  Plot 2  – Actual Utility (Predictor 2)
      Bar A : Perceived Utility vs Actual Utility
      Bar B : Predictor 2 vs Actual Utility

  Plot 3  – Actual Utility (Predictor 3)
      Bar A : Perceived Utility vs Actual Utility   (same baseline as Plot 2)
      Bar B : Predictor 3 vs Actual Utility

  Plot 4  – Utility comparison (Perceived / Predictor 2 / Predictor 3)
      Bar A : Perceived Utility
      Bar B : Predictor 2 (LUE_{x,d})
      Bar C : Predictor 3 (LUE_{x})

Each plot is saved three times — once per metric:
    *_balanced_accuracy.pdf  /  *_accuracy.pdf  /  *_macro_precision.pdf

Usage
-----
python bar_plots.py [--clf LogisticRegression XGBoost MLP]
                    [--layer-selection best|average]
                    [--output-dir /path/to/output]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import precision_score, balanced_accuracy_score, accuracy_score
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

_RESULTS_ROOT = "/NS/chatgpt/work/qwu/hallucinations_detection/results"
_PREDICTOR_ROOT = "/NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor"

TASK_CONFIGS = {
    "entity": dict(
        base_dir           = f"{_RESULTS_ROOT}/entity_hallucination/temp=0/main",
        perceived_need_dir = f"{_RESULTS_ROOT}/entity_hallucination/temp=0/main-perceived-need",
        predictor_subdir   = "",          # models live at {root}/{model}/predictor_{id}
        score_bins         = [-np.inf, 0.1, 0.9, np.inf],
        csv_suffix         = "",          # no_search_summary.csv  (no extra suffix)
        actual_need_thresh = 0.9,
        output_dir         = f"{_RESULTS_ROOT}/predictor",
    ),
    "bfcl": dict(
        base_dir           = f"{_RESULTS_ROOT}/bfcl_raw/tool_result/main",
        perceived_need_dir = f"{_RESULTS_ROOT}/bfcl_raw/tool_result/main-perceived-need",
        predictor_subdir   = "bfcl",      # models live at {root}/bfcl/{model}/predictor_{id}
        score_bins         = [-0.01, 0.99, 1.001],
        csv_suffix         = "_llm_judge",
        actual_need_thresh = 0.9,
        output_dir         = f"{_RESULTS_ROOT}/predictor/bfcl",
    ),
    "invivo": dict(
        base_dir           = f"{_RESULTS_ROOT}/real_query/temp=0/main",
        perceived_need_dir = f"{_RESULTS_ROOT}/real_query/temp=0/main-perceived-need",
        predictor_subdir   = "invivo",    # models live at {root}/invivo/{model}/predictor_{id}
        score_bins         = [-np.inf, 0.1, 0.9, np.inf],
        csv_suffix         = "",
        actual_need_thresh = 0.9,
        output_dir         = f"{_RESULTS_ROOT}/predictor/invivo",
    ),
}

# Defaults (overridden at runtime by --task)
BASE_DIR             = TASK_CONFIGS["entity"]["base_dir"]
PERCEIVED_NEED_DIR   = TASK_CONFIGS["entity"]["perceived_need_dir"]
PREDICTOR_DIR        = _PREDICTOR_ROOT
OUTPUT_DIR           = TASK_CONFIGS["entity"]["output_dir"]

ACTUAL_NEED_SCORE_THRESH = 0.9
UTILITY_POS_THRESH       = 0.0
SCORE_BINS               = TASK_CONFIGS["entity"]["score_bins"]
CSV_SUFFIX               = ""

CLASSIFIERS = ["LogisticRegression", "XGBoost", "MLP"]
# CLASSIFIERS = ["MLP"]

MODEL_NAMES = [
    "google_gemma-3-27b-it",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct",
]

MODEL_LABELS = {
    "google_gemma-3-27b-it":                        "Gemma3-27B",
    "openai_gpt-oss-120b":                           "GPT-OSS-120B",
    "Qwen_Qwen3-30B-A3B":                            "Qwen3-30B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507":              "Qwen3-30B-IT",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral-24B-IT",
    "meta-llama_Llama-3.2-3B-Instruct":              "Llama3.2-3B",
}

PREDICTOR_MODEL_NAMES = {
    "google_gemma-3-27b-it":                        "gemma-3-27b-it",
    "openai_gpt-oss-120b":                           "gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B":                            "Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507":              "Qwen3-30B-A3B-Instruct-2507",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct":              "Llama-3.2-3B-Instruct",
}

METRICS = ["balanced_accuracy", "accuracy", "macro_precision"]
METRIC_YLABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy":          "Accuracy",
    "macro_precision":   "Macro Precision",
}

NAN_METRICS = {k: float("nan") for k in METRICS}

_CURRENT_TASK = "entity"   # updated by _apply_task_config()


def _apply_task_config(task: str):
    """Update module-level globals for the given task."""
    global BASE_DIR, PERCEIVED_NEED_DIR, OUTPUT_DIR
    global ACTUAL_NEED_SCORE_THRESH, SCORE_BINS, CSV_SUFFIX, _CURRENT_TASK
    cfg = TASK_CONFIGS[task]
    BASE_DIR                 = cfg["base_dir"]
    PERCEIVED_NEED_DIR       = cfg["perceived_need_dir"]
    OUTPUT_DIR               = cfg["output_dir"]
    ACTUAL_NEED_SCORE_THRESH = cfg["actual_need_thresh"]
    SCORE_BINS               = cfg["score_bins"]
    CSV_SUFFIX               = cfg["csv_suffix"]
    _CURRENT_TASK            = task


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)


def bucketize_score(score: float) -> int:
    """Bin a score according to the current SCORE_BINS (set per task)."""
    if pd.isna(score):
        return np.nan
    for i in range(len(SCORE_BINS) - 1):
        if SCORE_BINS[i] < score <= SCORE_BINS[i + 1]:
            return i
    # Catch lower-bound edge (score == SCORE_BINS[0] for finite lower bounds)
    return 0


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute balanced accuracy, accuracy, and macro precision for a binary task.

    Returns a dict with keys:
        'balanced_accuracy' : float
        'accuracy'          : float
        'macro_precision'   : float
    All values are NaN if no valid samples exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return dict(NAN_METRICS)
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "accuracy":          accuracy_score(yt, yp),
        "macro_precision":   precision_score(yt, yp, average="macro", zero_division=0),
    }


# ── Perceived Need ────────────────────────────────────────────────────────────

def perceived_need_metrics(model_name: str) -> dict:
    """
    All three metrics for perceived need as a predictor of actual need.

    actual_need    = (no_search_score < ACTUAL_NEED_SCORE_THRESH)
    perceived_need = (yes_no_decision == 'yes')
    """
    no_search_path      = f"{BASE_DIR}/vllm_{model_name}_no_search_summary{CSV_SUFFIX}.csv"
    perceived_need_path = f"{PERCEIVED_NEED_DIR}/vllm_{model_name}_with_search_summary.csv"
    try:
        no_search_df      = load_csv(no_search_path)
        perceived_need_df = load_csv(perceived_need_path)
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        return dict(NAN_METRICS)

    no_search_df["score"] = pd.to_numeric(no_search_df["score"], errors="coerce").fillna(0)
    perceived_need_df.columns = perceived_need_df.columns.str.strip()

    min_len = min(len(no_search_df), len(perceived_need_df))
    scores  = no_search_df["score"].iloc[:min_len].values
    col     = "yes_no_decision" if "yes_no_decision" in perceived_need_df.columns else perceived_need_df.columns[0]
    raw     = perceived_need_df[col].iloc[:min_len]

    if pd.api.types.is_bool_dtype(raw):
        perceived_need = raw.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw):
        perceived_need = (raw != 0).astype(int).values
    else:
        perceived_need = (raw.astype(str).str.strip().str.lower() == "yes").astype(int).values

    actual_need = (scores < ACTUAL_NEED_SCORE_THRESH).astype(int)
    return compute_metrics(actual_need, perceived_need)


# ── Perceived Utility ─────────────────────────────────────────────────────────

def perceived_utility_metrics(model_name: str) -> dict:
    """
    All three metrics for perceived utility (search_called) vs actual utility.

    actual_utility    = (force_search bucket > no_search bucket)
    perceived_utility = (search_called == 1)
    """
    no_search_path    = f"{BASE_DIR}/vllm_{model_name}_no_search_summary{CSV_SUFFIX}.csv"
    force_search_path = f"{BASE_DIR}/vllm_{model_name}_force_search_summary{CSV_SUFFIX}.csv"
    auto_search_path  = f"{BASE_DIR}/vllm_{model_name}_with_search_summary.csv"
    try:
        no_search_df    = load_csv(no_search_path)
        force_search_df = load_csv(force_search_path)
        auto_search_df  = load_csv(auto_search_path)
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        return dict(NAN_METRICS)

    no_search_df["score"]    = pd.to_numeric(no_search_df["score"], errors="coerce").fillna(0)
    force_search_df["score"] = pd.to_numeric(force_search_df["score"], errors="coerce").fillna(0)
    auto_search_df.columns   = auto_search_df.columns.str.strip()

    min_len      = min(len(no_search_df), len(force_search_df), len(auto_search_df))
    s_no         = no_search_df["score"].iloc[:min_len].values
    s_force      = force_search_df["score"].iloc[:min_len].values

    bucket_no    = np.array([bucketize_score(x) for x in s_no],    dtype=float)
    bucket_force = np.array([bucketize_score(x) for x in s_force], dtype=float)
    valid        = ~np.isnan(bucket_no) & ~np.isnan(bucket_force)
    bucket_no    = bucket_no[valid].astype(int)
    bucket_force = bucket_force[valid].astype(int)

    actual_utility = (bucket_force > bucket_no).astype(int)

    raw = auto_search_df["search_called"].iloc[:min_len][valid]
    if pd.api.types.is_bool_dtype(raw):
        perceived_utility = raw.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw):
        perceived_utility = (raw != 0).astype(int).values
    else:
        perceived_utility = (
            raw.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        ).astype(int).values

    return compute_metrics(actual_utility, perceived_utility)


# ── Predictor ─────────────────────────────────────────────────────────────────

def _predictor_base(model_name: str, predictor_id: int) -> str:
    pred_model   = PREDICTOR_MODEL_NAMES.get(model_name, model_name)
    subdir       = TASK_CONFIGS[_CURRENT_TASK]["predictor_subdir"]
    if subdir:
        return os.path.join(_PREDICTOR_ROOT, subdir, pred_model, f"predictor_{predictor_id}")
    return os.path.join(_PREDICTOR_ROOT, pred_model, f"predictor_{predictor_id}")


def predictor_metrics(
    model_name: str,
    predictor_id: int,
    classifiers: list,
    **_kwargs,
) -> dict:
    """
    All three metrics for the given predictor_id.

    predictor.py convention:
        y_pred = 1 - df["prediction"]
        y_true = 1 - df["ground_truth_label"]
    """
    base = _predictor_base(model_name, predictor_id)

    all_metrics = []
    for clf in classifiers:
        f = os.path.join(base, f"predictions_{clf}.csv")
        if not os.path.exists(f):
            print(f"  [WARN] File not found: {f}")
            continue
        try:
            df     = pd.read_csv(f)
            y_pred = 1 - df["prediction"].astype(int)
            y_true = 1 - df["ground_truth_label"].astype(int)
            all_metrics.append(compute_metrics(y_true, y_pred))
        except Exception as e:
            print(f"  [WARN] Could not process {f}: {e}")

    if not all_metrics:
        return dict(NAN_METRICS)

    valid = [m for m in all_metrics if not np.isnan(m["macro_precision"])]
    if not valid:
        return dict(NAN_METRICS)

    return max(valid, key=lambda m: m["macro_precision"])


# ── Main computation ──────────────────────────────────────────────────────────

def compute_all(classifiers: list, layer_selection: str = "best") -> dict:
    """
    Returns a dict where each key (except 'models') holds a list of metric
    dicts — one per model — each with keys:
        'balanced_accuracy', 'accuracy', 'macro_precision'

    Top-level keys:
        "models"              : list of short display labels
        "perceived_need"      : list of metric dicts
        "predictor_1"         : list of metric dicts
        "perceived_utility"   : list of metric dicts
        "predictor_2"         : list of metric dicts
        "predictor_3"         : list of metric dicts
        "predictor1_on_p2_gt" : list of metric dicts
    """
    results = {k: [] for k in [
        "models", "perceived_need", "predictor_1",
        "perceived_utility", "predictor_2", "predictor_3", "predictor1_on_p2_gt",
    ]}

    for m in MODEL_NAMES:
        label = MODEL_LABELS.get(m, m)
        print(f"\n{'─'*65}")
        print(f"  Model: {label}  ({m})")
        print(f"{'─'*65}")

        pn   = perceived_need_metrics(m)
        pu   = perceived_utility_metrics(m)
        p1   = predictor_metrics(m, 1, classifiers)
        p2   = predictor_metrics(m, 2, classifiers)
        p3   = predictor_metrics(m, 3, classifiers)

        def fmt(d):
            return (f"bal_acc={d['balanced_accuracy']:.4f}  "
                    f"acc={d['accuracy']:.4f}  "
                    f"macro_prec={d['macro_precision']:.4f}")

        print(f"  Perceived Need  : {fmt(pn)}")
        print(f"  Predictor 1     : {fmt(p1)}")
        print(f"  Perceived Util  : {fmt(pu)}")
        print(f"  Predictor 2     : {fmt(p2)}")
        print(f"  Predictor 3     : {fmt(p3)}")

        results["models"].append(label)
        results["perceived_need"].append(pn)
        results["predictor_1"].append(p1)
        results["perceived_utility"].append(pu)
        results["predictor_2"].append(p2)
        results["predictor_3"].append(p3)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

BAR_COLORS = {
    "perceived":  "#4C72B0",
    "predictor":  "#DD8452",
    "predictor2": "#55A868",
    "predictor3": "#C44E52",
}


def _get_vals(metric_dicts: list, metric: str) -> list:
    return [d[metric] for d in metric_dicts]


def _annotate_bars(ax, bars, vals):
    for bar, v in zip(bars, vals):
        h = bar.get_height()
        if np.isnan(v) or h == 0:
            continue
        # ax.text(
        #     bar.get_x() + bar.get_width() / 2,
        #     h + 0.008,
        #     f"{h:.2f}",
        #     ha="center", va="bottom",
        #     fontsize=14, fontweight="bold", color="black",
        # )


def _style_ax(ax, model_labels, ylabel):
    n = len(model_labels)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(model_labels, fontsize=35, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=35, fontweight="bold", labelpad=10)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1),
               ncol=3, frameon=False, fontsize=25)


def _bar_plot(
    model_labels: list,
    bar_a_dicts: list,
    bar_b_dicts: list,
    label_a: str,
    label_b: str,
    output_path_prefix: str,
):
    """Two-bar grouped chart, saved once per metric."""
    n     = len(model_labels)
    x     = np.arange(n)
    width = 0.35
    gap   = 0.04

    for metric in METRICS:
        vals_a = _get_vals(bar_a_dicts, metric)
        vals_b = _get_vals(bar_b_dicts, metric)

        fig, ax = plt.subplots(figsize=(max(10, n * 1.8), 6))
        fig.patch.set_facecolor("white")

        bars_a = ax.bar(x - width / 2 - gap / 2, np.nan_to_num(vals_a), width,
                        label=label_a, color=BAR_COLORS["perceived"],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_b = ax.bar(x + width / 2 + gap / 2, np.nan_to_num(vals_b), width,
                        label=label_b, color=BAR_COLORS["predictor"],
                        edgecolor="white", linewidth=0.8, zorder=3)

        _annotate_bars(ax, bars_a, vals_a)
        _annotate_bars(ax, bars_b, vals_b)
        _style_ax(ax, model_labels, METRIC_YLABELS[metric])

        plt.tight_layout()
        out = f"{output_path_prefix}_{metric}.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ Saved: {out}")


def _bar_plot_three(
    model_labels: list,
    bar_a_dicts: list,
    bar_b_dicts: list,
    bar_c_dicts: list,
    label_a: str,
    label_b: str,
    label_c: str,
    output_path_prefix: str,
):
    """Three-bar grouped chart (predictors 1/2/3), saved once per metric."""
    n     = len(model_labels)
    x     = np.arange(n)
    width = 0.24

    for metric in METRICS:
        vals_a = _get_vals(bar_a_dicts, metric)
        vals_b = _get_vals(bar_b_dicts, metric)
        vals_c = _get_vals(bar_c_dicts, metric)

        fig, ax = plt.subplots(figsize=(max(11, n * 2.0), 6))
        fig.patch.set_facecolor("white")

        bars_a = ax.bar(x - width, np.nan_to_num(vals_a), width,
                        label=label_a, color=BAR_COLORS["predictor"],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_b = ax.bar(x,          np.nan_to_num(vals_b), width,
                        label=label_b, color=BAR_COLORS["predictor2"],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_c = ax.bar(x + width,  np.nan_to_num(vals_c), width,
                        label=label_c, color=BAR_COLORS["predictor3"],
                        edgecolor="white", linewidth=0.8, zorder=3)

        _annotate_bars(ax, bars_a, vals_a)
        _annotate_bars(ax, bars_b, vals_b)
        _annotate_bars(ax, bars_c, vals_c)
        _style_ax(ax, model_labels, METRIC_YLABELS[metric])

        plt.tight_layout()
        out = f"{output_path_prefix}_{metric}.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✓ Saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_task(task: str, classifiers: list, output_dir_override: str | None):
    _apply_task_config(task)
    out = output_dir_override or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    print("\n" + "═" * 65)
    print(f"  Task: {task.upper()}")
    print("  Computing metrics (balanced accuracy, accuracy, macro precision) …")
    print(f"  Classifiers : {classifiers}")
    print("═" * 65)

    data   = compute_all(classifiers)
    models = data["models"]

    # Plot 1: Actual Need
    _bar_plot(
        model_labels       = models,
        bar_a_dicts        = data["perceived_need"],
        bar_b_dicts        = data["predictor_1"],
        label_a            = "Perceived Need",
        label_b            = "Predicted Need",
        output_path_prefix = os.path.join(out, f"{task}_barplot_actual_need"),
    )

    # Plot 2: Actual Utility – Predictor 2
    _bar_plot(
        model_labels       = models,
        bar_a_dicts        = data["perceived_utility"],
        bar_b_dicts        = data["predictor_2"],
        label_a            = "Perceived Utility",
        label_b            = "Predicted Utility",
        output_path_prefix = os.path.join(out, f"{task}_barplot_actual_utility_pred2"),
    )

    # Plot 3: Actual Utility – Predictor 3
    _bar_plot(
        model_labels       = models,
        bar_a_dicts        = data["perceived_utility"],
        bar_b_dicts        = data["predictor_3"],
        label_a            = "Perceived Utility",
        label_b            = "Predicted Utility",
        output_path_prefix = os.path.join(out, f"{task}_barplot_actual_utility_pred3"),
    )

    # Plot 4: Perceived Utility vs Predictor 2 vs Predictor 3
    _bar_plot_three(
        model_labels       = models,
        bar_a_dicts        = data["perceived_utility"],
        bar_b_dicts        = data["predictor_2"],
        bar_c_dicts        = data["predictor_3"],
        label_a            = "Perceived Utility",
        label_b            = r"LUE$_{x,d}$",
        label_c            = r"LUE$_{x}$",
        output_path_prefix = os.path.join(out, f"{task}_barplot_pred2_vs_pred3"),
    )

    print(f"\n  All plots for {task} saved to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bar plots (balanced accuracy, accuracy, macro precision) "
                    "for need / utility predictors."
    )
    parser.add_argument(
        "--clf", nargs="+", default=CLASSIFIERS,
        metavar="CLF",
        help=f"Classifiers to include. Defaults to: {CLASSIFIERS}",
    )
    parser.add_argument(
        "--task", default="entity",
        choices=list(TASK_CONFIGS.keys()) + ["all"],
        help="Task to evaluate: entity, bfcl, invivo, or all. Default: entity",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save plots. Defaults to per-task output dir.",
    )
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS.keys()) if args.task == "all" else [args.task]
    for task in tasks:
        _run_task(task, args.clf, args.output_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()

