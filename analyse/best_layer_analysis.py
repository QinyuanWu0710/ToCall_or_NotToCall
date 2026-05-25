"""
best_layer_analysis.py

Single script that, for every task × model × predictor_id combination:

  1. Reads layer_search_summary.json  →  identifies the best layer index and
     best classifier (by OOF AUROC).
  2. Loads the corresponding predictions CSV produced by layer_search.py.
  3. Computes factuality scores under every search strategy:
       no_search | force_search | oracle | auto_call | predictor_best_layer
  4. Saves a per-task comparison CSV.
  5. Plots confusion matrices  (one per model × predictor).
  6. Plots AUROC-by-layer curves (one per task × predictor).

Outputs land in:
  {OUTPUT_ROOT}/{task}/comparison_best_layer.csv
  {OUTPUT_ROOT}/{task}/classifier_auroc_summary.csv
  {OUTPUT_ROOT}/{task}/figures/
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = "/NS/chatgpt/work/qwu/hallucinations_detection"
PREDICTOR_DIR = os.path.join(BASE_DIR, "data/tool_predictor")
OUTPUT_ROOT   = os.path.join(BASE_DIR, "results/predictor_best_layer")

PREDICTOR_IDS = [1, 2, 3]     # predictors to analyse
CLASSIFIERS   = ["MLP"]
SELECTION_METRIC = "accuracy"

# summary CSV model key  →  predictor folder name
MODEL_MAPPING = {
    "google_gemma-3-27b-it":                        "gemma-3-27b-it",
    "openai_gpt-oss-120b":                          "gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B":                           "Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507":             "Qwen3-30B-A3B-Instruct-2507",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503":"Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct":             "Llama-3.2-3B-Instruct",
}

MODEL_IN_LEGEND = {
    "google_gemma-3-27b-it":                        "Gemma3-27B",
    "openai_gpt-oss-120b":                          "GPT-OSS-120B",
    "Qwen_Qwen3-30B-A3B":                           "Qwen3-30B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507":             "Qwen3-30B-IT",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503":"Mistral3.1-24B",
    "meta-llama_Llama-3.2-3B-Instruct":             "Llama3.2-3B",
}

PREDICTOR_AXIS_LABELS = {
    1: (["Need", "No Need"], ["Need", "No Need"]),
    2: (["Help", "No Help"], ["Help", "No Help"]),
    3: (["Help", "No Help"], ["Help", "No Help"]),
}

TASKS = {
    "entity": dict(
        results_dir       = os.path.join(BASE_DIR, "results/entity_hallucination/temp=0"),
        predictor_subdir  = "",
        key               = "entity",
        score_suffix      = "",
        score_bins        = [-np.inf, 0.1, 0.9, np.inf],
        actual_need_thresh= 0.9,
    ),
    "bfcl": dict(
        results_dir       = os.path.join(BASE_DIR, "results/bfcl_raw/tool_result"),
        predictor_subdir  = "bfcl",
        key               = "query",
        score_suffix      = "",
        score_bins        = [-0.01, 0.99, 1.001],
        actual_need_thresh= 0.9,
    ),
    "invivo": dict(
        results_dir       = os.path.join(BASE_DIR, "results/real_query/temp=0"),
        predictor_subdir  = "invivo",
        key               = "entity",
        score_suffix      = "",
        score_bins        = [-np.inf, 0.1, 0.9, np.inf],
        actual_need_thresh= 0.9,
    ),
}

MAX_SAMPLES = 500

METRICS = ["balanced_accuracy", "accuracy", "macro_precision"]
METRIC_YLABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy":          "Accuracy",
    "macro_precision":   "Macro Precision",
}
NAN_METRICS = {k: float("nan") for k in METRICS}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


def predictor_root(predictor_name: str, task_cfg: dict) -> str:
    subdir = task_cfg["predictor_subdir"]
    if subdir:
        return os.path.join(PREDICTOR_DIR, subdir, predictor_name)
    return os.path.join(PREDICTOR_DIR, predictor_name)


def layer_search_dir(predictor_name: str, predictor_id: int, task_cfg: dict) -> str:
    return os.path.join(
        predictor_root(predictor_name, task_cfg),
        f"predictor_{predictor_id}_layer_search",
    )


def load_layer_search_summary(predictor_name: str, predictor_id: int, task_cfg: dict):
    path = os.path.join(
        layer_search_dir(predictor_name, predictor_id, task_cfg),
        "layer_search_summary.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def best_clf_from_summary(summary: dict) -> str:
    """Return classifier with highest SELECTION_METRIC at the best layer."""
    best_layer = summary["best_layer"]
    layer_scores = summary["all_layer_scores"].get(str(best_layer), {})
    best_clf, best_val = None, -1.0
    for clf, metrics in layer_scores.items():
        v = metrics.get(SELECTION_METRIC, -1.0) or -1.0
        if v > best_val:
            best_val = v
            best_clf = clf
    return best_clf


def load_best_predictions(predictor_name: str, predictor_id: int, task_cfg: dict):
    """
    Returns (pred_df, best_layer, best_clf, summary) or None if unavailable.
    pred_df has columns: original_row_index, prediction, ground_truth_label.
    """
    summary = load_layer_search_summary(predictor_name, predictor_id, task_cfg)
    if summary is None:
        return None

    best_layer = summary["best_layer"]
    best_clf   = "MLP"

    pred_path = os.path.join(
        layer_search_dir(predictor_name, predictor_id, task_cfg),
        f"layer_{best_layer}",
        f"predictions_{best_clf}_layer{best_layer}.csv",
    )
    pred_df = load_csv(pred_path)
    if pred_df is None:
        return None

    return pred_df, best_layer, best_clf, summary


# ──────────────────────────────────────────────────────────────────────────────
# Score computation  (mirrors compare.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_scores(model_name: str, task_cfg: dict) -> dict | None:
    predictor_name = MODEL_MAPPING[model_name]
    results_dir    = task_cfg["results_dir"]
    key            = task_cfg["key"]
    suffix         = task_cfg["score_suffix"]
    main_dir       = os.path.join(results_dir, "main")

    no_search    = load_csv(os.path.join(main_dir, f"vllm_{model_name}_no_search{suffix}_summary.csv"))
    force_search = load_csv(os.path.join(main_dir, f"vllm_{model_name}_force_search{suffix}_summary.csv"))
    auto_call    = load_csv(os.path.join(main_dir, f"vllm_{model_name}_with_search{suffix}_summary.csv"))

    if no_search is None or force_search is None:
        return None

    merged = (
        no_search[[key, "score"]].rename(columns={"score": "no_search_score"})
        .merge(
            force_search[[key, "score"]].rename(columns={"score": "force_search_score"}),
            on=key, how="inner",
        )
    )
    merged[["no_search_score", "force_search_score"]] = (
        merged[["no_search_score", "force_search_score"]].fillna(0)
    )
    merged = merged.iloc[:MAX_SAMPLES].reset_index(drop=True)
    n = len(merged)

    scores       = {}
    force_counts = {}
    meta         = {}   # stores best_layer / best_clf per predictor

    scores["no_search"]    = merged["no_search_score"].mean()
    force_counts["no_search"] = 0

    scores["force_search"] = merged["force_search_score"].mean()
    force_counts["force_search"] = n

    scores["oracle"]       = merged[["no_search_score", "force_search_score"]].max(axis=1).mean()
    force_counts["oracle"] = int((merged["force_search_score"] > merged["no_search_score"]).sum())

    if auto_call is not None:
        df = merged.merge(auto_call[[key, "search_called"]], on=key, how="inner")
        sc = df["search_called"]
        if pd.api.types.is_bool_dtype(sc):
            df["search_called_bool"] = sc.astype(int)
        elif pd.api.types.is_numeric_dtype(sc):
            df["search_called_bool"] = (sc != 0).astype(int)
        else:
            df["search_called_bool"] = (
                sc.astype(str).str.strip().str.upper().isin(["1", "TRUE", "YES", "Y"])
            ).astype(int)
        df["auto_score"] = np.where(df["search_called_bool"], df["force_search_score"], df["no_search_score"])
        scores["auto_call"]       = df["auto_score"].mean()
        force_counts["auto_call"] = int(df["search_called_bool"].sum())
    else:
        scores["auto_call"]       = None
        force_counts["auto_call"] = None

    for pid in PREDICTOR_IDS:
        k = f"predictor_{pid}_best"
        result = load_best_predictions(predictor_name, pid, task_cfg)
        if result is None:
            scores[k]       = None
            force_counts[k] = None
            meta[k]         = {}
            continue

        pred_df, best_layer, best_clf, summary = result
        pred_df = pred_df.set_index("original_row_index")["prediction"]

        valid_idx     = pred_df.index.tolist()
        merged_subset = merged.iloc[valid_idx]

        # predictor_1: label=0 → needs search (use force)
        # predictor_2+: label=1 → search helps (use force)
        use_force = pred_df.values == (0 if pid == 1 else 1)

        pred_score       = np.where(
            use_force,
            merged_subset["force_search_score"].values,
            merged_subset["no_search_score"].values,
        )
        scores[k]       = pred_score.mean()
        force_counts[k] = int(use_force.sum())
        meta[k]         = {"best_layer": best_layer, "best_clf": best_clf,
                           "oof_roc_auc": summary.get("best_score")}

    return {"scores": scores, "force_counts": force_counts, "meta": meta}


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

TASK_CM_VMAX = {
    "entity": 500,
    "invivo": 500,
    "bfcl":   314,
}


def plot_confusion_matrix(
    pred_df: pd.DataFrame,
    predictor_id: int,
    out_path: str,
    task: str = "entity",
):
    y_pred = 1 - pred_df["prediction"].astype(int)
    y_true = 1 - pred_df["ground_truth_label"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_pct = np.divide(
        cm.astype(float), col_sums,
        out=np.zeros_like(cm, dtype=float),
        where=col_sums != 0,
    ) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)"

    xticklabels, yticklabels = PREDICTOR_AXIS_LABELS.get(
        predictor_id, (["Pos", "Neg"], ["Pos", "Neg"])
    )

    vmax = TASK_CM_VMAX.get(task, 500)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        linewidths=0.5, linecolor="gray",
        xticklabels=xticklabels, yticklabels=yticklabels,
        ax=ax, cbar=False,
        annot_kws={"size": 35, "weight": "bold"},
        vmin=0, vmax=vmax,
    )
    ax.set_xlabel("Predicted", fontsize=25, labelpad=16, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Actual", fontsize=25, labelpad=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [CM] saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: AUROC by layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_auroc_by_layer(
    layer_data: dict,   # {model_key: {layer_idx: {clf: oof_roc_auc}}}
    predictor_id: int,
    task: str,
    out_path: str,
):
    """
    One line per model, x = layer index, y = max AUROC across classifiers.
    """
    if not layer_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for idx, (model_key, layer_dict) in enumerate(layer_data.items()):
        if not layer_dict:
            continue
        xs = sorted(layer_dict.keys())
        ys = [
            max(
                (layer_dict[x].get(clf) or 0.0 for clf in CLASSIFIERS),
                default=0.0,
            )
            for x in xs
        ]
        label = MODEL_IN_LEGEND.get(model_key, model_key)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5,
                color=colors[idx % len(colors)], label=label)

    ax.set_xlabel("Layer index", fontsize=13)
    ax.set_ylabel(f"OOF AUROC", fontsize=13)
    ax.set_title(f"AUROC by layer — task={task}  predictor={predictor_id}", fontsize=13)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="random")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [AUROC] saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: grouped bar chart — score comparison across strategies
# (style matches bar_perceived.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

BAR_COLORS = [
    "#4C72B0",  # no_search
    "#DD8452",  # force_search
    "#55A868",  # oracle
    "#C44E52",  # auto_call
    "#8172B2",  # predictor_1_best
    "#937860",  # predictor_2_best
    "#64B5CD",  # predictor_3_best
]

STRATEGY_LABELS = {
    "no_search":        "No Search",
    "force_search":     "Force Search",
    "oracle":           "Oracle",
    "auto_call":        "Auto Call",
    "predictor_1_best": "LNE",
    "predictor_2_best": r"LUE$_{x,d}$",
    "predictor_3_best": r"LUE$_{x}$",
}


def _style_ax(ax, model_labels, ylabel="Factuality Score"):
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


def plot_score_comparison(df_scores: pd.DataFrame, task: str, out_path: str):
    strategy_cols = [c for c in df_scores.columns if c != "model"]
    if not strategy_cols:
        return

    models = [MODEL_IN_LEGEND.get(m, m) for m in df_scores["model"]]
    n      = len(models)
    x      = np.arange(n)
    k      = len(strategy_cols)
    width  = 0.8 / k

    fig, ax = plt.subplots(figsize=(max(10, n * 1.8), 6))
    fig.patch.set_facecolor("white")

    for i, col in enumerate(strategy_cols):
        vals   = pd.to_numeric(df_scores[col], errors="coerce").fillna(0).values
        offset = (i - k / 2 + 0.5) * width
        label  = STRATEGY_LABELS.get(col, col)
        color  = BAR_COLORS[i % len(BAR_COLORS)]
        ax.bar(x + offset, vals, width * 0.9,
               label=label, color=color,
               edgecolor="white", linewidth=0.8, zorder=3)

    _style_ax(ax, models, ylabel="Factuality Score")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [BAR] saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-task classifier AUROC summary CSV
# ──────────────────────────────────────────────────────────────────────────────

def build_classifier_auroc_table(task: str, task_cfg: dict) -> pd.DataFrame:
    rows = []
    for model_key, predictor_name in MODEL_MAPPING.items():
        for pid in PREDICTOR_IDS:
            summary = load_layer_search_summary(predictor_name, pid, task_cfg)
            if summary is None:
                continue
            best_layer = summary["best_layer"]
            for layer_str, clf_metrics in summary["all_layer_scores"].items():
                layer_idx = int(layer_str)
                for clf, metrics in clf_metrics.items():
                    rows.append({
                        "task":          task,
                        "model":         model_key,
                        "predictor_id":  pid,
                        "layer":         layer_idx,
                        "classifier":    clf,
                        "oof_roc_auc":   metrics.get("oof_roc_auc"),
                        "mean_fold_roc": metrics.get("mean_fold_roc_auc"),
                        "accuracy":      metrics.get("accuracy"),
                        "balanced_acc":  metrics.get("balanced_accuracy"),
                        "is_best_layer": layer_idx == best_layer,
                    })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Classification metric helpers  (mirrors bar_perceived.py)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred) -> dict:
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


def _bucketize(score: float, score_bins: list) -> int:
    for i in range(len(score_bins) - 1):
        if score_bins[i] < score <= score_bins[i + 1]:
            return i
    return 0


def perceived_need_metrics(model_name: str, task_cfg: dict) -> dict:
    main_dir   = os.path.join(task_cfg["results_dir"], "main")
    pneed_dir  = os.path.join(task_cfg["results_dir"], "main-perceived-need")
    suffix     = task_cfg["score_suffix"]
    thresh     = task_cfg["actual_need_thresh"]

    ns_path = os.path.join(main_dir, f"vllm_{model_name}_no_search{suffix}_summary.csv")
    pn_path = os.path.join(pneed_dir, f"vllm_{model_name}_with_search_summary.csv")

    ns_df = load_csv(ns_path)
    pn_df = load_csv(pn_path)
    if ns_df is None or pn_df is None:
        return dict(NAN_METRICS)

    ns_df["score"] = pd.to_numeric(ns_df["score"], errors="coerce").fillna(0)
    pn_df.columns  = pn_df.columns.str.strip()

    n       = min(len(ns_df), len(pn_df), MAX_SAMPLES)
    scores  = ns_df["score"].iloc[:n].values
    col     = "yes_no_decision" if "yes_no_decision" in pn_df.columns else pn_df.columns[0]
    raw     = pn_df[col].iloc[:n]

    if pd.api.types.is_bool_dtype(raw):
        perceived = raw.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw):
        perceived = (raw != 0).astype(int).values
    else:
        perceived = (raw.astype(str).str.strip().str.lower() == "yes").astype(int).values

    actual = (scores < thresh).astype(int)
    return _compute_metrics(actual, perceived)


def perceived_utility_metrics(model_name: str, task_cfg: dict) -> dict:
    main_dir   = os.path.join(task_cfg["results_dir"], "main")
    suffix     = task_cfg["score_suffix"]
    score_bins = task_cfg["score_bins"]

    ns_path   = os.path.join(main_dir, f"vllm_{model_name}_no_search{suffix}_summary.csv")
    fs_path   = os.path.join(main_dir, f"vllm_{model_name}_force_search{suffix}_summary.csv")
    auto_path = os.path.join(main_dir, f"vllm_{model_name}_with_search_summary.csv")

    ns_df   = load_csv(ns_path)
    fs_df   = load_csv(fs_path)
    auto_df = load_csv(auto_path)
    if ns_df is None or fs_df is None or auto_df is None:
        return dict(NAN_METRICS)

    ns_df["score"]   = pd.to_numeric(ns_df["score"],   errors="coerce").fillna(0)
    fs_df["score"]   = pd.to_numeric(fs_df["score"],   errors="coerce").fillna(0)
    auto_df.columns  = auto_df.columns.str.strip()

    n           = min(len(ns_df), len(fs_df), len(auto_df), MAX_SAMPLES)
    bucket_no   = np.array([_bucketize(x, score_bins) for x in ns_df["score"].iloc[:n].values])
    bucket_force= np.array([_bucketize(x, score_bins) for x in fs_df["score"].iloc[:n].values])
    actual      = (bucket_force > bucket_no).astype(int)

    raw = auto_df["search_called"].iloc[:n]
    if pd.api.types.is_bool_dtype(raw):
        perceived = raw.astype(int).values
    elif pd.api.types.is_numeric_dtype(raw):
        perceived = (raw != 0).astype(int).values
    else:
        perceived = (
            raw.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        ).astype(int).values

    return _compute_metrics(actual, perceived)


def best_layer_predictor_metrics(model_key: str, predictor_id: int, task_cfg: dict) -> dict:
    predictor_name = MODEL_MAPPING[model_key]
    result = load_best_predictions(predictor_name, predictor_id, task_cfg)
    if result is None:
        return dict(NAN_METRICS)
    pred_df, _, _, _ = result
    y_pred = 1 - pred_df["prediction"].astype(int)
    y_true = 1 - pred_df["ground_truth_label"].astype(int)
    return _compute_metrics(y_true, y_pred)


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy bar plots  (style matches bar_perceived.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _get_vals(metric_dicts: list, metric: str) -> list:
    return [d[metric] for d in metric_dicts]


def _annotate_bars(ax, bars, vals):
    pass   # annotations commented out in bar_perceived.py


def _bar_plot(model_labels, bar_a_dicts, bar_b_dicts,
              label_a, label_b, output_path_prefix):
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
                        label=label_a, color=BAR_COLORS[0],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_b = ax.bar(x + width / 2 + gap / 2, np.nan_to_num(vals_b), width,
                        label=label_b, color=BAR_COLORS[1],
                        edgecolor="white", linewidth=0.8, zorder=3)

        _annotate_bars(ax, bars_a, vals_a)
        _annotate_bars(ax, bars_b, vals_b)
        _style_ax(ax, model_labels, ylabel=METRIC_YLABELS[metric])

        plt.tight_layout()
        out = f"{output_path_prefix}_{metric}.pdf"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [BAR] saved → {out}")


def _bar_plot_three(model_labels, bar_a_dicts, bar_b_dicts, bar_c_dicts,
                    label_a, label_b, label_c, output_path_prefix):
    """Three-bar grouped chart, saved once per metric."""
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
                        label=label_a, color=BAR_COLORS[1],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_b = ax.bar(x,          np.nan_to_num(vals_b), width,
                        label=label_b, color=BAR_COLORS[2],
                        edgecolor="white", linewidth=0.8, zorder=3)
        bars_c = ax.bar(x + width,  np.nan_to_num(vals_c), width,
                        label=label_c, color=BAR_COLORS[3],
                        edgecolor="white", linewidth=0.8, zorder=3)

        _annotate_bars(ax, bars_a, vals_a)
        _annotate_bars(ax, bars_b, vals_b)
        _annotate_bars(ax, bars_c, vals_c)
        _style_ax(ax, model_labels, ylabel=METRIC_YLABELS[metric])

        plt.tight_layout()
        out = f"{output_path_prefix}_{metric}.pdf"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [BAR] saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_task(task: str, task_cfg: dict):
    print(f"\n{'='*70}")
    print(f"  TASK: {task}")
    print(f"{'='*70}")

    out_dir  = os.path.join(OUTPUT_ROOT, task)
    fig_dir  = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Compute comparison scores ─────────────────────────────────────────
    score_rows  = []
    count_rows  = []
    meta_rows   = []

    for model_key in MODEL_MAPPING:
        result = compute_scores(model_key, task_cfg)
        if result is None:
            print(f"  skip {model_key} (missing score CSVs)")
            continue
        score_rows.append({"model": model_key, **result["scores"]})
        count_rows.append({"model": model_key, **result["force_counts"]})
        meta_rows.append( {"model": model_key, **result["meta"]})

    if score_rows:
        df_scores = pd.DataFrame(score_rows)
        df_counts = pd.DataFrame(count_rows)

        # Build combined CSV: interleave score + count columns
        score_cols = [c for c in df_scores.columns if c != "model"]
        combined = pd.DataFrame({"model": df_scores["model"]})
        for col in score_cols:
            combined[col] = df_scores[col]
            combined[f"{col}_count"] = df_counts[col]

        csv_path = os.path.join(out_dir, "comparison_best_layer.csv")
        combined.to_csv(csv_path, index=False)
        print(f"\n  [CSV] scores → {csv_path}")

        # pretty-print with counts in parentheses
        display = pd.DataFrame({"model": df_scores["model"]})
        for col in score_cols:
            display[col] = [
                (
                    f"{df_scores.loc[i, col]:.4f} ({int(df_counts.loc[i, col])})"
                    if pd.notna(df_scores.loc[i, col]) and pd.notna(df_counts.loc[i, col])
                    else "N/A"
                )
                for i in df_scores.index
            ]
        print(display.to_string(index=False))

        # ── score bar chart ──────────────────────────────────────────────────
        plot_score_comparison(
            df_scores,
            task=task,
            out_path=os.path.join(fig_dir, "score_comparison.pdf"),
        )

    # ── 2. Classifier AUROC summary table ────────────────────────────────────
    df_auroc = build_classifier_auroc_table(task, task_cfg)
    if not df_auroc.empty:
        auroc_path = os.path.join(out_dir, "classifier_auroc_summary.csv")
        df_auroc.to_csv(auroc_path, index=False)
        print(f"  [CSV] AUROC table → {auroc_path}")

    # ── 3. Confusion matrices + AUROC-by-layer plots ─────────────────────────
    for pid in PREDICTOR_IDS:
        # collect per-model layer AUROC data for the AUROC-by-layer figure
        layer_auroc_data: dict = {}

        for model_key, predictor_name in MODEL_MAPPING.items():
            summary = load_layer_search_summary(predictor_name, pid, task_cfg)
            if summary is None:
                continue

            # gather all-layer AUROC for this model
            layer_auroc_data[model_key] = {
                int(l): {clf: m.get("oof_roc_auc") for clf, m in clf_dict.items()}
                for l, clf_dict in summary["all_layer_scores"].items()
            }

            # confusion matrix at best layer
            result = load_best_predictions(predictor_name, pid, task_cfg)
            if result is None:
                continue
            pred_df, best_layer, best_clf, _ = result

            short_name = MODEL_IN_LEGEND.get(model_key, predictor_name)
            cm_path = os.path.join(
                fig_dir,
                f"predictor{pid}_{short_name}_layer{best_layer}_{best_clf}_cm.pdf",
            )
            plot_confusion_matrix(pred_df, pid, cm_path, task=task)

        # AUROC-by-layer for this task × predictor
        if layer_auroc_data:
            plot_auroc_by_layer(
                layer_auroc_data,
                predictor_id=pid,
                task=task,
                out_path=os.path.join(fig_dir, f"predictor{pid}_auroc_by_layer.pdf"),
            )

    # ── 4. Best layer summary table (one row per model × predictor) ──────────
    best_rows = []
    for model_key, predictor_name in MODEL_MAPPING.items():
        for pid in PREDICTOR_IDS:
            summary = load_layer_search_summary(predictor_name, pid, task_cfg)
            if summary is None:
                continue
            best_layer = summary["best_layer"]
            best_clf   = best_clf_from_summary(summary)
            best_score = summary.get("best_score")
            best_rows.append({
                "task":         task,
                "model":        model_key,
                "predictor_id": pid,
                "best_layer":   best_layer,
                "best_clf":     best_clf,
                SELECTION_METRIC: best_score,
            })

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        best_path = os.path.join(out_dir, "best_layer_per_model.csv")
        df_best.to_csv(best_path, index=False)
        print(f"  [CSV] best-layer table → {best_path}")
        print(df_best.to_string(index=False))

    # ── 5. Accuracy bar plots (balanced_accuracy / accuracy / macro_precision) ─
    print(f"\n  Computing classification metrics for bar plots …")
    model_labels     = [MODEL_IN_LEGEND.get(m, m) for m in MODEL_MAPPING]
    pneed_metrics    = [perceived_need_metrics(m, task_cfg)       for m in MODEL_MAPPING]
    putility_metrics = [perceived_utility_metrics(m, task_cfg)    for m in MODEL_MAPPING]
    pred1_metrics    = [best_layer_predictor_metrics(m, 1, task_cfg) for m in MODEL_MAPPING]
    pred2_metrics    = [best_layer_predictor_metrics(m, 2, task_cfg) for m in MODEL_MAPPING]
    pred3_metrics    = [best_layer_predictor_metrics(m, 3, task_cfg) for m in MODEL_MAPPING]

    # Plot A: Perceived Need vs LNE (Predictor 1)
    _bar_plot(
        model_labels       = model_labels,
        bar_a_dicts        = pneed_metrics,
        bar_b_dicts        = pred1_metrics,
        label_a            = "Perceived Need",
        label_b            = "LNE",
        output_path_prefix = os.path.join(fig_dir, f"{task}_barplot_need"),
    )

    # Plot B: Perceived Utility vs LUE_{x,d} (Predictor 2)
    _bar_plot(
        model_labels       = model_labels,
        bar_a_dicts        = putility_metrics,
        bar_b_dicts        = pred2_metrics,
        label_a            = "Perceived Utility",
        label_b            = r"LUE$_{x,d}$",
        output_path_prefix = os.path.join(fig_dir, f"{task}_barplot_pred2"),
    )

    # Plot C: Perceived Utility vs LUE_{x} (Predictor 3)
    _bar_plot(
        model_labels       = model_labels,
        bar_a_dicts        = putility_metrics,
        bar_b_dicts        = pred3_metrics,
        label_a            = "Perceived Utility",
        label_b            = r"LUE$_{x}$",
        output_path_prefix = os.path.join(fig_dir, f"{task}_barplot_pred3"),
    )

    # Plot D: Perceived Utility vs LUE_{x,d} vs LUE_{x} (three-bar)
    _bar_plot_three(
        model_labels       = model_labels,
        bar_a_dicts        = putility_metrics,
        bar_b_dicts        = pred2_metrics,
        bar_c_dicts        = pred3_metrics,
        label_a            = "Perceived Utility",
        label_b            = r"LUE$_{x,d}$",
        label_c            = r"LUE$_{x}$",
        output_path_prefix = os.path.join(fig_dir, f"{task}_barplot_pred2_vs_pred3"),
    )


def main():
    for task, task_cfg in TASKS.items():
        run_task(task, task_cfg)


if __name__ == "__main__":
    main()
