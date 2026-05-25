"""
predictor_affordability_plot.py

Plots a combined affordability figure showing (per model):
  1. Ideal affordability curve   — solid line, oracle ordering by delta
  2. Actual model performance    — dashed line + dots from cost-aware budget v2 CSVs
  3. Predictor affordability     — entities ranked by predicted probability of needing search

Predictor search-label convention (mirrors best_layer_analysis.py):
  predictor_1 : label=0  → needs search  → rank by prob_class_0  (↑ = more likely to need)
  predictor_2+: label=1  → search helps  → rank by prob_class_1  (↑ = more likely to benefit)

Usage:
    python predictor_affordability_plot.py [--task entity] [--predictor 1] [--models ...]
"""

import os
import sys
import argparse
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Locate the analyse directory and import sibling modules ──────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import new_analysis as na
import best_layer_analysis as bla

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_TASK      = "invivo"
# Change this to 2 or 3 to switch predictors
DEFAULT_PREDICTOR = 1

PREDICTOR_LEGEND_NAMES = {
    1: r"LNE",
    2: r"LUE$_{x,d}$",
    3: r"LUE$_{x}$",
}

BUDGET_TOTAL      = 10_000
COST_SETUPS       = [10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0]

PALETTE = [
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

MODEL_NAMES = [
    "google_gemma-3-27b-it",
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct",
]

# ── Predictor probability column selection ────────────────────────────────────

def _prob_col_for_predictor(predictor_id: int) -> str:
    """
    Return the probability column name to rank entities by.
    predictor_1: search is triggered when label=0  → use prob_class_0
    predictor_2+: search is triggered when label=1 → use prob_class_1
    """
    return "prob_class_0" if predictor_id == 1 else "prob_class_1"


# ── Predictor affordability curve ─────────────────────────────────────────────

def compute_predictor_affordability(
    merged: pd.DataFrame,
    pred_df: pd.DataFrame,
    predictor_id: int,
) -> pd.DataFrame:
    """
    Build an affordability curve by ranking entities using the predicted
    probability rather than the true delta.

    Parameters
    ----------
    merged : DataFrame with columns no_search_score, force_search_score
             (first MAX_SAMPLES rows already applied).
    pred_df : predictions CSV with columns:
              original_row_index, prediction, prob_class_0, prob_class_1
    predictor_id : 1, 2, or 3

    Returns
    -------
    DataFrame with columns k, net_gain, optimized_score, baseline_score.
    """
    prob_col = _prob_col_for_predictor(predictor_id)
    if prob_col not in pred_df.columns:
        raise ValueError(
            f"Column '{prob_col}' not found in predictions CSV. "
            f"Available columns: {list(pred_df.columns)}"
        )

    # Align predictions with merged rows via original_row_index
    pred_aligned = pred_df.set_index("original_row_index")

    # Keep only indices present in both
    valid_idx = pred_aligned.index.intersection(merged.index)
    if len(valid_idx) == 0:
        raise ValueError("No overlapping indices between predictions and merged scores.")

    merged_sub = merged.loc[valid_idx].copy()
    probs      = pred_aligned.loc[valid_idx, prob_col].values

    n              = len(merged_sub)
    baseline_score = merged["no_search_score"].mean()   # over full merged, consistent with ideal
    no_scores      = merged["no_search_score"].values.copy().astype(float)
    fs_scores      = merged["force_search_score"].values.copy().astype(float)

    # Sort by probability descending (most likely to need search first)
    sort_order     = np.argsort(-probs)
    sorted_idx     = valid_idx[sort_order]              # original_row_index values, sorted
    sorted_fs      = merged_sub["force_search_score"].values[sort_order]
    sorted_ns      = merged_sub["no_search_score"].values[sort_order]

    # Build curve: for k entities with highest probability, swap to force_search_score
    running_scores = no_scores.copy()
    rows = [{"k": 0, "net_gain": 0.0, "optimized_score": float(running_scores.mean())}]

    for k in range(1, n + 1):
        pos = sorted_idx[k - 1]          # position in the original merged index
        running_scores[pos] = sorted_fs[k - 1]
        gain = float(running_scores.mean()) - baseline_score
        rows.append({
            "k":               k,
            "net_gain":        round(gain, 8),
            "optimized_score": round(float(running_scores.mean()), 8),
        })

    df = pd.DataFrame(rows)
    df["baseline_score"] = round(baseline_score, 6)
    return df


# ── Budget-aware v2 experimental points ──────────────────────────────────────

def _load_budget_aware_v2_points(
    model_name: str,
    results_base_dir: str,
    no_search_df: pd.DataFrame,
    force_search_df: pd.DataFrame,
    csv_suffix: str = "",
) -> list:
    """
    Read budget-aware-v2 CSVs and return a list of (plot_pct, net_gain) tuples.
    Mirrors the experimental-points logic in new_analysis.plot_affordability.
    """
    baseline_score = no_search_df["score"].values.astype(float).mean()
    total_n        = len(no_search_df)

    force_scores = force_search_df["score"].values.copy().astype(float)
    nan_force    = np.isnan(force_scores)
    if nan_force.any():
        force_scores[nan_force] = no_search_df["score"].values[nan_force]

    points = []
    for cost in COST_SETUPS:
        cost_tag = str(int(cost))
        csv_path = (
            f"{results_base_dir}/tool-cost-{cost_tag}-budget-aware-v2/"
            f"vllm_{model_name}_with_search_summary{csv_suffix}.csv"
        )
        if not os.path.exists(csv_path):
            continue
        try:
            exp_df = na.load_csv(csv_path)
            if "search_called" not in exp_df.columns:
                continue

            k_budget   = total_n if cost == 0 else int(BUDGET_TOTAL / cost)
            k_actual   = int(exp_df["search_called"].sum())
            search_mask = exp_df["search_called"].astype(bool).values

            # Cap search calls to budget
            if search_mask.sum() > k_budget:
                true_idx    = np.where(search_mask)[0]
                capped      = np.zeros_like(search_mask, dtype=bool)
                capped[true_idx[:k_budget]] = True
                search_mask = capped

            merged_s = no_search_df["score"].values.copy().astype(float)
            merged_s[search_mask] = force_scores[search_mask]
            net_gain = float(merged_s.mean()) - baseline_score

            plot_pct = 100.0 if cost == 0 else min(BUDGET_TOTAL / cost / total_n * 100, 100.0)
            points.append((plot_pct, net_gain))

        except Exception as e:
            print(f"  WARNING [{model_name}] cost={cost}: {e}")

    points.sort(key=lambda x: x[0])
    return points


# ── Combined plot ─────────────────────────────────────────────────────────────

def plot_combined_affordability(
    model_names: list,
    task: str,
    predictor_id: int,
    output_path: str,
):
    """
    For each model plot:
      - Ideal affordability curve (solid line)
      - Budget-aware v2 experimental points (dashed line + circle markers)
      - Predictor affordability curve (dotted line + triangle markers)
    """
    # Apply task config to new_analysis globals so its functions work correctly
    na._apply_task_config(task)

    task_cfg    = bla.TASKS[task]
    results_dir = task_cfg["results_dir"]
    main_dir    = os.path.join(results_dir, "main")
    key         = task_cfg["key"]
    suffix      = task_cfg["score_suffix"]
    csv_suffix  = na.CSV_SUFFIX           # set by _apply_task_config

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")

    for model_idx, model_name in enumerate(model_names):
        color = PALETTE[model_idx % len(PALETTE)]
        legend_name = na.MODEL_IN_LEGEND.get(model_name, model_name)

        # ── Load raw score CSVs ───────────────────────────────────────────────
        ns_path = os.path.join(main_dir, f"vllm_{model_name}_no_search{suffix}_summary.csv")
        fs_path = os.path.join(main_dir, f"vllm_{model_name}_force_search{suffix}_summary.csv")

        try:
            no_search_df    = na.load_csv(ns_path)
            force_search_df = na.load_csv(fs_path)
        except FileNotFoundError as e:
            print(f"  SKIP {model_name}: {e}")
            continue

        no_search_df["score"]    = no_search_df["score"].fillna(0)
        force_search_df["score"] = force_search_df["score"].fillna(0)

        # ── Build merged df (same logic as best_layer_analysis.compute_scores) ─
        merged = (
            no_search_df[[key, "score"]].rename(columns={"score": "no_search_score"})
            .merge(
                force_search_df[[key, "score"]].rename(columns={"score": "force_search_score"}),
                on=key, how="inner",
            )
        ).iloc[:bla.MAX_SAMPLES].reset_index(drop=True)
        merged[["no_search_score", "force_search_score"]] = (
            merged[["no_search_score", "force_search_score"]].fillna(0)
        )

        total_n        = len(no_search_df)
        baseline_score = no_search_df["score"].mean()

        # ── 1. Ideal affordability curve ──────────────────────────────────────
        utility_df = na.compute_utility(force_search_df, no_search_df)
        need_df    = na.compute_need(no_search_df)
        utility_df["actual_need"] = need_df["actual_need"].values[:len(utility_df)]
        rank_df    = na.compute_rank_list(utility_df)
        afford_df  = na.compute_affordability(rank_df)

        x_k    = afford_df["k"].to_numpy(dtype=float)
        y      = afford_df["net_gain"].to_numpy(dtype=float)
        valid  = (x_k > 0) & (x_k < total_n)
        x_plot = x_k[valid] / total_n * 100
        y_plot = y[valid]

        if len(x_plot) > 0:
            order  = np.argsort(x_plot)
            x_plot = x_plot[order]
            y_plot = y_plot[order]

            right_y = y[x_k >= total_n][-1] if np.any(x_k >= total_n) else y_plot[-1]
            left_y  = y_plot[0]

            x_full = np.concatenate(([0.0], x_plot, [100.0]))
            y_full = np.concatenate(([left_y], y_plot, [right_y]))

            ax.plot(
                x_full, y_full,
                color=color, linewidth=5, linestyle="-",
                solid_capstyle="round", alpha=0.95,
                label=legend_name,
            )

        # ── 2. Budget-aware v2 experimental points ────────────────────────────
        ba_points = _load_budget_aware_v2_points(
            model_name, results_dir, no_search_df, force_search_df, csv_suffix
        )
        if ba_points:
            xs_ba, ys_ba = zip(*ba_points)
            ax.plot(xs_ba, ys_ba, linestyle="--", linewidth=4,
                    color=color, alpha=0.8, zorder=4)
            ax.scatter(xs_ba, ys_ba, s=250, color=color,
                       edgecolor="white", linewidth=2, zorder=5, alpha=0.95)

        # ── 3. Predictor affordability curve ─────────────────────────────────
        predictor_name = bla.MODEL_MAPPING.get(model_name)
        if predictor_name is None:
            print(f"  SKIP predictor for {model_name}: no MODEL_MAPPING entry")
            continue

        result = bla.load_best_predictions(predictor_name, predictor_id, task_cfg)
        if result is None:
            print(f"  SKIP predictor for {model_name}: predictions not found")
            continue

        pred_df, best_layer, best_clf, summary = result
        print(
            f"  [{model_name}] Predictor {predictor_id}: "
            f"best_layer={best_layer}, best_clf={best_clf}"
        )

        try:
            pred_afford_df = compute_predictor_affordability(merged, pred_df, predictor_id)
        except ValueError as e:
            print(f"  SKIP predictor affordability for {model_name}: {e}")
            continue

        n_pred  = len(pred_df)
        xp_k    = pred_afford_df["k"].to_numpy(dtype=float)
        yp      = pred_afford_df["net_gain"].to_numpy(dtype=float)
        xp_pct  = xp_k / total_n * 100   # convert to % of all queries

        # Plot full curve (k=0 to n_pred)
        ax.plot(
            xp_pct, yp,
            color=color, linewidth=4, linestyle="-.",
            alpha=0.95, zorder=3,
        )
        # Mark the end point (all predicted-need entities searched)
        ax.scatter(
            [xp_pct[-1]], [yp[-1]],
            s=350, color=color,
            marker="D", edgecolor="white", linewidth=2,
            zorder=6, alpha=0.95,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()

    ideal_handle = Line2D([0], [0], color="black", linewidth=4, linestyle="-")
    ba_handle    = Line2D(
        [0], [0], color="black", linewidth=3, linestyle="--",
        marker="o", markersize=9, markerfacecolor="black",
        markeredgecolor="white", markeredgewidth=1.5,
    )
    pred_handle  = Line2D(
        [0], [0], color="black", linewidth=3, linestyle="-.",
        marker="*", markersize=9, markerfacecolor="black",
        markeredgecolor="white", markeredgewidth=1.5,
    )
    pred_label = PREDICTOR_LEGEND_NAMES.get(predictor_id, f"Predictor {predictor_id}")
    handles += [ideal_handle, ba_handle, pred_handle]
    labels  += [
        "Ideal curve",
        "Cost Description",
        pred_label,
    ]

    ax.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=3, frameon=False, fontsize=18,
        columnspacing=1.2, handletextpad=0.5,
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-2, 105)
    tick_positions = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{p}%" for p in tick_positions], fontsize=20)
    ax.set_xlabel("% Queries with Tool Calls", fontsize=25, fontweight="bold", labelpad=12)
    ax.set_ylabel("Utility Gain over No-Tool",  fontsize=25, fontweight="bold")
    ax.tick_params(axis="y", labelsize=22)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(top=0.78)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Combined affordability figure saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ideal + budget-aware v2 + predictor affordability curves."
    )
    parser.add_argument(
        "--task", default=DEFAULT_TASK,
        choices=list(bla.TASKS),
        help="Task to analyse (default: entity).",
    )
    parser.add_argument(
        "--predictor", type=int, default=DEFAULT_PREDICTOR,
        choices=[1, 2, 3],
        help=(
            "Predictor ID to plot (default: 1). "
            "predictor_1 ranks by prob_class_0; predictor_2/3 by prob_class_1."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_NAMES,
        help="Model name keys (matching filenames). Default: all six models.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output PDF path. Defaults to "
            "<results>/predictor_best_layer/<task>/"
            "figures/predictor_affordability_combined_p<N>.pdf"
        ),
    )
    args = parser.parse_args()

    predictor_ids = [1, 2, 3] if args.output is None else [args.predictor]

    for pid in predictor_ids:
        if args.output is None:
            out_dir  = os.path.join(bla.OUTPUT_ROOT, args.task, "figures")
            out_path = os.path.join(
                out_dir,
                f"predictor_affordability_combined_p{pid}.pdf",
            )
        else:
            out_path = args.output

        plot_combined_affordability(
            model_names  = args.models,
            task         = args.task,
            predictor_id = pid,
            output_path  = out_path,
        )
