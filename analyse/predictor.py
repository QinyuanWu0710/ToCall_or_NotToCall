import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import re
import glob


def get_matching_prediction_files(model_name, predictor_id, clf, task):
    if task == 'entity':
        base_dir = (
            f"/NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/"
            f"{model_name}/predictor_{predictor_id}"
        )
    elif task == 'bfcl':
        base_dir = (
            f"/NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/bfcl/"
            f"{model_name}/predictor_{predictor_id}"
        )        
    elif task == 'invivo':
        base_dir = (
            f"/NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/invivo/"
            f"{model_name}/predictor_{predictor_id}"
        )        

    pattern = os.path.join(base_dir, f"predictions_{clf}.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    return files


def get_axis_labels(predictor_id):
    label_map = {
        "1": (["Need", "No Need"], ["Need", "No Need"]),
        "2": (["Help", "No Help"], ["Help", "No Help"]),
        "3": (["Help", "No Help"], ["Help", "No Help"]),
        "4": (["Help", "No Help"], ["Help", "No Help"]),
    }

    if predictor_id not in label_map:
        raise ValueError(f"Invalid Predictor ID: {predictor_id}")

    return label_map[predictor_id]


def plot_confusion_matrix(model_name, predictor_id, clf, path, task):
    print(f"Loading: {path}")
    df = pd.read_csv(path)

    y_pred = 1 - df["prediction"].astype(int)
    y_true = 1 - df["ground_truth_label"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    # Compute column-wise percentages (percentage of predicted class)
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_pct = np.divide(
        cm.astype(float),
        col_sums,
        out=np.zeros_like(cm, dtype=float),
        where=col_sums != 0
    ) * 100

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)"

    xticklabels, yticklabels = get_axis_labels(predictor_id)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
        cbar=False,
        annot_kws={"size": 35, "weight": "bold"},
        vmin=0,
        vmax=100
    )

    ax.set_xlabel("Predicted", fontsize=25, labelpad=16, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Actual", fontsize=25, labelpad=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20)

    # Optional title
    # ax.set_title(
    #     f"{model_name} | predictor_{predictor_id} | {clf} | layer {layer}\n"
    #     f"Acc={acc:.3f}  Prec={precision:.3f}  Rec={recall:.3f}  F1={f1:.3f}",
    #     fontsize=11,
    #     pad=14,
    # )

    plt.tight_layout()

    out_dir = '/NS/chatgpt/work/qwu/hallucinations_detection/results/predictor'
    out_path = os.path.join(
        out_dir,
        f"{task}_{predictor_id}_{model_name}_{clf}_confusion_matrix.pdf"
    )
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot confusion matrix for predictions.")
    parser.add_argument("--model_name", required=True, help="Model name, e.g. gpt-4")
    parser.add_argument("--predictor_id", required=True, help="Predictor ID, e.g. 1")
    parser.add_argument("--clf", required=True, default="LogisticRegression", help="Which classifier")
    parser.add_argument("--task", required=True, default="entity", help="Which task")

    args = parser.parse_args()

    try:
        matching_files = get_matching_prediction_files(args.model_name, args.predictor_id, args.clf, args.task)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print(f"Found {len(matching_files)} matching file(s).")
    for file_path in matching_files:
        plot_confusion_matrix(args.model_name, args.predictor_id, args.clf, file_path, args.task)