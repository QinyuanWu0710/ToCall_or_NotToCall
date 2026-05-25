import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import glob
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_LIST = [
    "openai_gpt-oss-120b",
    "Qwen_Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "google_gemma-3-27b-it",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503",
    "meta-llama_Llama-3.2-3B-Instruct",
]

MODEL_IN_LEGEND = {
    "google_gemma-3-27b-it": "Gemma3-27B-IT",
    "openai_gpt-oss-120b": "GPT-OSS-120B",
    "Qwen_Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen-3-30B-IT",
    "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral3.1-24B-IT",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama3.2-3B-IT",
}

# ── Per-run settings ───────────────────────────────────────────────────────────
CSV_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination"
CSV_FILES = glob.glob(os.path.join(CSV_DIR, "*.csv"))  # adjust pattern as needed

SCORE_COLUMN = "score"
FIGURE_TITLE = "Factuality Distribution"
OUTPUT_DIR   = "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/figures"


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_model_name(csv_path: str) -> str:
    """Derive a display name from the CSV filename."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return MODEL_IN_LEGEND.get(stem, stem)


def load_scores(csv_path: str, score_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if score_col not in df.columns:
        raise ValueError(
            f"Column '{score_col}' not found in {csv_path}.\n"
            f"Available: {list(df.columns)}"
        )
    return df[score_col].fillna(0).values


def plot_histograms(data: dict, title: str, output_path: str | None):
    names  = list(data.keys())
    scores = list(data.values())
    n      = len(names)
    palette = plt.cm.tab10.colors

    # Build a consistent color map by base model name
    base_models = []
    for name in names:
        base_name = name.rsplit(" (", 1)[0]   # strips " (no-search)" / " (force-search)"
        if base_name not in base_models:
            base_models.append(base_name)

    color_map = {
        model: palette[i % len(palette)]
        for i, model in enumerate(base_models)
    }

    ncols = min(n,4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    # fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for i, (name, arr) in enumerate(zip(names, scores)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        base_name = name.rsplit(" (", 1)[0]
        color = color_map[base_name]

        mn  = arr.mean()
        med = np.median(arr)

        ax.hist(arr, bins=30, color=color, alpha=0.65, edgecolor="white", linewidth=0.5)
        ax.axvline(mn,  color=color,     linestyle="--", linewidth=1.8, label=f"mean={mn:.3f}")
        ax.axvline(med, color="dimgray", linestyle=":",  linewidth=1.8, label=f"median={med:.3f}")
        ax.axvline(0.1, color="black",   linestyle="-",  linewidth=1.2, alpha=0.85)
        ax.axvline(0.9, color="black",   linestyle="-",  linewidth=1.2, alpha=0.85)

        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_xlabel("Facuality", fontsize=15)
        ax.set_ylabel("Count", fontsize=15)

        ax.legend(fontsize=15, framealpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    data = {}
    TASK = 'bfcl'
    for model_name in MODEL_LIST:
        if TASK == 'entity':
            csv_files = [
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main/vllm_{model_name}_no_search_summary.csv",
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main/vllm_{model_name}_force_search_summary.csv",
            ]
        elif TASK == 'invivol':
            csv_files = [
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main/vllm_{model_name}_no_search_summary.csv",
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/main/vllm_{model_name}_force_search_summary.csv",
            ]
        elif TASK == 'bfcl':
            csv_files = [
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main/vllm_{model_name}_no_search_summary.csv",
                f"/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main/vllm_{model_name}_force_search_summary.csv",
            ]
        display_name = MODEL_IN_LEGEND[model_name]
        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                print(f"Warning: file not found, skipping — {csv_path}")
                continue
            # Distinguish the two variants in the legend
            variant = "no-search" if "no_search" in csv_path else "force-search"
            if variant == "no-search":
                pad = "No Tool"
            else:
                pad = "With Tool"
            label = f"{display_name} ({pad})"
            arr = load_scores(csv_path, SCORE_COLUMN)
            print(f"{label}: n={len(arr)}, mean={arr.mean():.3f}, median={np.median(arr):.3f}")
            data[label] = arr

    if not data:
        raise FileNotFoundError("No CSV files were loaded. Check paths and MODEL_LIST.")

    combined_output = os.path.join(OUTPUT_DIR, f"{TASK}_all_models_histogram_scores.pdf")
    plot_histograms(data, FIGURE_TITLE, combined_output)