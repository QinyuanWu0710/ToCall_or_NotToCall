# '''
# Compare different models' performance under different setups

# no search score: /NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main/vllm_{model_name}_no_search_summary.csv
# force search score: /NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main/vllm_{model_name}_force_search_summary.csv

# auto call: /NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main/vllm_{model_name}_with_search_summary.csv
# perceived call: /NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/main-perceived-need/vllm_{model_name}_with_search_summary.csv
# predict_1: /NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/{model_name}/predictor_1/predictions_MLP.csv
# predict_2: /NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/{model_name}/predictor_2/predictions_MLP.csv
# predict_3: /NS/chatgpt/work/qwu/hallucinations_detection/data/tool_predictor/{model_name}/predictor_3/predictions_MLP.csv


# We want numbers in total:
# 1. no search score average: average the no search 'score' column
# 2. force search score average: average the force search 'score' column
# 3. oracle score: per-entity max(no_search, force_search), then averaged
# 4. auto call score: check the 'search_called' column in auto call file, if it's true, use force score for that sample, if it's false, use no search score for that sample, then compute the average
# 6. predictor_* score: check the prediction column, then 1 is using force_search score and 0 is using no search score

# All the csv files should have the same 500 entities in total.
# '''


import os
import pandas as pd
import numpy as np

BASE_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection"
PREDICTOR_DIR = os.path.join(BASE_DIR, "data/tool_predictor")

# Task configurations
TASKS = {
    "entity": dict(
        results_dir     = os.path.join(BASE_DIR, "results/entity_hallucination/temp=0"),
        predictor_subdir= "",       # predictor lives at PREDICTOR_DIR/{predictor_name}/
        key             = "entity",
    ),
    # "bfcl": dict(
    #     results_dir     = os.path.join(BASE_DIR, "results/bfcl_raw/tool_result"),
    #     predictor_subdir= "bfcl",   # predictor lives at PREDICTOR_DIR/bfcl/{predictor_name}/
    #     key             = "query",
    # ),
    # "invivo": dict(
    #     results_dir     = os.path.join(BASE_DIR, "results/real_query/temp=0"),
    #     predictor_subdir= "invivo",  # predictor lives at PREDICTOR_DIR/invivo/{predictor_name}/
    #     key             = "entity",
    # ),
    # "perplexity": dict(
    #     results_dir     = os.path.join(BASE_DIR, "results/perplexity/temp=0"),
    #     predictor_subdir= "perplexity",  # predictor lives at PREDICTOR_DIR/invivo/{predictor_name}/
    #     key             = "entity",
    # ),
}

# summary CSV model names -> predictor model names
MODEL_MAPPING = {
    # "google_gemma-3-27b-it": "gemma-3-27b-it",
    "openai_gpt-oss-120b": "gpt-oss-120b",
    # "Qwen_Qwen3-30B-A3B": "Qwen3-30B-A3B",
    # "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instruct-2507",
    # "mistralai_Mistral-Small-3.1-24B-Instruct-2503": "Mistral-Small-3.1-24B-Instruct-2503",
    # "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
}


def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_scores(model_name, predictor_name, task_cfg):
    results_dir      = task_cfg["results_dir"]
    predictor_subdir = task_cfg["predictor_subdir"]
    key              = task_cfg["key"]

    main_dir = os.path.join(results_dir, "main")
    pred_dir = (
        os.path.join(PREDICTOR_DIR, predictor_subdir, predictor_name)
        if predictor_subdir
        else os.path.join(PREDICTOR_DIR, predictor_name)
    )

    no_search   = load_csv(os.path.join(main_dir, f"vllm_{model_name}_no_search_summary.csv"))
    force_search = load_csv(os.path.join(main_dir, f"vllm_{model_name}_force_search_summary.csv"))
    auto_call   = load_csv(os.path.join(main_dir, f"vllm_{model_name}_with_search_summary.csv"))
    # no_search   = load_csv(os.path.join(main_dir, f"vllm_{model_name}_no_search_summary_gpt-5.2.csv"))
    # force_search = load_csv(os.path.join(main_dir, f"vllm_{model_name}_force_search_summary_gpt-5.2.csv"))
    # auto_call   = load_csv(os.path.join(main_dir, f"vllm_{model_name}_with_search_summary_gpt-5.2.csv"))

    if no_search is None or force_search is None:
        return None

    # Merge on key to align rows
    merged = no_search[[key, "score"]].rename(columns={"score": "no_search_score"})
    merged = merged.merge(
        force_search[[key, "score"]].rename(columns={"score": "force_search_score"}),
        on=key, how="inner"
    )
    merged[["no_search_score", "force_search_score"]] = merged[["no_search_score", "force_search_score"]].fillna(0)

    n = len(merged)
    scores = {}
    force_counts = {}

    # 1. No search score (never uses force)
    scores["no_search"] = merged["no_search_score"].mean()
    force_counts["no_search"] = 0

    # 2. Force search score (always uses force)
    scores["force_search"] = merged["force_search_score"].mean()
    force_counts["force_search"] = n

    # 3. Oracle score: per-entity max(no_search, force_search)
    scores["oracle"] = merged[["no_search_score", "force_search_score"]].max(axis=1).mean()
    force_counts["oracle"] = int((merged["force_search_score"] > merged["no_search_score"]).sum())

    # 4. Auto call score
    if auto_call is not None:
        df = merged.merge(auto_call[[key, "search_called"]], on=key, how="inner")
        df["auto_score"] = np.where(df["search_called"], df["force_search_score"], df["no_search_score"])
        scores["auto_call"] = df["auto_score"].mean()
        force_counts["auto_call"] = int(df["search_called"].sum())
    else:
        scores["auto_call"] = None
        force_counts["auto_call"] = None

    # 5. Predictor scores
    for i in range(1, 4):
        pred_path = os.path.join(pred_dir, f"predictor_{i}", "predictions_MLP.csv")
        pred_df = load_csv(pred_path)
        k = f"predict_{i}"
        if pred_df is not None:
            pred_df = pred_df.set_index("original_row_index")["prediction"]
            merged_subset = merged.iloc[pred_df.index]
            use_force = pred_df.values == (0 if i == 1 else 1)
            pred_score = np.where(use_force, merged_subset["force_search_score"].values, merged_subset["no_search_score"].values)
            scores[k] = pred_score.mean()
            force_counts[k] = int(use_force.sum())
        else:
            scores[k] = None
            force_counts[k] = None

    return scores, force_counts


def run_task(task_name, task_cfg):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    score_rows = []
    count_rows = []
    for model_name, predictor_name in MODEL_MAPPING.items():
        result = compute_scores(model_name, predictor_name, task_cfg)
        if result is not None:
            scores, force_counts = result
            score_rows.append({"model": model_name, **scores})
            count_rows.append({"model": model_name, **force_counts})
        else:
            print(f"  Skipping {model_name} (missing files)")

    if not score_rows:
        print("  No results found.")
        return

    df_scores = pd.DataFrame(score_rows).set_index("model")
    df_counts = pd.DataFrame(count_rows).set_index("model")

    def fmt(score, count):
        if score is None:
            return "N/A"
        s = f"{score:.4f}"
        if count is not None:
            s += f" ({count})"
        return s

    cols = list(df_scores.columns)
    display = pd.DataFrame(index=df_scores.index, columns=cols)
    for col in cols:
        display[col] = [fmt(df_scores.loc[m, col], df_counts.loc[m, col]) for m in df_scores.index]

    print(display.to_string())

    out_dir = os.path.join(task_cfg["results_dir"], "compare")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_summary.csv")
    df_scores[cols].to_csv(out_path)
    print(f"\nSaved to {out_path}")


def main():
    for task_name, task_cfg in TASKS.items():
        run_task(task_name, task_cfg)


if __name__ == "__main__":
    main()
