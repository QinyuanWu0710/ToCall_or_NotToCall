import json
import pandas as pd
import numpy as np
from pathlib import Path

# Inputs
result_jsonl = "/NS/chatgpt/work/qwu/hallucinations_detection/results/factual_score/entity_hallucination/qwen3-4b-instruct_no-search_temp0/scores.jsonl"
new_column = "qwen3-4b-instruct_temp0_no-search_correctness"
csv_file = "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/user_unique_data_sampled_500_refine_factuality.csv"

# Outputs
out_csv = csv_file.replace(".csv", ".csv")


# ----------------------------
# Load JSONL and build entity -> score mapping
# ----------------------------
rows = []
with open(result_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj_ = json.loads(line)
        obj = obj_.get("result")
        # robust extraction
        ent = obj.get("entity")
        score = obj.get("correctness_score")
        if ent is None:
            continue
        try:
            score = float(score) if score is not None else np.nan
        except Exception:
            score = np.nan
        rows.append((ent, score))

jsonl_df = pd.DataFrame(rows, columns=["entity", f"{new_column}"])

# If the same entity appears multiple times, average its score
entity_scores = (
    jsonl_df.groupby("entity", as_index=False)[f"{new_column}"]
    .mean()
)

# ----------------------------
# Load CSV and merge
# ----------------------------
df = pd.read_csv(csv_file)

if "entity_text" not in df.columns:
    raise KeyError(f"CSV is missing required column 'entity_text'. Found columns: {list(df.columns)}")

if new_column in df.columns:
    df = df.drop(columns=[new_column])

df_merged = df.merge(
    entity_scores,
    how="left",
    left_on="entity_text",
    right_on="entity",
)

# Drop the helper join column "entity" (keep entity_text)
if "entity" in df_merged.columns:
    df_merged = df_merged.drop(columns=["entity"])

# Save updated CSV
df_merged.to_csv(out_csv, index=False)
print(f"✅ Wrote updated CSV: {out_csv}")

# ----------------------------
# Report metrics
# ----------------------------
valid_scores = df_merged[f"{new_column}"].dropna()
overall_avg = valid_scores.mean() if len(valid_scores) else np.nan
print("\n=== Overall ===")
print(f"Average correctness_score (non-null): {overall_avg:.4f}")
print(f"Count non-null: {len(valid_scores)} / {len(df_merged)}")

if "category" not in df_merged.columns:
    raise KeyError(f"CSV is missing required column 'category'. Found columns: {list(df_merged.columns)}")

print("\n=== By category (distribution summary) ===")
print(f"\n=== Summary: {new_column}, mean: {df_merged[f'{new_column}'].mean()}")

dist = (
    df_merged.groupby("category")[f"{new_column}"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        q25=lambda s: s.quantile(0.25),
        median="median",
        q75=lambda s: s.quantile(0.75),
        max="max",
    )
    .sort_values(["count", "mean"], ascending=[False, False])
)
print(dist.to_string(float_format=lambda x: f"{x:.4f}"))

# Optional: also show counts including NaNs per category
print("\n=== By category (missingness) ===")
missing = df_merged.groupby("category")[f"{new_column}"].apply(lambda s: s.isna().sum()).rename("num_missing")
total = df_merged.groupby("category")[f"{new_column}"].size().rename("total")
missing_table = pd.concat([total, missing], axis=1)
missing_table["missing_rate"] = missing_table["num_missing"] / missing_table["total"]
print(missing_table.sort_values("missing_rate", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

# Aggregate mean and std per category
summary_df = (
    df_merged
    .dropna(subset=[new_column])
    .groupby("category")[new_column]
    .agg(mean="mean", std="std")
    .reset_index()
)
