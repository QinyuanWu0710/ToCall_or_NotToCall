import json
import pandas as pd
from pathlib import Path

jsonl_path = Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/openai_gpt-4.1-mini_with_search_auto_20260209_171454.jsonl")
csv_path   = Path("/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/user_unique_entities_sampled_500_with_correctness_score.csv")

# --- Load JSONL: entity -> did_search ---
rows = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        rows.append({
            "entity_text": item.get("entity"),
            "auto_did_search": item.get("did_search"),
        })

jsonl_df = pd.DataFrame(rows)

# If the JSONL contains duplicates for the same entity, decide how to collapse:
# Here: if ANY instance did_search==True, mark True.
jsonl_df = (
    jsonl_df.dropna(subset=["entity_text"])
            .groupby("entity_text", as_index=False)["auto_did_search"]
            .max()
)

# --- Load CSV and merge ---
df = pd.read_csv(csv_path)
if "entity_text" not in df.columns:
    raise ValueError(f"'entity_text' column not found in CSV. Found columns: {list(df.columns)}")

df2 = df.merge(jsonl_df, on="entity_text", how="left")

# Optional: treat missing auto_did_search as False (choose what you want)
# df2["auto_did_search"] = df2["auto_did_search"].fillna(False)

# --- Save updated CSV ---
out_path = csv_path.with_name(csv_path.stem + "_with_auto_did_search.csv")
df2.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

# --- Compute means for all *_correctness columns by auto_did_search ---
correct_cols = [c for c in df2.columns if c.endswith("_correctness")]
if not correct_cols:
    raise ValueError("No columns ending with '_correctness' were found.")

summary = (
    df2.groupby("auto_did_search")[correct_cols]
       .mean(numeric_only=True)
       .sort_index()
)

# Also show counts
counts = df2["auto_did_search"].value_counts(dropna=False).sort_index()
print("\nCounts by auto_did_search:")
print(counts)

print("\nAverage correctness by auto_did_search:")
print(summary)

# If you want a tidy (long) table:
# tidy = summary.reset_index().melt(id_vars="auto_did_search", var_name="metric", value_name="mean")
# print("\nTidy summary:")
# print(tidy.sort_values(["metric","auto_did_search"]))
