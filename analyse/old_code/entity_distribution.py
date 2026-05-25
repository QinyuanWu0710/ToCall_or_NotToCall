import json
import pandas as pd
import plotly.express as px
from pathlib import Path

# paths
csv_file = Path(
    "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/user_unique_entities_sampled_500.csv"
)
jsonl_file = Path(
    "/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/"
    "openai_gpt-4.1-mini_with_search_auto_20260209_171454.jsonl"
)

out_csv_png = Path("csv_entity_category_distribution.png")
out_ds_png = Path("did_search_entity_category_distribution.png")

# ======================
# Load CSV
# ======================
df = pd.read_csv(csv_file)

assert {"entity_text", "category"}.issubset(df.columns)

df["entity_text_norm"] = df["entity_text"].astype(str).str.strip()

# ======================
# 1) CSV: 500 entities category distribution
# ======================
csv_counts = (
    df.groupby("category", dropna=False)
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

fig_csv = px.bar(
    csv_counts,
    x="category",
    y="count",
    text="count",
    title="Category distribution (500 sampled entities)"
)
fig_csv.update_layout(
    xaxis_title="Category",
    yaxis_title="Count",
    xaxis_tickangle=-35
)

fig_csv.write_image(out_csv_png)
print(f"Saved: {out_csv_png}")

# ======================
# 2) JSONL: did_search == True
# ======================
did_search_entities = []

with jsonl_file.open() as f:
    for line in f:
        item = json.loads(line)
        if item.get("did_search") is True and item.get("entity") is not None:
            did_search_entities.append(str(item["entity"]).strip())

ds_df = pd.DataFrame({"entity_text_norm": did_search_entities})

# join with CSV to get categories
joined = ds_df.merge(
    df[["entity_text_norm", "category"]],
    on="entity_text_norm",
    how="left"
)

matched = joined["category"].notna().sum()
unmatched = joined["category"].isna().sum()

ds_counts = (
    joined.groupby("category", dropna=False)
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
)

ds_counts["category"] = ds_counts["category"].fillna("UNMATCHED_TO_CSV")

fig_ds = px.bar(
    ds_counts,
    x="category",
    y="count",
    text="count",
    title=f"did_search == True entity category distribution "
          f"(matched={matched}, unmatched={unmatched})"
)
fig_ds.update_layout(
    xaxis_title="Category",
    yaxis_title="Count",
    xaxis_tickangle=-35
)

fig_ds.write_image(out_ds_png)
print(f"Saved: {out_ds_png}")
