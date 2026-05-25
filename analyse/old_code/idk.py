"""
Analysis script for comparing entity hallucination results.
Compares models with and without web search, generates statistics and visualizations.
"""
import os
import re
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = '/NS/chatgpt/work/qwu/hallucinations_detection/code/eval/entity_knowledge/entity_hallucination_with_fastMCP/results'
exp_name = 'vllm__NS_factual-knowledge-and-hallucination_nobackup_qwu_llm_base_model_gpt-oss-20b_with_search_20260223_001940'
file_path = os.path.join(BASE_DIR, f'{exp_name}.jsonl')

entity_file = '/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/user_unique_entities.csv'

output_dir = '/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/figures'


# -----------------------------
# Helpers
# -----------------------------
REFUSAL_PATTERNS = [
    r"\bi['’]m sorry\b",
    r"\bi['’]?m unable to\b",
    r"\bi can['’]?t provide\b",
    r"\bi cannot provide\b",
    r"\bi don['’]?t know\b",
    r"\bi do not know\b",
    r"\bi don['’]?t have any information\b",
    r"\bi do not have any information\b",
    r"\bno information\b",
    r"\bi am not aware\b"
]

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)

def is_refusal(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(REFUSAL_RE.search(text))

def normalize_entity(s: str) -> str:
    return (s or "").strip()

def entity_in_response(entity: str, response: str) -> bool:
    """
    Check whether the entity appears in the response.
    Uses a regex word-boundary-ish match, but allows entities with spaces/punct.
    Falls back to substring check if regex compilation fails.
    """
    if not isinstance(entity, str) or not isinstance(response, str):
        return False
    ent = entity.strip()
    if not ent:
        return False

    try:
        pattern = re.compile(r"(?i)(?<!\w)" + re.escape(ent) + r"(?!\w)")
        if pattern.search(response):
            return True
    except re.error:
        pass

    return ent.lower() in response.lower()


# -----------------------------
# Load experiment jsonl
# -----------------------------
rows = []
with open(file_path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decode error at line {line_no}: {e}")

df = pd.DataFrame(rows)

required_cols = {"entity", "response"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in jsonl: {missing}")

df["entity"] = df["entity"].astype(str)
df["response"] = df["response"].astype(str)

# -----------------------------
# NEW: Rate limit error detection from augmented_prompt
# -----------------------------
RATE_LIMIT_SUBSTR = "Error: Rate limit exceeded"  # NEW

# True if augmented_prompt exists AND contains the rate limit error string
df["is_rate_limit_error"] = (
    df.get("augmented_prompt")  # returns None if column doesn't exist
      .apply(lambda x: isinstance(x, str) and RATE_LIMIT_SUBSTR in x)
    if "augmented_prompt" in df.columns
    else False
)  # NEW

n_rate_limit_errors = int(df["is_rate_limit_error"].sum())  # NEW

# NEW: only use non-error items for refusal/entity-not-mentioned metrics
df_eval = df[~df["is_rate_limit_error"]].copy()  # NEW


# (1) refusal-like (computed on df_eval only)  # CHANGED
df_eval["is_refusal"] = df_eval["response"].apply(is_refusal)

# (2) entity not mentioned (computed on df_eval only)  # CHANGED
df_eval["entity_mentioned"] = df_eval.apply(
    lambda r: entity_in_response(r["entity"], r["response"]), axis=1
)
df_eval["entity_not_mentioned"] = ~df_eval["entity_mentioned"]


# -----------------------------
# Load entity categories
# -----------------------------
df_entity = pd.read_csv(entity_file)

if "entity_text" not in df_entity.columns or "category" not in df_entity.columns:
    raise ValueError("entity_file must contain columns: 'entity_text' and 'category'")

df_entity["entity_text"] = df_entity["entity_text"].astype(str).map(normalize_entity)
df_entity["category"] = df_entity["category"].astype(str).fillna("Unknown")

entity_to_cat = (
    df_entity.drop_duplicates(subset=["entity_text"])
             .set_index("entity_text")["category"]
             .to_dict()
)

# Map category onto df_eval (and df if you still want it there)  # CHANGED
df_eval["category"] = df_eval["entity"].map(entity_to_cat).fillna("Unknown")


# -----------------------------
# Aggregations by category (exclude rate-limit error items)
# -----------------------------
grp = df_eval.groupby("category", dropna=False)  # CHANGED
summary = grp.agg(
    n_items=("entity", "size"),
    n_refusal=("is_refusal", "sum"),
    n_entity_not_mentioned=("entity_not_mentioned", "sum"),
).reset_index()

summary["pct_refusal"] = (summary["n_refusal"] / summary["n_items"] * 100).round(2)
summary["pct_entity_not_mentioned"] = (summary["n_entity_not_mentioned"] / summary["n_items"] * 100).round(2)

summary = summary.sort_values("n_items", ascending=False).reset_index(drop=True)

print("\n=== Overall ===")
print(f"Total items (all): {len(df)}")
print(f"Rate limit errors (augmented_prompt contains '{RATE_LIMIT_SUBSTR}'): {n_rate_limit_errors}")  # NEW
print(f"Total items (excluding rate limit errors): {len(df_eval)}")  # NEW
print(f"Refusals (excluding rate limit errors): {df_eval['is_refusal'].sum()} ({df_eval['is_refusal'].mean()*100:.2f}%)")
print(f"Entity not mentioned (excluding rate limit errors): {df_eval['entity_not_mentioned'].sum()} ({df_eval['entity_not_mentioned'].mean()*100:.2f}%)")

print("\n=== By category (excluding rate limit errors) ===")
print(summary)


# -----------------------------
# Plotting
# -----------------------------
def plot_counts_and_pcts(
    summary_df: pd.DataFrame,
    count_col: str,
    pct_col: str,
    title: str,
    out_name: str,
    output_dir: str,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_plot = summary_df[["category", count_col, pct_col]].copy()
    df_plot["label"] = df_plot.apply(
        lambda r: f"{int(r[count_col])} ({r[pct_col]:.2f}%)", axis=1
    )

    fig_html = px.bar(
        df_plot,
        x="category",
        y=count_col,
        text="label",
        title=title,
    )
    fig_html.update_traces(textposition="outside")
    fig_html.update_layout(xaxis_tickangle=-45)

    html_path = os.path.join(output_dir, f"{out_name}.html")
    fig_html.write_html(html_path, include_plotlyjs="cdn")
    print(f"[Saved] {html_path}")


plot_counts_and_pcts(
    summary,
    count_col="n_refusal",
    pct_col="pct_refusal",
    title="Refusal-like responses by entity category (excluding rate limit errors)",
    out_name=f"{exp_name}__refusal_by_category",
    output_dir=output_dir,
)

plot_counts_and_pcts(
    summary,
    count_col="n_entity_not_mentioned",
    pct_col="pct_entity_not_mentioned",
    title="Entity not mentioned in response by entity category (excluding rate limit errors)",
    out_name=f"{exp_name}__entity_not_mentioned_by_category",
    output_dir=output_dir,
)
