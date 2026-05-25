#!/usr/bin/env python3
"""
Dot plot + category-level bar plot of entity factuality.

Produces a single HTML file containing:
  1) Entity-level dot plot
  2) Category-level grouped bar plot
"""

import argparse
import os
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# Keep "_correctness" in SUFFIXES
SUFFIXES = [
    "temp0_no-search_correctness",
    "temp0_auto-search_correctness",
    "temp0_force-search_correctness",
]
SUFFIX_LABELS = {
    "temp0_no-search_correctness": "no-search",
    "temp0_auto-search_correctness": "auto-search",
    "temp0_force-search_correctness": "force-search",
}


def _strip_correctness(suffix: str) -> str:
    return suffix[:-len("_correctness")] if suffix.endswith("_correctness") else suffix


def _resolve_model_cols(df_cols: List[str], model: str) -> Optional[Dict[str, str]]:
    colset = set(df_cols)
    mapping: Dict[str, str] = {}
    for s in SUFFIXES:
        col_with = f"{model}_{s}"
        col_without = f"{model}_{_strip_correctness(s)}"
        if col_with in colset:
            mapping[s] = col_with
        elif col_without in colset:
            mapping[s] = col_without
        else:
            return None
    return mapping


def detect_models(columns: List[str]) -> List[str]:
    pat = re.compile(
        r"^(?P<model>.+)_(?P<suffix>temp0_(?:no-search|auto-search|force-search)(?:_correctness)?)$"
    )

    candidates: Dict[str, set] = {}
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        model = m.group("model")
        suffix = m.group("suffix")
        candidates.setdefault(model, set()).add(suffix)

    ok_models = []
    for model in candidates:
        if _resolve_model_cols(columns, model) is not None:
            ok_models.append(model)

    return sorted(set(ok_models))


# ===============================
# ENTITY-LEVEL DOT PLOT
# ===============================
def make_dot_plot(df: pd.DataFrame, model: str):
    mapping = _resolve_model_cols(list(df.columns), model)
    if mapping is None:
        raise ValueError(f"Missing required columns for model {model}")

    d = df[["entity_text", "category"] + list(mapping.values())].copy()

    for c in mapping.values():
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["category"] = d["category"].astype(str)
    d["entity_text"] = d["entity_text"].astype(str)
    d = d.sort_values(["category", "entity_text"], kind="stable").reset_index(drop=True)

    d["x"] = np.arange(len(d))

    spans = []
    for cat, grp in d.groupby("category", sort=False):
        spans.append((cat, grp["x"].min(), grp["x"].max()))

    jitter = {
        "temp0_no-search_correctness": -0.18,
        "temp0_auto-search_correctness": 0.0,
        "temp0_force-search_correctness": 0.18,
    }

    colors = {
        "temp0_no-search_correctness": "#1f77b4",
        "temp0_auto-search_correctness": "#ff7f0e",
        "temp0_force-search_correctness": "#2ca02c",
    }

    fig = go.Figure()

    for suffix in SUFFIXES:
        col = mapping[suffix]

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "category: %{customdata[1]}<br>"
            f"{SUFFIX_LABELS[suffix]} score: %{{y:.3f}}"
            "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=d["x"] + jitter[suffix],
                y=d[col],
                mode="markers",
                name=SUFFIX_LABELS[suffix],
                marker=dict(size=7, color=colors[suffix], opacity=0.9),
                customdata=np.stack([d["entity_text"], d["category"]], axis=1),
                hovertemplate=hover,
            )
        )

    fig.update_layout(
        title=f"Entity factuality scores (model: {model})",
        xaxis=dict(
            tickmode="array",
            tickvals=d["x"],
            ticktext=d["entity_text"],
            tickangle=90,
        ),
        yaxis=dict(title="Factuality score", rangemode="tozero"),
        height=800,
        width=max(1400, 12 * len(d)),
    )

    return fig


# ===============================
# CATEGORY-LEVEL BAR PLOT
# ===============================
def make_category_bar_plot(df: pd.DataFrame, model: str):
    mapping = _resolve_model_cols(list(df.columns), model)

    d = df[["category"] + list(mapping.values())].copy()

    for c in mapping.values():
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Compute mean per category
    agg = d.groupby("category").mean(numeric_only=True).reset_index()

    colors = {
        "temp0_no-search_correctness": "#1f77b4",
        "temp0_auto-search_correctness": "#ff7f0e",
        "temp0_force-search_correctness": "#2ca02c",
    }

    fig = go.Figure()

    for suffix in SUFFIXES:
        col = mapping[suffix]
        fig.add_trace(
            go.Bar(
                x=agg["category"],
                y=agg[col],
                name=SUFFIX_LABELS[suffix],
                marker_color=colors[suffix],
            )
        )

    fig.update_layout(
        barmode="group",
        title=f"Category mean factuality (model: {model})",
        yaxis=dict(title="Mean factuality score", rangemode="tozero"),
        height=600,
        width=1200,
    )

    return fig


# ===============================
# MAIN
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--basename", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    models = detect_models(list(df.columns))
    if not models:
        raise SystemExit("No valid models detected.")

    model = args.model or models[0]

    dot_fig = make_dot_plot(df, model)
    bar_fig = make_category_bar_plot(df, model)

    os.makedirs(args.outdir, exist_ok=True)
    base = args.basename or f"entity_factuality_{model}"
    html_path = os.path.join(args.outdir, f"{base}.html")

    # Combine both figures into one HTML
    with open(html_path, "w") as f:
        f.write(pio.to_html(dot_fig, include_plotlyjs="cdn", full_html=True))
        f.write("<hr>")
        f.write(pio.to_html(bar_fig, include_plotlyjs=False, full_html=False))

    print("Wrote:", html_path)


if __name__ == "__main__":
    main()