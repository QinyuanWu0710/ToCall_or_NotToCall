"""
Analyze entity categories and create Plotly stacked bar charts:
1) Entities per Category (stacked by verified 0/1)
2) When verified=0: distribution of corrected categories by original category (100% stacked)
3) When verified=0: corrected category counts by original category (stacked counts)

Input:
  - CSV file with at least columns: entity_category, verified
  - For corrected distribution plots, also needs: corrected_category (configurable)

Outputs (in the chosen output directory):
  - stacked_bar_entities_by_category.html
  - stacked_bar_entities_by_category.png
  - unverified_corrected_distribution_100pct.html
  - unverified_corrected_distribution_100pct.png
  - unverified_corrected_counts.html
  - unverified_corrected_counts.png

Usage:
  python analyze_entities_plotly.py --csv /path/to/user_unique_entities_annotate.csv --outdir ./out

Notes:
  - PNG export uses Plotly's Kaleido engine.
    Compatibility note:
      * Plotly < 6  -> install kaleido==0.2.1
      * Plotly >= 6 -> install kaleido>=1
    If export fails, install one of:
      pip install -U kaleido==0.2.1
      pip install -U kaleido
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

RED = "#d62728"   # verified == 0
GREEN = "#2ca02c" # verified == 1

# Change this if your CSV uses a different column name for corrections
CORRECTED_COL_DEFAULT = "corrected"


def _normalize_str_series(s: pd.Series, unknown_label: str = "Unknown") -> pd.Series:
    return (
        s.astype(str)
         .fillna(unknown_label)
         .replace({"nan": unknown_label})
         .str.strip()
         .replace({"": unknown_label})
    )


def build_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "entity_category" not in df.columns:
        raise ValueError("Missing required column: 'entity_category'")
    if "verified" not in df.columns:
        raise ValueError("Missing required column: 'verified'")

    d = df.copy()
    d["entity_category"] = _normalize_str_series(d["entity_category"])
    d["verified"] = pd.to_numeric(d["verified"], errors="coerce").fillna(-1).astype(int)

    # Keep only 0/1
    d = d[d["verified"].isin([0, 1])].copy()

    counts = (
        d.groupby(["entity_category", "verified"])
         .size()
         .reset_index(name="count")
    )

    wide = (
        counts.pivot_table(index="entity_category", columns="verified", values="count", fill_value=0)
              .reset_index()
    )

    # Ensure both columns exist
    for col in [0, 1]:
        if col not in wide.columns:
            wide[col] = 0

    wide["total"] = wide[0] + wide[1]
    wide["verified_accuracy"] = wide.apply(
        lambda r: (r[1] / r["total"]) if r["total"] > 0 else 0.0,
        axis=1
    )

    wide = wide.sort_values("total", ascending=False).reset_index(drop=True)
    return wide


def compute_overall_accuracy(df: pd.DataFrame) -> float:
    """Overall accuracy = count(verified==1) / count(verified in {0,1})."""
    if "verified" not in df.columns:
        raise ValueError("Missing required column: 'verified'")

    v = pd.to_numeric(df["verified"], errors="coerce").astype("Int64")
    mask = v.isin([0, 1])
    total = int(mask.sum())
    if total == 0:
        return float("nan")
    num_verified = int((v[mask] == 1).sum())
    return num_verified / total


def make_verified_stacked_figure(wide: pd.DataFrame) -> go.Figure:
    categories = wide["entity_category"].tolist()
    count_0 = wide[0].tolist()
    count_1 = wide[1].tolist()
    totals = wide["total"].tolist()

    fig = go.Figure()

    fig.add_bar(
        x=categories,
        y=count_0,
        name="verified = 0",
        marker_color=RED,
        text=[str(v) if v > 0 else "" for v in count_0],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="Category=%{x}<br>verified=0: %{y}<extra></extra>",
    )
    fig.add_bar(
        x=categories,
        y=count_1,
        name="verified = 1",
        marker_color=GREEN,
        text=[str(v) if v > 0 else "" for v in count_1],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="Category=%{x}<br>verified=1: %{y}<extra></extra>",
    )

    fig.add_scatter(
        x=categories,
        y=totals,
        mode="text",
        text=[str(t) for t in totals],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_layout(
        title="Entities per Category (stacked by verified)",
        xaxis_title="Entity Category",
        yaxis_title="Number of Entities",
        barmode="stack",
        template="plotly_white",
        legend_title_text="Verification",
        margin=dict(l=60, r=20, t=70, b=140),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(rangemode="tozero")
    return fig


def build_unverified_corrections(
    df: pd.DataFrame,
    corrected_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      mat: crosstab (rows=original entity_category, cols=corrected category, values=count) for verified==0
      totals: per-row totals (unverified counts per original category)
    """
    needed = {"entity_category", "verified", corrected_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) for correction plots: {sorted(missing)}")

    d = df.copy()
    d["entity_category"] = _normalize_str_series(d["entity_category"])
    d[corrected_col] = _normalize_str_series(d[corrected_col])
    d["verified"] = pd.to_numeric(d["verified"], errors="coerce").fillna(-1).astype(int)

    d = d[d["verified"] == 0].copy()

    mat = pd.crosstab(d["entity_category"], d[corrected_col])

    # Sort rows by total descending
    totals = mat.sum(axis=1).sort_values(ascending=False)
    mat = mat.loc[totals.index]

    return mat, totals


def _collapse_top_k_columns(mat: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep top_k columns by total; collapse the rest into 'Other'."""
    if mat.empty:
        return mat

    col_totals = mat.sum(axis=0).sort_values(ascending=False)
    top_cols = col_totals.head(top_k).index.tolist()
    other_cols = [c for c in mat.columns if c not in top_cols]

    out = mat[top_cols].copy()
    if other_cols:
        out["UNKNOWN"] = mat[other_cols].sum(axis=1)
    return out


def make_unverified_distribution_figure(
    mat: pd.DataFrame,
    totals: pd.Series,
    top_k_corrected: int = 12,
    top_n_original: int | None = None,
) -> go.Figure:
    """
    100% stacked:
      x = original category (verified==0)
      stacks = corrected category (top K global, rest -> Other)
    Optionally show only top_n_original original categories by unverified volume.
    """
    if mat.empty:
        fig = go.Figure()
        fig.update_layout(
            title="When verified=0: distribution of corrected categories (no data after filtering)",
            template="plotly_white",
        )
        return fig

    # Optionally limit original categories to top N by unverified count
    if top_n_original is not None and top_n_original > 0:
        keep_rows = totals.head(top_n_original).index
        mat = mat.loc[keep_rows]
        totals = totals.loc[keep_rows]

    mat2 = _collapse_top_k_columns(mat, top_k_corrected)

    row_sums = mat2.sum(axis=1).replace(0, 1)
    prop = mat2.div(row_sums, axis=0)

    categories = prop.index.tolist()

    fig = go.Figure()
    for col in prop.columns:
        fig.add_bar(
            x=categories,
            y=prop[col].tolist(),
            name=str(col),
            hovertemplate=(
                "Original=%{x}<br>"
                f"Corrected={col}<br>"
                "Share=%{y:.1%}<extra></extra>"
            ),
        )

    # Total (absolute unverified count) label above each bar
    fig.add_scatter(
        x=categories,
        y=[1.0] * len(categories),
        mode="text",
        text=[str(int(t)) for t in totals.loc[categories].tolist()],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_layout(
        title="When verified=0: distribution of corrected categories by original category (100% stacked)",
        xaxis_title="Original Entity Category (verified=0 only)",
        yaxis_title="Share of corrected categories",
        barmode="stack",
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=160),
        legend_title_text="Corrected Category",
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def make_unverified_counts_figure(
    mat: pd.DataFrame,
    top_k_corrected: int = 12,
    top_n_original: int | None = None,
) -> go.Figure:
    """
    Stacked counts:
      x = original category (verified==0)
      stacks = corrected category (top K global, rest -> Other)
    Optionally show only top_n_original original categories by unverified volume.
    """
    if mat.empty:
        fig = go.Figure()
        fig.update_layout(
            title="When verified=0: corrected category counts (no data after filtering)",
            template="plotly_white",
        )
        return fig

    totals = mat.sum(axis=1).sort_values(ascending=False)
    if top_n_original is not None and top_n_original > 0:
        keep_rows = totals.head(top_n_original).index
        mat = mat.loc[keep_rows]

    mat2 = _collapse_top_k_columns(mat, top_k_corrected)

    categories = mat2.index.tolist()

    fig = go.Figure()
    for col in mat2.columns:
        fig.add_bar(
            x=categories,
            y=mat2[col].tolist(),
            name=str(col),
            hovertemplate=(
                "Original=%{x}<br>"
                f"Corrected={col}<br>"
                "Count=%{y}<extra></extra>"
            ),
        )

    fig.update_layout(
        title="When verified=0: corrected category counts by original category (stacked)",
        xaxis_title="Original Entity Category (verified=0 only)",
        yaxis_title="Count",
        barmode="stack",
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=160),
        legend_title_text="Corrected Category",
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(rangemode="tozero")
    return fig


def _write_plotly_outputs(fig: go.Figure, html_path: Path, png_path: Path) -> None:
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception as e:
        raise RuntimeError(
            "PNG export failed. This is usually a Plotly/Kaleido version mismatch. "
            "Try: (a) pip install -U kaleido==0.2.1 (for Plotly<6) or "
            "(b) pip install -U plotly kaleido (for Plotly>=6). "
            f"Original error: {e}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/exported_entities_human_verify/user_unique_entities_annotate.csv",
        help="Path to the CSV file",
    )
    p.add_argument(
        "--outdir",
        default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/exported_entities_human_verify",
        help="Output directory",
    )
    p.add_argument(
        "--corrected-col",
        default=CORRECTED_COL_DEFAULT,
        help=f"Column name for corrected category (default: {CORRECTED_COL_DEFAULT})",
    )
    p.add_argument(
        "--top-k-corrected",
        type=int,
        default=12,
        help="For correction plots: keep top K corrected categories globally; rest collapsed into 'Other'",
    )
    p.add_argument(
        "--top-n-original",
        type=int,
        default=0,
        help="For correction plots: show only top N original categories by unverified volume (0 = show all)",
    )
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
        sep=None,
        engine="python",
        on_bad_lines="warn",
    )

    # 1) Overall verified accuracy
    overall_acc = compute_overall_accuracy(df)
    if pd.isna(overall_acc):
        print("Overall verified accuracy: n/a (no rows with verified in {0,1})")
    else:
        print(f"Overall verified accuracy (verified==1 / total 0/1): {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    # 2) Verified stacked by category
    wide = build_counts(df)
    fig_verified = make_verified_stacked_figure(wide)

    html_verified = outdir / "stacked_bar_entities_by_category.html"
    png_verified = outdir / "stacked_bar_entities_by_category.png"
    _write_plotly_outputs(fig_verified, html_verified, png_verified)

    print(f"Saved HTML: {html_verified}")
    print(f"Saved PNG : {png_verified}")

    # 3) Correction plots for verified == 0
    top_n_original = args.top_n_original if args.top_n_original and args.top_n_original > 0 else None

    mat, totals = build_unverified_corrections(df, corrected_col=args.corrected_col)

    fig_dist = make_unverified_distribution_figure(
        mat,
        totals,
        top_k_corrected=args.top_k_corrected,
        top_n_original=top_n_original,
    )
    fig_counts = make_unverified_counts_figure(
        mat,
        top_k_corrected=args.top_k_corrected,
        top_n_original=top_n_original,
    )

    html_dist = outdir / "unverified_corrected_distribution_100pct.html"
    png_dist = outdir / "unverified_corrected_distribution_100pct.png"
    html_cnt = outdir / "unverified_corrected_counts.html"
    png_cnt = outdir / "unverified_corrected_counts.png"

    _write_plotly_outputs(fig_dist, html_dist, png_dist)
    _write_plotly_outputs(fig_counts, html_cnt, png_cnt)

    print(f"Saved 100% dist HTML: {html_dist}")
    print(f"Saved 100% dist PNG : {png_dist}")
    print(f"Saved counts HTML   : {html_cnt}")
    print(f"Saved counts PNG    : {png_cnt}")


if __name__ == "__main__":
    main()
