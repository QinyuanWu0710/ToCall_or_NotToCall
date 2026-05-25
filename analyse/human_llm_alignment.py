"""
Analyse human annotation agreement and alignment with LLM-as-judge results.

Inputs:
  - annotate_sample100_SD.csv   (semicolon-delimited, utf-8)
  - annotate_sample100_sean.csv (comma-delimited, latin-1)
  - annotate_sample100_QW.csv   (semicolon-delimited, utf-8)

For each row, claims 1–36 may be populated.
  claim_N_human      : 1.0 = correct, 0.0 = incorrect, NaN = claim absent
  claim_N_is_correct : True / False  (LLM-as-judge label)

Outputs printed to stdout:
  1. Basic annotation counts per annotator
  2. Pairwise inter-annotator agreement (all pairs)
  3. Human vs LLM-as-judge alignment per annotator
  4. Four-way summary (SD, Sean, QW, LLM) on shared claims
  5. Per-claim-number breakdown (with -v / --verbose flag)
"""

import sys
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, classification_report, confusion_matrix, accuracy_score
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_DIR = "/NS/chatgpt/work/qwu/hallucinations_detection/results/human"

ANNOTATORS = {
    "SD":   dict(path=f"{DATA_DIR}/annotate_sample100_SD.csv",   sep=";", encoding="utf-8"),
    "Sean": dict(path=f"{DATA_DIR}/annotate_sample100_sean.csv", sep=",", encoding="latin-1"),
    "QW":   dict(path=f"{DATA_DIR}/annotate_sample100_QW.csv",  sep=";", encoding="utf-8"),
}

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

CLAIM_NUMS = range(1, 37)
human_cols = [f"claim_{n}_human"      for n in CLAIM_NUMS]
llm_cols   = [f"claim_{n}_is_correct" for n in CLAIM_NUMS]

# ---------------------------------------------------------------------------
# Load data — align all on shared custom_id index
# ---------------------------------------------------------------------------
dfs = {}
for name, cfg in ANNOTATORS.items():
    df = pd.read_csv(cfg["path"], sep=cfg["sep"], encoding=cfg["encoding"])
    dfs[name] = df.set_index("custom_id")

common_ids = dfs["SD"].index
for df in dfs.values():
    common_ids = common_ids.intersection(df.index)

dfs = {name: df.loc[common_ids] for name, df in dfs.items()}
print(f"Rows shared by all annotators: {len(common_ids)}")

# Use SD's LLM columns as the ground-truth judge (all files share the same LLM output)
df_llm = dfs["SD"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_paired_human_llm(df, df_judge=None):
    """Flatten (human, llm) pairs where both are non-NaN."""
    if df_judge is None:
        df_judge = df
    h_vals, l_vals = [], []
    for h_col, l_col in zip(human_cols, llm_cols):
        mask = df[h_col].notna() & df_judge[l_col].notna()
        if mask.sum() == 0:
            continue
        h_vals.append(df.loc[mask, h_col].astype(int))
        l_vals.append(df_judge.loc[mask, l_col].astype(int))
    return np.concatenate(h_vals), np.concatenate(l_vals)


def get_paired_human_human(df_a, df_b):
    """Flatten (a, b) pairs where both annotators provided a label."""
    a_vals, b_vals = [], []
    for col in human_cols:
        mask = df_a[col].notna() & df_b[col].notna()
        if mask.sum() == 0:
            continue
        a_vals.append(df_a.loc[mask, col].astype(int))
        b_vals.append(df_b.loc[mask, col].astype(int))
    return np.concatenate(a_vals), np.concatenate(b_vals)


# ---------------------------------------------------------------------------
# 1. Basic annotation counts
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  1. Annotation counts per annotator")
print(f"{'='*60}")

for name, df in dfs.items():
    annotated_mask = df[human_cols].notna()
    n_annotated = int(annotated_mask.sum().sum())
    n_correct   = int((df[human_cols] == 1).sum().sum())
    n_incorrect = int((df[human_cols] == 0).sum().sum())
    claims_per_row = annotated_mask.sum(axis=1)
    zero_rows = (claims_per_row == 0).sum()

    print(f"\n  [{name}]")
    print(f"    Total claims annotated : {n_annotated}")
    print(f"    Correct   (human=1)    : {n_correct}  ({100*n_correct/n_annotated:.1f}%)")
    print(f"    Incorrect (human=0)    : {n_incorrect}  ({100*n_incorrect/n_annotated:.1f}%)")
    print(f"    Per-row: min={claims_per_row.min()}, max={claims_per_row.max()}, "
          f"mean={claims_per_row.mean():.1f}, median={claims_per_row.median():.0f}")
    if zero_rows:
        print(f"    WARNING: {zero_rows} row(s) with no annotations")


# ---------------------------------------------------------------------------
# 2. Pairwise inter-annotator agreement
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  2. Pairwise inter-annotator agreement")
print(f"{'='*60}")

annotator_names = list(dfs.keys())
for name_a, name_b in itertools.combinations(annotator_names, 2):
    df_a, df_b = dfs[name_a], dfs[name_b]
    a_arr, b_arr = get_paired_human_human(df_a, df_b)
    n_shared = len(a_arr)
    agree    = (a_arr == b_arr).sum()
    kappa    = cohen_kappa_score(a_arr, b_arr)
    cm       = confusion_matrix(a_arr, b_arr, labels=[0, 1])

    print(f"\n  {name_a} vs {name_b}")
    print(f"    Claims with both labels  : {n_shared}")
    print(f"    Agreement                : {agree} ({100*agree/n_shared:.1f}%)")
    print(f"    Cohen's kappa            : {kappa:.4f}")
    print(f"    Confusion matrix (rows={name_a}, cols={name_b}):")
    print(f"                  {name_b}=0  {name_b}=1")
    print(f"      {name_a}=0       {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"      {name_a}=1       {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Disagreement breakdown
    n_disagree = int((a_arr != b_arr).sum())
    a0_b1 = int(((a_arr == 0) & (b_arr == 1)).sum())  # a=incorrect, b=correct
    a1_b0 = int(((a_arr == 1) & (b_arr == 0)).sum())  # a=correct,   b=incorrect
    print(f"    Disagreements            : {n_disagree}")
    print(f"      {name_a}=0, {name_b}=1 (only {name_b} marks correct): {a0_b1}")
    print(f"      {name_a}=1, {name_b}=0 (only {name_a} marks correct): {a1_b0}")

    if VERBOSE:
        print(f"\n    Detailed disagreements ({name_a} vs {name_b}):")
        for n_idx, (h_col, l_col) in enumerate(zip(human_cols, llm_cols), 1):
            mask = (
                df_a[h_col].notna() & df_b[h_col].notna() &
                (df_a[h_col] != df_b[h_col])
            )
            for idx in df_a.index[mask]:
                llm_val = df_llm.loc[idx, l_col]
                llm_str = ("TRUE" if bool(llm_val) else "FALSE") if pd.notna(llm_val) else "N/A"
                print(f"      [{df_a.loc[idx, 'entity']}] claim_{n_idx}: "
                      f"{name_a}={int(df_a.loc[idx, h_col])} "
                      f"{name_b}={int(df_b.loc[idx, h_col])} LLM={llm_str}")


# ---------------------------------------------------------------------------
# 3. Human vs LLM-as-judge alignment per annotator
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  3. Human vs LLM-as-judge alignment")
print(f"{'='*60}")

for name, df in dfs.items():
    h_arr, l_arr = get_paired_human_llm(df, df_llm)
    n     = len(h_arr)
    acc   = accuracy_score(h_arr, l_arr)
    kappa = cohen_kappa_score(h_arr, l_arr)

    tp = int(((h_arr == 1) & (l_arr == 1)).sum())
    tn = int(((h_arr == 0) & (l_arr == 0)).sum())
    fp = int(((h_arr == 0) & (l_arr == 1)).sum())
    fn = int(((h_arr == 1) & (l_arr == 0)).sum())

    print(f"\n  [{name}]")
    print(f"    Claim pairs evaluated    : {n}")
    print(f"    Accuracy (human=LLM)     : {100*acc:.2f}%")
    print(f"    Cohen's kappa            : {kappa:.4f}")
    print(f"    Confusion matrix (rows=human, cols=LLM):")
    print(f"                  LLM=0  LLM=1")
    print(f"      human=0     {tn:5d}  {fp:5d}   (TN / FP)")
    print(f"      human=1     {fn:5d}  {tp:5d}   (FN / TP)")

    n_mismatch = fp + fn
    print(f"    Total mismatches         : {n_mismatch} / {n} ({100*n_mismatch/n:.1f}%)")
    print(f"      LLM over-credits  (LLM=1, human=0): {fp}")
    print(f"      LLM under-credits (LLM=0, human=1): {fn}")

    print(f"\n    Classification report (positive = 'correct'):")
    report = classification_report(h_arr, l_arr, target_names=["incorrect", "correct"])
    for line in report.splitlines():
        print(f"      {line}")

    if VERBOSE:
        print(f"\n    Per-claim-number mismatch rate:")
        for n_idx, (h_col, l_col) in enumerate(zip(human_cols, llm_cols), 1):
            mask = df[h_col].notna() & df_llm[l_col].notna()
            if mask.sum() == 0:
                continue
            h = df.loc[mask, h_col].astype(int)
            l = df_llm.loc[mask, l_col].astype(int)
            mismatch = int((h != l).sum())
            total = int(mask.sum())
            if mismatch > 0:
                print(f"      claim_{n_idx:2d}: {mismatch}/{total} mismatches "
                      f"({100*mismatch/total:.0f}%)")


# ---------------------------------------------------------------------------
# 4. Four-way summary (SD, Sean, QW, LLM) on claims all three humans annotated
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  4. Four-way summary (SD, Sean, QW, LLM) on shared claims")
print(f"{'='*60}")

arrays = {name: [] for name in annotator_names}
arrays["LLM"] = []

for h_col, l_col in zip(human_cols, llm_cols):
    # Require all three humans AND LLM to have a label
    mask = df_llm[l_col].notna()
    for name in annotator_names:
        mask = mask & dfs[name][h_col].notna()
    if mask.sum() == 0:
        continue
    for name in annotator_names:
        arrays[name].append(dfs[name].loc[mask, h_col].astype(int))
    arrays["LLM"].append(df_llm.loc[mask, l_col].astype(int))

arrs = {k: np.concatenate(v) for k, v in arrays.items()}
n_total = len(arrs["SD"])

print(f"\n  Claims where all three humans + LLM have labels: {n_total}")

# All agree
all_agree = np.ones(n_total, dtype=bool)
for k in arrs:
    all_agree &= (arrs[k] == arrs["SD"])
print(f"  All four agree                      : {all_agree.sum()} ({100*all_agree.mean():.1f}%)")

# All humans agree, LLM differs
humans_agree = (arrs["SD"] == arrs["Sean"]) & (arrs["Sean"] == arrs["QW"])
humans_agree_llm_diff = humans_agree & (arrs["SD"] != arrs["LLM"])
print(f"  All humans agree, LLM differs       : {humans_agree_llm_diff.sum()} "
      f"({100*humans_agree_llm_diff.mean():.1f}%)")

# LLM + majority humans agree (2 of 3 humans + LLM agree)
for name in annotator_names:
    others = [n for n in annotator_names if n != name]
    llm_and_others = (arrs["LLM"] == arrs[others[0]]) & (arrs["LLM"] == arrs[others[1]])
    outlier = llm_and_others & (arrs[name] != arrs["LLM"])
    print(f"  LLM+{others[0]}+{others[1]} agree, {name} differs : {outlier.sum()} "
          f"({100*outlier.mean():.1f}%)")

# All four disagree (binary, so impossible — just note
# that with binary labels at least 2 must agree)
print()

# Agreement rate table for each pair including LLM
all_keys = annotator_names + ["LLM"]
print(f"  Pairwise agreement rates (on shared {n_total} claims):")
header = f"{'':8s}" + "".join(f"{k:>8s}" for k in all_keys)
print(f"    {header}")
for k1 in all_keys:
    row = f"    {k1:<8s}"
    for k2 in all_keys:
        if k1 == k2:
            row += f"{'—':>8s}"
        else:
            pct = 100 * (arrs[k1] == arrs[k2]).mean()
            row += f"{pct:>7.1f}%"
    print(row)

print()
