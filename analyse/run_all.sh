#!/usr/bin/env bash
# run_all.sh — Check Mistral parsing, then generate all figures for all tasks.
#
# Usage:
#   bash run_all.sh                   # all tasks, all models
#   bash run_all.sh --task invivo     # single task
#   bash run_all.sh --models gpt-oss-120b gemma-3-27b-it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${TASK:-all}"        # override via env: TASK=bfcl bash run_all.sh
EXTRA_ARGS=("$@")          # pass-through args to new_analysis.py / venn_diagram.py

# ── Step 1: Check / patch Mistral search_called parsing ───────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Step 1: check_mistral.py  (task=${TASK})               ║"
echo "╚══════════════════════════════════════════════════════════╝"
python check_mistral.py --task "${TASK}"

# ── Step 2: Full normative analysis (confusion matrices + affordability) ───────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Step 2: new_analysis.py   (task=${TASK})               ║"
echo "╚══════════════════════════════════════════════════════════╝"
python new_analysis.py --task "${TASK}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# ── Step 3: Venn diagrams ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Step 3: venn_diagram.py   (task=${TASK})               ║"
echo "╚══════════════════════════════════════════════════════════╝"
python venn_diagram.py --task "${TASK}"

echo ""
echo "All done."
