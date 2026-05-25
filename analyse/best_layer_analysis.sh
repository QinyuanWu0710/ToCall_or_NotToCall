#!/usr/bin/env bash
# best_layer_analysis.sh
#
# Runs best_layer_analysis.py which produces, for every task × model × predictor:
#   - comparison_best_layer.csv          (factuality scores under each search strategy)
#   - classifier_auroc_summary.csv       (OOF AUROC for every layer × classifier)
#   - best_layer_per_model.csv           (best layer + clf per model × predictor)
#   - figures/score_comparison.pdf       (grouped bar chart across strategies)
#   - figures/predictor{N}_auroc_by_layer.pdf   (AUROC vs. layer, one line per model)
#   - figures/predictor{N}_{model}_cm.pdf       (confusion matrices)
#
# All outputs land under:
#   /NS/chatgpt/work/qwu/hallucinations_detection/results/predictor_best_layer/{task}/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/best_layer_analysis.py"
