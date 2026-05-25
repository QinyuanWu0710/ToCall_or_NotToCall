#!/usr/bin/env bash
#SBATCH --partition=h100,a100,h200,a40
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00-10:00
#SBATCH --mem=100GB
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
setup_runtime


TOOL_DESCRIPTIONS=(
    "main"
    # "main-perceived-need"
    # "main-perceived-need-v1"
    # "main-perceived-need-v2"
    # "tool-cost-0"
    # "tool-cost-10"
    # "tool-cost-100"
    # "tool-cost-1000"
    # "tool-cost-10000"
    # "tool-cost-10000-aware"
    # "tool-cost-cheap"
    # "tool-cost-expensive"
    # ── budget-aware variants (append live total/finish/call counts) ──
    # "tool-cost-0-budget-aware-v2"
    # "tool-cost-10-budget-aware-v2"
    # "tool-cost-20-budget-aware-v2"
    # "tool-cost-25-budget-aware-v2"
    # "tool-cost-29-budget-aware-v2"
    # "tool-cost-33-budget-aware-v2"
    # "tool-cost-40-budget-aware-v2"
    # "tool-cost-50-budget-aware-v2"
    # "tool-cost-67-budget-aware-v2"
    # "tool-cost-100-budget-aware-v2"
    # "tool-cost-200-budget-aware-v2"
    # "tool-cost-222-budget-aware-v2"
    # "tool-cost-250-budget-aware-v2"
    # "tool-cost-500-budget-aware-v2"
    # "tool-cost-1000-budget-aware-v2"
    # "tool-cost-10000-budget-aware-v2"
    # "tool-cost-0-budget-aware"
    # "tool-cost-10-budget-aware"
    # "tool-cost-20-budget-aware"
    # "tool-cost-25-budget-aware"
    # "tool-cost-29-budget-aware"
    # "tool-cost-33-budget-aware"
    # "tool-cost-40-budget-aware"
    # "tool-cost-50-budget-aware"
    # "tool-cost-67-budget-aware"
    # "tool-cost-100-budget-aware"
    # "tool-cost-200-budget-aware"
    # "tool-cost-222-budget-aware"
    # "tool-cost-250-budget-aware"
    # "tool-cost-500-budget-aware"
    # "tool-cost-1000-budget-aware"
    # "tool-cost-10000-budget-aware"
    # "tool-cost-cheap-budget-aware"
    # "tool-cost-expensive-budget-aware"
)

MODEL="${MODEL:-Qwen/Qwen3-4B}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$RESULTS_DIR}"
MAIN_SCRIPT="${MAIN_SCRIPT:-$SCRIPT_DIR/main.py}"
DATA="${DATA:-$DATA_DIR/Entity.csv}"

TASK="${TASK:-entity}"
BFCL_GT_FILE="${BFCL_GT_FILE:-}"
SKIP_SCORER="${SKIP_SCORER:-false}"
SEARCH_PROVIDER="${SEARCH_PROVIDER:-google}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
LIMIT="${LIMIT:-2}"

if [[ "$SEARCH_PROVIDER" == "google" ]]; then
    require_env SERPAPI_API_KEY
elif [[ "$SEARCH_PROVIDER" == "brave" ]]; then
    require_env BRAVE_API_KEY
fi

if [[ "$SKIP_SCORER" != "true" ]]; then
    require_env OPENAI_API_KEY
fi

for TOOL_DES in "${TOOL_DESCRIPTIONS[@]}"; do
    echo "========================================================"
    echo "Running experiment with tool description: $TOOL_DES"
    echo "Results will be saved to: ${BASE_OUTPUT_DIR}/${TOOL_DES}/"
    echo "========================================================"

    args=(
        --data "$DATA"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --model-name "$MODEL"
        --temperature "$TEMPERATURE"
        --tool-description "$TOOL_DES"
        --output-dir "$BASE_OUTPUT_DIR"
        --search-provider "$SEARCH_PROVIDER"
        --web-search
        --task "$TASK"
    )

    if [[ "$SKIP_SCORER" == "true" ]]; then
        args+=(--skip-scorer)
    fi
    if [[ "$TASK" == "bfcl" && -n "$BFCL_GT_FILE" ]]; then
        args+=(--bfcl-gt-file "$BFCL_GT_FILE")
    fi
    if [[ -n "$LIMIT" ]]; then
        args+=(--limit "$LIMIT")
    fi

    "$PYTHON_BIN" "$MAIN_SCRIPT" "${args[@]}"

    echo "Finished: $TOOL_DES"
    echo ""
done

echo "All tool-description experiments completed."
