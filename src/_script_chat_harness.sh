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

# Use the local/vLLM chat-template tool harness.
export AGENT_HARNESS="${AGENT_HARNESS:-chat}"

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
)

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$RESULTS_DIR}"
MAIN_SCRIPT="${MAIN_SCRIPT:-$SCRIPT_DIR/main.py}"
DATA="${DATA:-$DATA_DIR/Entity.csv}"

SEARCH_PROVIDER="${SEARCH_PROVIDER:-google}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
FORCE_SEARCH="${FORCE_SEARCH:-true}"
LIMIT="${LIMIT:-2}"
SKIP_SCORER="${SKIP_SCORER:-false}"

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
    echo "Running chat-harness experiment with tool description: $TOOL_DES"
    echo "Results will be saved to: ${BASE_OUTPUT_DIR}/${TOOL_DES}/"
    echo "Output files will include suffix: _chat_harness"
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
    )

    if [[ "$FORCE_SEARCH" == "true" ]]; then
        args+=(--force-search)
    fi
    if [[ "$SKIP_SCORER" == "true" ]]; then
        args+=(--skip-scorer)
    fi
    if [[ -n "$LIMIT" ]]; then
        args+=(--limit "$LIMIT")
    fi

    "$PYTHON_BIN" "$MAIN_SCRIPT" "${args[@]}"

    echo "Finished: $TOOL_DES"
    echo ""
done

echo "All chat-harness tool-description experiments completed."
