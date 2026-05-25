#!/usr/bin/env bash

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
)

MODEL="${MODEL:-gpt-5.5}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$RESULTS_DIR}"
MAIN_SCRIPT="${MAIN_SCRIPT:-$SCRIPT_DIR/main.py}"
DATA="${DATA:-$DATA_DIR/Entity.csv}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
WEB_SEARCH="${WEB_SEARCH:-false}"
FORCE_SEARCH="${FORCE_SEARCH:-false}"
SEARCH_PROVIDER="${SEARCH_PROVIDER:-google}"
LIMIT="${LIMIT:-}"
SKIP_SCORER="${SKIP_SCORER:-false}"

require_env OPENAI_API_KEY
if [[ "$WEB_SEARCH" == "true" && "$SEARCH_PROVIDER" == "google" ]]; then
    require_env SERPAPI_API_KEY
elif [[ "$WEB_SEARCH" == "true" && "$SEARCH_PROVIDER" == "brave" ]]; then
    require_env BRAVE_API_KEY
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
    )

    if [[ "$WEB_SEARCH" == "true" ]]; then
        args+=(--search-provider "$SEARCH_PROVIDER" --web-search)
    fi
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

echo "All tool-description experiments completed."
