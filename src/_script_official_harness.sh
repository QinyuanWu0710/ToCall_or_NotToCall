#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
setup_runtime

# Use the OpenAI Responses API official tool harness.
export OPENAI_AGENT_BACKEND="${OPENAI_AGENT_BACKEND:-official}"
export OPENAI_AGENT_TOOL_BACKEND="${OPENAI_AGENT_TOOL_BACKEND:-responses_url}"
export MCP_SEARCH_PROVIDER="${MCP_SEARCH_PROVIDER:-mcp-serp}"

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

SEARCH_PROVIDER="${SEARCH_PROVIDER:-google}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LIMIT="${LIMIT:-}"

require_env OPENAI_API_KEY
if [[ "$SEARCH_PROVIDER" == "google" ]]; then
    require_env SERPAPI_API_KEY
elif [[ "$SEARCH_PROVIDER" == "brave" ]]; then
    require_env BRAVE_API_KEY
fi

for TOOL_DES in "${TOOL_DESCRIPTIONS[@]}"; do
    echo "========================================================"
    echo "Running official-harness experiment with tool description: $TOOL_DES"
    echo "Results will be saved to: ${BASE_OUTPUT_DIR}/${TOOL_DES}/"
    echo "Output files will include suffix: _official_harness"
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

    if [[ -n "$LIMIT" ]]; then
        args+=(--limit "$LIMIT")
    fi

    "$PYTHON_BIN" "$MAIN_SCRIPT" "${args[@]}"

    echo "Finished: $TOOL_DES"
    echo ""
done

echo "All official-harness tool-description experiments completed."
