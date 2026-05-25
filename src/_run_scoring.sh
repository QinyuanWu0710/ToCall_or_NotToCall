#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/_common.sh"
setup_runtime
require_env OPENAI_API_KEY

SCORING_SCRIPT="${SCORING_SCRIPT:-$SCRIPT_DIR/run_scoring.py}"
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-}"
EXTRA_SCORING_ARGS="${EXTRA_SCORING_ARGS:-}"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
    echo "Usage:" >&2
    echo "  INPUT=/path/to/results.jsonl OUTPUT=/path/to/summary.csv $0" >&2
    echo "" >&2
    echo "Optional overrides: PYTHON_BIN, SCORING_SCRIPT, EXTRA_SCORING_ARGS" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCORING_SCRIPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    $EXTRA_SCORING_ARGS
