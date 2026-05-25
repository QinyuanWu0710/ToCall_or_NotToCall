#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logging}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOG_DIR/.cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$LOG_DIR}"

configure_cuda() {
    if [[ -z "${CUDA_HOME:-}" ]]; then
        if [[ -d /usr/local/cuda ]]; then
            export CUDA_HOME=/usr/local/cuda
        elif [[ -d /usr/lib/cuda-12.8 ]]; then
            export CUDA_HOME=/usr/lib/cuda-12.8
        elif [[ -d /usr/lib/cuda ]]; then
            export CUDA_HOME=/usr/lib/cuda
        fi
    fi

    if [[ -n "${CUDA_HOME:-}" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: set $name in your environment before running this script." >&2
        exit 1
    fi
}

setup_runtime() {
    mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$TRITON_CACHE_DIR"
    cd "$SCRIPT_DIR"
    configure_cuda

    PYTHON_BIN="${PYTHON_BIN:-python3}"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Error: Python executable not found: $PYTHON_BIN" >&2
        echo "Set PYTHON_BIN=/path/to/python for your cluster or virtualenv." >&2
        exit 1
    fi
}
