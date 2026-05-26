#!/bin/bash
# -----------------------------------------------------------------------
# layer_search_worker.sh
#
# Run layer_search.py for one (MODEL_NAME, MODEL_PATH, TASK, PREDICTOR_ID)
# combination.  Intended to be submitted via submit_layer_search.sh using:
#
#   sbatch --export=ALL,MODEL_NAME=...,MODEL_PATH=...,TASK=...,PREDICTOR_ID=... \
#          layer_search_worker.sh
#
# TASK must be one of: entity | bfcl | invivo
# PREDICTOR_ID must be one of: 1 | 2 | 3
# -----------------------------------------------------------------------

export TRITON_CACHE_DIR='./logging/.cache/triton'

# Set for A100, H100 and H200
export CUDA_HOME=/usr/lib/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export VLLM_CACHE_ROOT=./logging

# -----------------------------------------------------------------------
# Validate required env vars
# -----------------------------------------------------------------------
: "${MODEL_NAME:?ERROR: MODEL_NAME is not set}"
: "${MODEL_PATH:?ERROR: MODEL_PATH is not set}"
: "${TASK:?ERROR: TASK is not set (entity | bfcl | invivo)}"
: "${PREDICTOR_ID:?ERROR: PREDICTOR_ID is not set (1 | 2 | 3)}"

echo "========================================================"
echo "  layer_search_worker"
echo "  MODEL_NAME   = $MODEL_NAME"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  TASK         = $TASK"
echo "  PREDICTOR_ID = $PREDICTOR_ID"
echo "========================================================"

BASE_RESULTS="./results"
BASE_DATA="./data/tool_predictor"

# -----------------------------------------------------------------------
# Search variant for results_path:
#   predictor 1  → no_search
#   predictor 2/3 → with_search
# -----------------------------------------------------------------------
case "$PREDICTOR_ID" in
    1) SEARCH_VARIANT="no_search" ;;
    2|3) SEARCH_VARIANT="with_search" ;;
    *) echo "ERROR: unsupported PREDICTOR_ID=$PREDICTOR_ID"; exit 1 ;;
esac

# -----------------------------------------------------------------------
# Paths per task
# -----------------------------------------------------------------------
case "$TASK" in
    entity)
        RESULTS_DIR="${BASE_RESULTS}/entity_hallucination/temp=0/main"
        DATA_ROOT="${BASE_DATA}"
        ;;
    bfcl)
        RESULTS_DIR="${BASE_RESULTS}/bfcl_raw/tool_result/main"
        DATA_ROOT="${BASE_DATA}/bfcl"
        ;;
    invivo)
        RESULTS_DIR="${BASE_RESULTS}/real_query/temp=0/main"
        DATA_ROOT="${BASE_DATA}/invivo"
        ;;
    *)
        echo "ERROR: unsupported TASK=$TASK (must be entity | bfcl | invivo)"
        exit 1
        ;;
esac

RESULTS_PATH="${RESULTS_DIR}/vllm_${MODEL_NAME}_${SEARCH_VARIANT}.jsonl"
LABEL_PATH="${RESULTS_DIR}/vllm_${MODEL_NAME}_no_search_summary.csv"

echo "  RESULTS_PATH = $RESULTS_PATH"
echo "  LABEL_PATH   = $LABEL_PATH"
echo "  DATA_ROOT    = $DATA_ROOT"
echo "========================================================"

python layer_search.py \
  --results_path "$RESULTS_PATH" \
  --model_name_or_path "$MODEL_PATH" \
  --label_path "$LABEL_PATH" \
  --data_root "$DATA_ROOT" \
  --setting auto \
  --predictor_id "$PREDICTOR_ID" \
  --device cuda \
  --batch_size 8 \
  --force_rebuild
