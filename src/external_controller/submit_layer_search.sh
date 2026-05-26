#!/bin/bash
# -----------------------------------------------------------------------
# submit_layer_search.sh
#
# Submit one SLURM job per (model × task × predictor) combination for
# layer_search.py.
#
# Usage:
#   # Submit all combinations:
#   bash submit_layer_search.sh
#
#   # Restrict to specific tasks, predictors, or models:
#   TASKS="bfcl invivo"          bash submit_layer_search.sh
#   PREDICTOR_IDS="1"            bash submit_layer_search.sh
#   MODEL_FILTER="Qwen3-30B-A3B" bash submit_layer_search.sh
#
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/layer_search_worker.sh"

# -----------------------------------------------------------------------
# Model registry  (MODEL_NAME → MODEL_PATH)
# -----------------------------------------------------------------------
declare -A MODELS

MODELS["Qwen_Qwen3-30B-A3B-Instruct-2507"]="./Qwen/Qwen3-30B-A3B-Instruct-2507"
MODELS["google_gemma-3-27b-it"]="./google/gemma-3-27b-it"
MODELS["openai_gpt-oss-120b"]="./openai/gpt-oss-120b"
MODELS["meta-llama_Llama-3.2-3B-Instruct"]="./meta-llama/Llama-3.2-3B-Instruct"
MODELS["mistralai_Mistral-Small-3.1-24B-Instruct-2503"]="./mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODELS["Qwen_Qwen3-30B-A3B"]="./Qwen/Qwen3-30B-A3B"

# -----------------------------------------------------------------------
# Dimensions to sweep (override via env vars before calling this script)
# -----------------------------------------------------------------------
TASKS="${TASKS:-entity bfcl invivo}"
# TASKS="${TASKS:-bfcl}"
PREDICTOR_IDS="${PREDICTOR_IDS:-1 2 3}"

# -----------------------------------------------------------------------
# Submit
# -----------------------------------------------------------------------
SUBMITTED=0
SKIPPED=0

for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"

    # Optional name-based filter
    if [[ -n "${MODEL_FILTER:-}" && "$MODEL_NAME" != *"$MODEL_FILTER"* ]]; then
        continue
    fi

    for TASK in $TASKS; do
        for PREDICTOR_ID in $PREDICTOR_IDS; do

            JOB_NAME="ls_${MODEL_NAME}_${TASK}_p${PREDICTOR_ID}"
            # SLURM job names have a ~255-char limit but let's keep it short
            JOB_NAME="${JOB_NAME:0:100}"

            echo "Submitting: MODEL=${MODEL_NAME}  TASK=${TASK}  PRED=${PREDICTOR_ID}"
            echo "  job-name: ${JOB_NAME}"

            sbatch \
                --partition=h200,h100,a100 \
                --gres=gpu:2 \
                -c 32 \
                -N 1 \
                --ntasks-per-node=1 \
                -t 01-00:00 \
                --mem=480GB \
                -o ./logging/%j.out \
                -e ./logging/%j.err \
                --job-name="${JOB_NAME}" \
                --export=ALL,MODEL_NAME="${MODEL_NAME}",MODEL_PATH="${MODEL_PATH}",TASK="${TASK}",PREDICTOR_ID="${PREDICTOR_ID}" \
                "${WORKER}"

            if [[ $? -eq 0 ]]; then
                (( SUBMITTED++ ))
            else
                echo "  WARNING: sbatch failed for ${JOB_NAME}"
                (( SKIPPED++ ))
            fi

            echo ""
        done
    done
done

echo "========================================================"
echo "  Done.  Submitted=${SUBMITTED}  Failed=${SKIPPED}"
echo "========================================================"
