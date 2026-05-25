for task in \
  entity \
  invivo \
  bfcl 
do
  for model_name in \
    Qwen3-30B-A3B-Instruct-2507 \
    gemma-3-27b-it \
    Llama-3.2-3B-Instruct \
    Mistral-Small-3.1-24B-Instruct-2503 \
    Qwen3-30B-A3B \
    gpt-oss-120b 
  do
    for predictor_id in 1 2 3; do
      for clf in LogisticRegression XGBoost MLP; do
        python /NS/chatgpt/work/qwu/hallucinations_detection/code/eval/entity_knowledge/analyse/predictor.py \
          --model_name ${model_name} \
          --predictor_id ${predictor_id} \
          --clf ${clf} \
          --task ${task}
      done
    done
  done
done