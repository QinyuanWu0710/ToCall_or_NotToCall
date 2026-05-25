# Full pipeline (extraction + verification)
# python /NS/chatgpt/work/qwu/hallucinations_detection/code/all_users/entity_extraction.py --api_key "." \
#  --model_name gpt-5.1 \
#  --input_csv /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/annotation_sample_100_conversations.csv \
#  --output_dir /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_entities_gpt-5.1_test
# Extraction only
# python script.py --api_key YOUR_API_KEY --input_csv data.csv --output_dir output/ --skip_verification

# Use OpenAI's batch API
python /NS/chatgpt/work/qwu/hallucinations_detection/code/data_process/entity_extraction_batch.py --api_key "." \
 --model_name gpt-5.1 \
 --input_csv /NS/chatgpt/work/qwu/hallucinations_detection/data/wildchat/extracted_conversations.csv \
 --output_dir /NS/chatgpt/work/qwu/hallucinations_detection/data/wildchat/extracted_entities_gpt-5.1_batch \
 --skip_verification