"""
Reorganise the result for BFCL.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SAVE_DIR = '/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/refined'
MAIN_DIR = '/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/response_summary_csv'
OUTPUT_DIR = '/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/main'

MODEL_DICT = {
    'gpt-oss-120b':              'openai_gpt-oss-120b',
    'gemma-3-27b-it':            'google_gemma-3-27b-it',
    'llama-3.2-3b-it':           'meta-llama_Llama-3.2-3B-Instruct',
    'mistral-small-3.1-24b-it':  'mistralai_Mistral-Small-3.1-24B-Instruct-2503',
    'qwen3-30b-A3B-it':          'Qwen_Qwen3-30B-A3B-Instruct-2507',
    'qwen3-30b-A3B':             'Qwen_Qwen3-30B-A3B',
}

COLUMN_RENAME_MAP = {
    'tool_call(y/n)': 'search_called',
    'llm_score':      'score',
}

TOOL_CALL_VALUE_MAP = {
    'YES': 'True',
    'NO':  'False',
}


def process_model(model_name: str, mapped_name: str) -> None:
    input_path = Path(MAIN_DIR) / f'bfcl_web_search_{model_name}-no_search_llm_judge.csv'
    output_path = Path(OUTPUT_DIR) / f'vllm_{mapped_name}_no_search_summary.csv'

    if not input_path.exists():
        print(f'[SKIP] File not found: {input_path}')
        return

    df = pd.read_csv(input_path)

    # Rename columns (only those that exist in the dataframe)
    existing_renames = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    # Map YES/NO → True/False in 'search_called'
    if 'search_called' in df.columns:
        df['search_called'] = df['search_called'].map(TOOL_CALL_VALUE_MAP).fillna(df['search_called'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'[OK] {model_name} → {output_path}')


def main() -> None:
    for model_name, mapped_name in MODEL_DICT.items():
        process_model(model_name, mapped_name)


if __name__ == '__main__':
    main()