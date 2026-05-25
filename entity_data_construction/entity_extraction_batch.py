"""
Entity extraction and verification pipeline using Wolfram Language Entity Types and GPT-4o.
Uses OpenAI Batch API for cost-effective processing.

Pipeline:
1. Reads messages from CSV
2. Checks for existing result files and filters out already processed messages
3. Reports how many new messages need annotation
4. Creates batch requests for entity extraction
5. Submits batch job and waits for completion
6. Optionally creates verification batch, we're planning to only do the verification for the 100 humann annotation data.
7. Saves one JSON per message (same format as original code)
"""
import os
import json
import argparse
import pandas as pd
import time
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

# Import prompts and schemas from entity_extraction.py
from entity_extraction import (
    EXTRACTION_JSON_SCHEMA,
    VERIFICATION_JSON_SCHEMA,
    PROMPT_ENTITY_EXTRACTION,
    PROMPT_ENTITY_VERIFICATION
)

# =========================
# ARGUMENTS
# =========================
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06")
argparser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
argparser.add_argument(
    "--input_csv",
    type=str,
    default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_conversations.csv",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_entities",
)
argparser.add_argument(
    "--batch_dir",
    type=str,
    default="/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/batch_files",
    help="Directory to store batch input/output files"
)
argparser.add_argument(
    "--skip_verification",
    action="store_true",
    help="Skip verification step (extraction only)"
)
argparser.add_argument(
    "--check_interval",
    type=int,
    default=60,
    help="Seconds to wait between batch status checks"
)
args = argparser.parse_args()

MODEL_NAME = args.model_name
INPUT_CSV = args.input_csv
OUTPUT_DIR = args.output_dir
BATCH_DIR = args.batch_dir
SKIP_VERIFICATION = args.skip_verification
CHECK_INTERVAL = args.check_interval

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
client = OpenAI(api_key=args.api_key)


def print_statistics(extraction_results: dict, verification_results: dict, skip_verification: bool):
    """Print statistics about entity extraction and verification"""
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    # Extraction statistics
    total_turns = len(extraction_results)
    turns_with_entities = 0
    turns_without_entities = 0
    total_entities = 0
    entity_type_counts = defaultdict(int)
    
    for custom_id, entity_data in extraction_results.items():
        entities = entity_data.get("entities", [])
        if entities and len(entities) > 0:
            turns_with_entities += 1
            total_entities += len(entities)
            for entity in entities:
                entity_type = entity.get("entity_type_category", "Unknown")
                entity_type_counts[entity_type] += 1
        else:
            turns_without_entities += 1
    
    print(f"\nExtraction Results:")
    print(f"  Total turns processed: {total_turns}")
    print(f"  Turns with entities: {turns_with_entities} ({turns_with_entities/total_turns*100:.1f}%)")
    print(f"  Turns without entities: {turns_without_entities} ({turns_without_entities/total_turns*100:.1f}%)")
    print(f"  Total entities extracted: {total_entities}")
    if turns_with_entities > 0:
        print(f"  Average entities per turn (with entities): {total_entities/turns_with_entities:.2f}")
    
    # Entity type breakdown
    if entity_type_counts:
        print(f"\nEntity Types Distribution:")
        sorted_types = sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)
        for entity_type, count in sorted_types:
            print(f"  {entity_type}: {count} ({count/total_entities*100:.1f}%)")
    
    # Verification statistics
    if not skip_verification and verification_results:
        total_verified = len(verification_results)
        verified_true = 0
        verified_false = 0
        verified_unknown = 0
        correct_classification = 0
        incorrect_classification = 0
        
        for custom_id, verification_data in verification_results.items():
            verified = verification_data.get("verified")
            correct = verification_data.get("correct_entity_type_classification")
            
            if verified is True:
                verified_true += 1
            elif verified is False:
                verified_false += 1
            else:
                verified_unknown += 1
            
            if correct is True:
                correct_classification += 1
            elif correct is False:
                incorrect_classification += 1
        
        print(f"\nVerification Results:")
        print(f"  Total entities verified: {total_verified}")
        print(f"  Verified as TRUE: {verified_true} ({verified_true/total_verified*100:.1f}%)")
        print(f"  Verified as FALSE: {verified_false} ({verified_false/total_verified*100:.1f}%)")
        if verified_unknown > 0:
            print(f"  Unknown/Error: {verified_unknown} ({verified_unknown/total_verified*100:.1f}%)")
    
    print("="*60 + "\n")


# =========================
# BATCH API FUNCTIONS
# =========================

def check_existing_results(df: pd.DataFrame):
    """Check which messages already have result files and return filtered dataframe"""
    existing_count = 0
    new_indices = []
    
    print("\nChecking for existing result files...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking existing files"):
        user_id = row["user_id"]
        message_id = row["message_id"]
        
        output_path = os.path.join(OUTPUT_DIR, f"user-{user_id}_msg-{message_id}.json")
        if os.path.exists(output_path):
            existing_count += 1
        else:
            new_indices.append(idx)
    
    df_new = df.loc[new_indices].copy() if new_indices else pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"FILE CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Total messages in CSV: {len(df)}")
    print(f"Already processed: {existing_count}")
    print(f"New messages to annotate: {len(df_new)}")
    print(f"{'='*60}\n")
    
    return df_new, existing_count


def create_extraction_batch_file(df: pd.DataFrame, batch_input_path: str):
    """Create JSONL file for extraction batch requests"""
    request_count = 0
    
    with open(batch_input_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            user_id = row["user_id"]
            conversation_id = row["conversation_id"]
            message_id = row["message_id"]
            message_content = str(row["message_content"])
            
            # Create batch request with unique counter to prevent duplicates
            # Using idx ensures every request has a unique custom_id
            request = {
                "custom_id": f"extract_{user_id}_{conversation_id}_{message_id}_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "response_format": EXTRACTION_JSON_SCHEMA,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise entity extraction system."
                        },
                        {
                            "role": "user",
                            "content": PROMPT_ENTITY_EXTRACTION + "\n\nTEXT:\n" + message_content
                        }
                    ],
                    "temperature": 0
                }
            }
            f.write(json.dumps(request) + '\n')
            request_count += 1
    
    print(f"Created extraction batch file with {request_count} requests: {batch_input_path}")
    return request_count


def create_verification_batch_file(extraction_results: dict, df: pd.DataFrame, batch_input_path: str):
    """Create JSONL file for verification batch requests"""
    request_count = 0
    
    with open(batch_input_path, 'w', encoding='utf-8') as f:
        for custom_id, entity_data in extraction_results.items():
            # Parse custom_id to get identifiers
            parts = custom_id.split('_')
            user_id = parts[1]
            conversation_id = parts[2]
            message_id = parts[3]
            # parts[4] is the idx we added for uniqueness
            
            # Get original message content
            message_row = df[
                (df["user_id"] == int(user_id)) & 
                # (df["conversation_id"] == conversation_id) &
                (df["message_id"] == message_id)
            ]
            
            if message_row.empty:
                continue
            
            message_content = str(message_row.iloc[0]["message_content"])
            entities = entity_data.get("entities", [])
            
            # Skip if no entities extracted
            if not entities or len(entities) == 0:
                continue
            
            for idx, entity in enumerate(entities):
                entity_text = entity["entity_text"]
                entity_type_category = entity["entity_type_category"]
                
                prompt = PROMPT_ENTITY_VERIFICATION.format(
                    original_text=message_content,
                    entity_text=entity_text,
                    entity_type_category=entity_type_category
                )
                
                # Ensure unique custom_id by using both the original custom_id and entity idx
                request = {
                    "custom_id": f"{custom_id}_entity_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "response_format": VERIFICATION_JSON_SCHEMA,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an entity verification expert with access to real-world knowledge."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0
                    }
                }
                f.write(json.dumps(request) + '\n')
                request_count += 1
    
    print(f"Created verification batch file with {request_count} requests: {batch_input_path}")
    return request_count


def submit_batch(batch_input_path: str, description: str):
    """Upload batch file and create batch job"""
    # Upload the file
    with open(batch_input_path, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"Uploaded batch file: {batch_input_file.id}")
    
    # Create batch job
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    
    print(f"Created batch job: {batch.id}")
    return batch


def wait_for_batch(batch_id: str):
    """Wait for batch to complete and return final status"""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        total = batch.request_counts.total if batch.request_counts.total else 0
        completed = batch.request_counts.completed if batch.request_counts.completed else 0
        
        print(f"Status: {status} | Completed: {completed}/{total}")
        
        if status == "completed":
            print("Batch completed successfully!")
            return batch
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch ended with status: {status}")
            if batch.errors:
                print(f"Errors: {batch.errors}")
            return batch
        
        time.sleep(CHECK_INTERVAL)


def download_batch_results(batch: object, output_path: str):
    """Download batch results to a file"""
    if not batch.output_file_id:
        print("No output file available")
        return {}
    
    # Download the results
    file_response = client.files.content(batch.output_file_id)
    
    with open(output_path, 'wb') as f:
        f.write(file_response.content)
    
    print(f"Downloaded results to: {output_path}")
    
    # Parse results
    results = {}
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            
            if result.get("response") and result["response"].get("body"):
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    results[custom_id] = json.loads(content)
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing result for {custom_id}: {e}")
                    continue
    
    return results

# =========================
# MAIN PIPELINE WITH BATCH SIZE CONSTRAINT
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)
    required_cols = {"user_id", "conversation_id", "message_id", "message_role", "message_content"}
    
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Filter out rows with missing or empty message_content
    df = df[~df["message_content"].isna()]
    df = df[df["message_content"].str.strip() != ""]
    
    # Only keep the user query and the assistant response
    df = df[df['message_role'].isin(['user', 'assistant'])]
    
    # Test
    # df = df[:20]
    
    print(f'Number of sequences to annotate: {len(df)}')
    
    # =========================
    # CHECK FOR EXISTING RESULTS
    # =========================
    df_new, existing_count = check_existing_results(df)
    
    # If no new messages to process, exit early
    if len(df_new) == 0:
        print("All messages have already been processed. Nothing to do!")
        return
    
    # =========================
    # BATCH SIZE CONSTRAINT
    # =========================
    MAX_BATCH_SIZE = 25000
    total_rows = len(df_new)
    num_chunks = (total_rows + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE  # Ceiling division
    
    print(f"\nProcessing {total_rows} NEW rows in {num_chunks} chunk(s) (max {MAX_BATCH_SIZE} per batch)")
    
    all_extraction_results = {}
    all_verification_results = {}
    
    # Process data in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * MAX_BATCH_SIZE
        end_idx = min((chunk_idx + 1) * MAX_BATCH_SIZE, total_rows)
        df_chunk = df_new.iloc[start_idx:end_idx].copy()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING CHUNK {chunk_idx + 1}/{num_chunks}")
        print(f"Rows: {start_idx} to {end_idx - 1} ({len(df_chunk)} rows)")
        print(f"{'='*60}")
        
        # =========================
        # STEP 1: EXTRACTION BATCH
        # =========================
        extraction_batch_input = os.path.join(BATCH_DIR, f"extraction_batch_input_chunk_{chunk_idx}.jsonl")
        extraction_batch_output = os.path.join(BATCH_DIR, f"extraction_batch_output_chunk_{chunk_idx}.jsonl")
        
        print("\n=== STEP 1: Entity Extraction ===")
        request_count = create_extraction_batch_file(df_chunk, extraction_batch_input)
        
        if request_count == 0:
            print(f"No new messages to process for extraction in chunk {chunk_idx + 1}.")
            continue
        
        # Submit extraction batch
        extraction_batch = submit_batch(extraction_batch_input, f"Entity extraction batch - Chunk {chunk_idx + 1}/{num_chunks}")
        extraction_batch = wait_for_batch(extraction_batch.id)
        
        # Download and parse results
        extraction_results = download_batch_results(extraction_batch, extraction_batch_output)
        all_extraction_results.update(extraction_results)
        
        print(f"Extracted entities for {len(extraction_results)} messages in chunk {chunk_idx + 1}")
        
        # =========================
        # STEP 2: VERIFICATION BATCH (Optional)
        # =========================
        if not SKIP_VERIFICATION and extraction_results:
            print("\n=== STEP 2: Entity Verification ===")
            verification_batch_input = os.path.join(BATCH_DIR, f"verification_batch_input_chunk_{chunk_idx}.jsonl")
            verification_batch_output = os.path.join(BATCH_DIR, f"verification_batch_output_chunk_{chunk_idx}.jsonl")
            
            request_count = create_verification_batch_file(extraction_results, df_chunk, verification_batch_input)
            
            if request_count == 0:
                print(f"No entities to verify in chunk {chunk_idx + 1}.")
                verification_results = {}
            else:
                # Check if verification batch exceeds limit
                if request_count > MAX_BATCH_SIZE:
                    print(f"WARNING: Verification batch has {request_count} requests, splitting...")
                    # You may need to implement further splitting for verification if needed
                    # For now, we'll process it as-is with a warning
                
                # Submit verification batch
                verification_batch = submit_batch(verification_batch_input, f"Entity verification batch - Chunk {chunk_idx + 1}/{num_chunks}")
                verification_batch = wait_for_batch(verification_batch.id)
                
                # Download and parse results
                verification_results = download_batch_results(verification_batch, verification_batch_output)
                all_verification_results.update(verification_results)
                
                print(f"Verified {len(verification_results)} entities in chunk {chunk_idx + 1}")
        else:
            verification_results = {}
    
    # =========================
    # STEP 3: COMBINE AND SAVE RESULTS
    # =========================
    print("\n" + "="*60)
    print("=== STEP 3: Saving Results ===")
    print("="*60)
    
    for custom_id, entity_data in tqdm(all_extraction_results.items(), desc="Saving results"):
        # Parse custom_id to get identifiers
        parts = custom_id.split('_')
        user_id = parts[1]
        conversation_id = parts[2]
        message_id = parts[3]
        # parts[4] is the idx we added for uniqueness
        
        # Get original message content from df_new
        message_row = df_new[
            (df_new["user_id"] == int(user_id)) & 
            # (df_new["conversation_id"] == conversation_id) &
            (df_new["message_id"] == message_id)
        ]
        
        # If not found in df_new, try the original df
        if message_row.empty:
            message_row = df[
                (df["user_id"] == int(user_id)) & 
                # (df["conversation_id"] == conversation_id) &
                (df["message_id"] == message_id)
            ]
        
        if message_row.empty:
            print(f"It's not matched for {parts}")
            continue
        
        message_content = str(message_row.iloc[0]["message_content"])
        
        entities = entity_data.get("entities", [])
        
        # Add verification results if available
        if not SKIP_VERIFICATION:
            for idx, entity in enumerate(entities):
                verification_key = f"{custom_id}_entity_{idx}"
                # print(f'Verification key:{verification_key}')
                if verification_key in all_verification_results:
                    entity["verification"] = all_verification_results[verification_key]
                else:
                    print(f'No Match for the verification key : {verification_key}')
                    entity["verification"] = {
                        "verified": False,
                        "correct_entity_type_classification": None,
                        "reason": "Verification failed"
                    }
        else:
            for entity in entities:
                entity["verification"] = {
                    "verified": None,
                    "correct_entity_type_classification": None,
                    "reason": "Verification skipped"
                }
        
        # Save to output file (same format as original code)
        output_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "message_content": message_content,
            "entities": entities
        }
        
        output_path = os.path.join(OUTPUT_DIR, f"user-{user_id}_msg-{message_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # =========================
    # STEP 4: PRINT STATISTICS
    # =========================
    print_statistics(all_extraction_results, all_verification_results, SKIP_VERIFICATION)
    
    print(f"\nComplete! Results saved to {OUTPUT_DIR}")
    print(f"Total files created: {len(all_extraction_results)}")
    print(f"Total messages processed (including existing): {existing_count + len(all_extraction_results)}")


if __name__ == "__main__":
    main()