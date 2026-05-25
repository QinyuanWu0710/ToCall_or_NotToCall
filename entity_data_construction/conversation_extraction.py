

'''
This file is used to extract the conversation from all users' raw data.
'''

import os
import json
import csv
from typing import List, Dict, Any

import random
from collections import defaultdict

def extract_messages_iterative(
    mapping: Dict[str, Any],
    start_node_id: str,
    conversation_data: Dict[str, Any],
    user_id: int,
    results: List[Dict[str, Any]]
):
    stack = [(start_node_id, 0)]
    visited = set()

    while stack:
        node_id, message_index = stack.pop()

        if node_id in visited or node_id not in mapping:
            continue

        visited.add(node_id)
        node = mapping[node_id]

        if node.get('message'):
            message = node['message']
            name = message.get('author', {}).get('name', '')
            role = message.get('author', {}).get('role', '')
            language = message.get('author', {}).get('language', '')
            content_type = message.get('author', {}).get('content_type', '')
            content_parts = message.get('content', {}).get('parts', []) or []
            model_slug = message.get('metadata',{}).get('model_slug', '')
            urls = message.get('metadata',{}).get('safe_urls', '')
            children = node.get('children', [])
            children_str = ','.join(children)

            if content_parts:
                for content in content_parts:
                    results.append({
                        'user_id': user_id,
                        'conversation_id': conversation_data['id'],
                        'conversation_title': conversation_data['title'],
                        'conversation_urls': urls,
                        'message_index': message_index,
                        'message_id': node_id,
                        'role_name': name,
                        'message_role': role,
                        'message_content': str(content),
                        'language': language,
                        'content_type': content_type,
                        'model': model_slug,
                        'message_child': children_str
                    })
            else:
                results.append({
                    'user_id': user_id,
                    'conversation_id': conversation_data['id'],
                    'conversation_title': conversation_data['title'],
                    'conversation_urls': urls,
                    'message_index': message_index,
                    'message_id': node_id,
                    'role_name': name,
                    'message_role': role,
                    'message_content': '',
                    'language': language,
                    'content_type': content_type,
                    'model': model_slug,
                    'message_child': children_str
                })

            message_index += 1

        # Push children (reverse keeps original order)
        for child_id in reversed(node.get('children', [])):
            stack.append((child_id, message_index))


def process_user_file(user_id: int, file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single user's conversation file.
    
    Args:
        user_id: User ID
        file_path: Path to the user's JSON file
    
    Returns:
        List of extracted message dictionaries
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Process each conversation
        for conversation in conversations:
            conversation_data = {
                'id': conversation.get('id', ''),
                'title': conversation.get('title', ''),
                'safe_urls': conversation.get('safe_urls', ''),
                'create_time': conversation.get('create_time', '')
            }
            
            # Get the mapping
            mapping = conversation.get('mapping', {})
            
            if not mapping:
                continue
            
            # Find the root node
            if 'client-created-root' not in mapping:
                continue
            
            root_node = mapping['client-created-root']
            children = root_node.get('children', [])
            
            if not children:
                continue
            
            # Start processing from the first child of root
            first_message_id = children[0]
            # extract_messages(mapping, first_message_id, conversation_data, 
                        #    user_id, message_index=0, results=results)
            extract_messages_iterative(
                                        mapping,
                                        first_message_id,
                                        conversation_data,
                                        user_id,
                                        results
                                    )
    
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
    
    return results


def process_wild_chat_user_file(user_id: int, file_path: str) -> List[Dict[str, Any]]:
    """
    Process a single user's conversation file.
    
    Args:
        user_id: User ID
        file_path: Path to the user's JSON file
    
    Returns:
        List of extracted message dictionaries
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # print(conversations[0])
        # exit()
        # Process each conversation
        for conversation in conversations:
            for i, conversation_data in enumerate(conversation['conversation']):
                results.append(
                    {
                        'user_id':user_id,
                        'conversation_id': conversation['conversation_hash'],
                        'message_index': i,
                        'message_id':  conversation_data['turn_identifier'],
                        'message_role': conversation_data['role'],
                        'message_content': conversation_data['content'],
                        
                    }
                )

    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
    
    return results

def sample_conversations_for_annotation(
    all_results: List[Dict[str, Any]],
    sample_size: int,
    output_file: str
):
    """
    Randomly sample conversations for human annotation and save to CSV.

    Args:
        all_results: All extracted message rows
        sample_size: Number of conversations to sample
        output_file: Path to output CSV
    """
    # Group rows by conversation_id
    conv_dict = defaultdict(list)
    for row in all_results:
        conv_dict[row['conversation_id']].append(row)

    conversation_ids = list(conv_dict.keys())
    if not conversation_ids:
        print("No conversations available for sampling.")
        return

    # Sample conversation IDs
    sampled_ids = random.sample(
        conversation_ids,
        min(sample_size, len(conversation_ids))
    )

    # Collect sampled rows
    sampled_rows = []
    for cid in sampled_ids:
        sampled_rows.extend(conv_dict[cid])

    fieldnames = [
        'user_id', 'conversation_id', 'conversation_title',
        'conversation_urls', 'message_index', 'message_id','content_type', 'language', 'model',
        'message_role', 'role_name','message_content', 'message_child'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Sampled {len(sampled_ids)} conversations "
          f"({len(sampled_rows)} messages) saved to:")
    print(output_file)

def main():
    """
    Main function to process all user files and generate CSV output.
    """
    dataset_name = 'invivo_gpt'
    if dataset_name == 'invivo_gpt':
        RAW_DATA_PATH = '/NS/chatgpt/work/data/prolific_all_files'
        all_results = []
        
        # Process users from 0 to 160
        for user_id in range(310):
            file_name = 'conversations.json'
            file_path = os.path.join(RAW_DATA_PATH, f'user_{user_id}', file_name)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Processing user {user_id}...")
            user_results = process_user_file(user_id, file_path)
            all_results.extend(user_results)
        
        # Write results to CSV
        output_file = '/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_conversations.csv'
        
        if all_results:
            fieldnames = ['user_id', 'conversation_id', 'conversation_title', 'conversation_urls',
                        'message_index', 'message_id', 'message_role','role_name', 'message_content', 'message_child', 'content_type', 'language', 'model']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"\nExtraction complete! {len(all_results)} messages extracted.")
            print(f"Output saved to: {output_file}")
        else:
            print("No data extracted.")

        # Sample 100 conversations for human annotation
        annotation_output_file = (
            '/NS/chatgpt/work/qwu/hallucinations_detection/data/'
            'all_users/annotation_sample_100_conversations.csv'
        )

        sample_conversations_for_annotation(
            all_results=all_results,
            sample_size=100,
            output_file=annotation_output_file
        )

    elif dataset_name == 'wildchat':
        RAW_DATA_PATH = '/NS/chatgpt/work/data/WildChat/raw'
        all_results = []
        
        # Process users from 0 to 309
        for user_id in range(160):
            file_path = os.path.join(RAW_DATA_PATH, f'user_{user_id}.json')
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Processing user {user_id}...")
            user_results = process_wild_chat_user_file(user_id, file_path)
            all_results.extend(user_results)
        
        # Write results to CSV
        output_file = '/NS/chatgpt/work/qwu/hallucinations_detection/data/wildchat/extracted_conversations.csv'
        
        if all_results:
            fieldnames = ['user_id', 'conversation_id', 
                        'message_index', 'message_id', 'message_role', 'message_content']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"\nExtraction complete! {len(all_results)} messages extracted.")
            print(f"Output saved to: {output_file}")
        else:
            print("No data extracted.")

        # Sample 100 conversations for human annotation
        annotation_output_file = (
            '/NS/chatgpt/work/qwu/hallucinations_detection/data/'
            'wildchat/annotation_sample_100_conversations.csv'
        )

        sample_conversations_for_annotation(
            all_results=all_results,
            sample_size=100,
            output_file=annotation_output_file
        )


if __name__ == '__main__':
    main()