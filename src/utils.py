"""
Utility functions for file I/O, chat templates, and result writing.
"""

import os
import json
import pandas as pd
from typing import Dict, Optional, List
from jinja2 import Template


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV or JSON file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        DataFrame with data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")
    
    # Ensure entity_text column exists
    if 'entity_text' not in df.columns and 'entity' in df.columns:
        df['entity_text'] = df['entity']
    
    if 'entity_text' not in df.columns:
        raise ValueError("File must contain 'entity_text' or 'entity' column")
    
    return df


def load_chat_template(template_path: Optional[str]) -> Optional[Template]:
    """
    Load a Jinja2 chat template from file.
    
    Args:
        template_path: Path to the .jinja template file
        
    Returns:
        Jinja2 Template object or None if path is not provided
    """
    if not template_path:
        return None
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Chat template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_str = f.read()
    
    return Template(template_str)


def apply_chat_template(
    template: Optional[Template],
    prompt: str,
    system_message: Optional[str] = None
) -> str:
    """
    Apply chat template to format the prompt.
    
    Args:
        template: Jinja2 Template object (or None to skip formatting)
        prompt: The user prompt text
        system_message: Optional system message
        
    Returns:
        Formatted prompt string
    """
    if template is None:
        return prompt
    
    # Create messages in ChatML format
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    # Render the template
    try:
        formatted = template.render(messages=messages, add_generation_prompt=True)
        return formatted
    except Exception as e:
        print(f"Warning: Failed to apply chat template: {e}")
        return prompt


class ResultWriter:
    """
    Streams results to disk safely (append-only JSONL format).
    """
    
    def __init__(self, output_dir: str, base_name: str):
        """
        Initialize result writer.
        
        Args:
            output_dir: Directory to save results
            base_name: Base name for output file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.jsonl_path = os.path.join(output_dir, f"{base_name}.jsonl")
        
        # Open JSONL in line-buffered mode
        self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
    
    def write_one(self, row: Dict):
        """
        Write a single result row to file.
        
        Args:
            row: Dictionary containing result data
        """
        self._jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_f.flush()
        os.fsync(self._jsonl_f.fileno())
    
    def write_batch(self, rows: List[Dict]):
        """
        Write multiple rows at once.
        
        Args:
            rows: List of result dictionaries
        """
        for row in rows:
            self.write_one(row)
    
    def close(self):
        """Close the file handle."""
        try:
            self._jsonl_f.close()
        except Exception as e:
            print(f"Warning: Error closing file: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()