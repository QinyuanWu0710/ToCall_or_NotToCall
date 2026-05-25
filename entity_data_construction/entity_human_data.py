"""
Simplified script to export extracted entities to CSV.

No statistics or figure generation - just data export.

Data sources:
- Annotation sample CSV:
  /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/annotation_sample_100_conversations.csv
- Extracted entity JSONs:
  /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_entities_gpt-5.1_test_batch
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# configuration & typing
# -----------------------------------------------------------------------------

MessageInfo = Tuple[str, str, str]  # (user_id, conversation_id, message_id)


@dataclass(frozen=True)
class Config:
    raw_data_csv: Path
    entity_data_dir: Path
    output_dir: Path


CONFIG = Config(
    raw_data_csv=Path(
        "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/"
        "annotation_sample_100_conversations.csv"
    ),
    entity_data_dir=Path(
        "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/"
        "extracted_entities_gpt-5.1_test_batch"
    ),
    output_dir=Path(
        "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/"
        "exported_entities_human_verify"
    ),
)


ENTITY_TYPE_MAP: Dict[str, str] = {
    'Computing-Related Entities': 'Computing-Related',
    'NotAvailable': 'UNKNOWN',
    'NotSpecified': 'UNKNOWN',
    'INVALID': 'UNKNOWN',
    'Concept': 'UNKNOWN',
    'Other': 'UNKNOWN',
    'NotInList': 'UNKNOWN',
    'UNMAPPED': 'UNKNOWN',
    'Person': 'People & Personal Attributes',
    'NotApplicable': 'UNKNOWN',
    'Financial': 'Finance-related',
    'Date & Time': 'Time-Related',
    'PhysicalSciences': 'Physical Sciences',
    'Finance-related Entities': 'Finance-related',
    'Physical-Object': 'Physical Sciences',
    'Organization': 'UNKNOWN',
    'NOT_IN_LIST': 'UNKNOWN',
    'People & PersonalAttributes': 'People & Personal Attributes',
    'Time & Date': 'Time-Related',
    'Disease': 'Medical Entities',
    'Uncategorized': 'UNKNOWN',
    'Education': 'UNKNOWN',
    'Concepts': 'UNKNOWN',
    '(none)': 'UNKNOWN',
    'FileFormat': 'UNKNOWN',
    '(none of the listed high-level categories applied)': 'UNKNOWN',
    'Unclassified': 'UNKNOWN',
    'Time-related': 'Time-Related',
    'Physical Object': 'Physical Sciences',
    'Physical Systems': 'Physical Sciences',
    'Gender': 'People & Personal Attributes',
    'Weather  & Earth Science': 'Weather & Earth Science',
    'PhysicalSystem': 'Physical Sciences',
    'Unknown': 'UNKNOWN',
    'NOT_FOUND_IN_SCHEMA': 'UNKNOWN',
    'Time Entities': 'Time-Related'
}

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def remap_category(raw_category: Optional[str]) -> Optional[str]:
    if raw_category is None:
        return None
    return ENTITY_TYPE_MAP.get(raw_category, raw_category)


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------


def load_message_info(csv_path: Path) -> Tuple[List[MessageInfo], List[MessageInfo]]:
    """
    Read the annotation sample CSV and return structured message info.

    Returns:
        user_info_list: list of (user_id, conversation_id, message_id)
        assistant_info_list: list of (user_id, conversation_id, message_id)
    """
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {
        "user_id",
        "conversation_id",
        "message_id",
        "message_role",
        "message_content",
    }

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df = df.dropna(subset=["message_content"])
    df = df[df["message_content"].str.strip() != ""]
    df = df[df["message_role"].isin(["user", "assistant"])]

    user_info: List[MessageInfo] = []
    assistant_info: List[MessageInfo] = []

    for _, row in df.iterrows():
        info: MessageInfo = (
            str(row["user_id"]),
            str(row["conversation_id"]),
            str(row["message_id"]),
            str(row["message_content"])
        )

        if row["message_role"] == "user":
            user_info.append(info)
        else:
            assistant_info.append(info)

    logger.info(
        f"Loaded {len(user_info)} user messages and "
        f"{len(assistant_info)} assistant messages"
    )
    return user_info, assistant_info


# -----------------------------------------------------------------------------
# entity export
# -----------------------------------------------------------------------------

def export_entities_to_csv(
    info_list: List[MessageInfo],
    role_label: str,
    entity_dir: Path,
    output_dir: Path,
) -> None:
    """
    Export unique entities (deduped by entity_text + remapped entity_category) to CSV.
    Keeps one example row plus occurrence count and optional ID lists.
    """
    ensure_dir(output_dir)

    # key: (normalized_entity_text, remapped_category)
    unique_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    def normalize_entity_text(s: str) -> str:
        # basic normalization: strip + collapse whitespace
        return " ".join(s.strip().split())

    for user_id, conversation_id, message_id, message_content in tqdm(
        info_list, desc=f"Processing {role_label} entity JSONs"
    ):
        json_path = entity_dir / f"user-{user_id}_msg-{message_id}.json"

        if not json_path.exists():
            logger.warning(f"Missing JSON: {json_path.name}")
            continue

        data = safe_load_json(json_path)
        if not data:
            continue

        entities = data.get("entities")
        if not isinstance(entities, list):
            continue

        for ent in entities:
            if not isinstance(ent, dict):
                continue

            text = ent.get("entity_text")
            raw_category = ent.get("entity_type_category")
            category = remap_category(raw_category)

            if not text or not category:
                continue

            norm_text = normalize_entity_text(text)
            key = (norm_text, category)

            if key not in unique_map:
                unique_map[key] = {
                    "entity_text": norm_text,
                    "entity_category": category,
                    "raw_category": raw_category or "",
                    "role": role_label,
                    "occurrence_count": 1,

                    # keep a concrete example for human verification
                    "example_user_id": user_id,
                    "example_conversation_id": conversation_id,
                    "example_message_id": message_id,
                    "example_message_content": message_content,

                    # optional traceability: collect all IDs (can be removed if too large)
                    "user_ids": {user_id},
                    "conversation_ids": {conversation_id},
                    "message_ids": {message_id},
                }
            else:
                rec = unique_map[key]
                rec["occurrence_count"] = int(rec["occurrence_count"]) + 1
                rec["user_ids"].add(user_id)
                rec["conversation_ids"].add(conversation_id)
                rec["message_ids"].add(message_id)

    if unique_map:
        csv_path = output_dir / f"{role_label}_unique_entities.csv"

        # flatten sets for CSV
        rows: List[Dict[str, object]] = []
        for rec in unique_map.values():
            rows.append({

                # optional: aggregated ID lists
                "user_ids": ";".join(sorted(rec["user_ids"])),
                "conversation_ids": ";".join(sorted(rec["conversation_ids"])),
                "message_ids": ";".join(sorted(rec["message_ids"])),

                "role": rec["role"],
                "message_content": rec["example_message_content"],
                "entity_text": rec["entity_text"],
                "entity_category": rec["entity_category"],
                

                "occurrence_count": rec["occurrence_count"],

            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Exported {len(df)} unique entities to {csv_path}")
    else:
        logger.warning(f"No entities found for {role_label}")


# -----------------------------------------------------------------------------
# argument parser
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export extracted entities to CSV files"
    )
    
    parser.add_argument(
        "--raw-data-csv",
        type=Path,
        default=CONFIG.raw_data_csv,
        help=f"Path to the raw data CSV file (default: {CONFIG.raw_data_csv})"
    )
    
    parser.add_argument(
        "--entity-data-dir",
        type=Path,
        default=CONFIG.entity_data_dir,
        help=f"Path to the entity data directory (default: {CONFIG.entity_data_dir})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.output_dir,
        help=f"Path to the output directory (default: {CONFIG.output_dir})"
    )
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_arguments()
    
    config = Config(
        raw_data_csv=args.raw_data_csv,
        entity_data_dir=args.entity_data_dir,
        output_dir=args.output_dir,
    )
    
    user_info, assistant_info = load_message_info(config.raw_data_csv)

    export_entities_to_csv(
        info_list=user_info,
        role_label="user",
        entity_dir=config.entity_data_dir,
        output_dir=config.output_dir,
    )

    export_entities_to_csv(
        info_list=assistant_info,
        role_label="assistant",
        entity_dir=config.entity_data_dir,
        output_dir=config.output_dir,
    )

    logger.info("Export complete!")


if __name__ == "__main__":
    main()