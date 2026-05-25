"""
Refined script for statistical analysis of extracted entities.

Adds CSV caching:
- If statistics/{role}_unique_entities.csv exists, skip JSON processing and
  compute plots/stats directly from the CSV.
- Otherwise, process JSONs, write the CSV, then plot.

Also stores provenance per unique (entity_text, category):
- raw_data_0, raw_data_1, ... columns
- each cell is a JSON string of:
  {"user_id": ..., "conversation_id": ..., "message_id": ..., "message_content": ...}

Data sources:
- Conversations CSV:
  /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_conversations.csv
- Extracted entity JSONs:
  /NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/extracted_entities_gpt-5.1_batch
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
from tqdm import tqdm
from datasets import load_dataset

# -----------------------------------------------------------------------------
# configuration & typing
# -----------------------------------------------------------------------------

MessageInfo = Tuple[str, str, str]  # (user_id, conversation_id, message_id)


@dataclass(frozen=True)
class Config:
    raw_data_csv: Path
    entity_data_dir: Path
    stats_dir_name: str = "statistics"


CONFIG = Config(
    raw_data_csv=Path(
        "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/"
        "extracted_conversations.csv"
    ),
    entity_data_dir=Path(
        "/NS/chatgpt/work/qwu/hallucinations_detection/data/all_users/"
        "extracted_entities_gpt-5.1_batch"
    ),
)

ENTITY_TYPE_MAP: Dict[str, str] = {
    "Computing-Related Entities": "Computing-Related",
    "NotAvailable": "UNKNOWN",
    "NotSpecified": "UNKNOWN",
    "INVALID": "UNKNOWN",
    "Concept": "UNKNOWN",
    "Other": "UNKNOWN",
    "NotInList": "UNKNOWN",
    "UNMAPPED": "UNKNOWN",
    "Person": "People & Personal Attributes",
    "NotApplicable": "UNKNOWN",
    "Financial": "Finance-related",
    "Date & Time": "Time-Related",
    "PhysicalSciences": "Physical Sciences",
    "Finance-related Entities": "Finance-related",
    "Physical-Object": "Physical Sciences",
    "Organization": "UNKNOWN",
    "NOT_IN_LIST": "UNKNOWN",
    "People & PersonalAttributes": "People & Personal Attributes",
    "Time & Date": "Time-Related",
    "Disease": "Medical Entities",
    "Uncategorized": "UNKNOWN",
    "Education": "UNKNOWN",
    "Concepts": "UNKNOWN",
    "(none)": "UNKNOWN",
    "FileFormat": "UNKNOWN",
    "(none of the listed high-level categories applied)": "UNKNOWN",
    "Unclassified": "UNKNOWN",
    "Time-related": "Time-Related",
    "Physical Object": "Physical Sciences",
    "Physical Systems": "Physical Sciences",
    "Gender": "People & Personal Attributes",
    "Weather  & Earth Science": "Weather & Earth Science",
    "PhysicalSystem": "Physical Sciences",
    "Unknown": "UNKNOWN",
    "NOT_FOUND_IN_SCHEMA": "UNKNOWN",
    "Time Entities": "Time-Related",
    "No matching category": "UNKNOWN",
    "ProgrammingLanguage": "Computing-Related",
    "None": "UNKNOWN",
    "Time": "Time-Related",
    "PhysicalObject": "Physical Sciences",
    "NotProvided": "UNKNOWN",
    "Time & Dates": "Time-Related",
    "NotableComputer": "Computing-Related",
    "nan": "UNKNOWN",
    "(none of the listed high-level categories apply)": "UNKNOWN",
    "NOT_FOUND": "UNKNOWN",
    "string": "UNKNOWN",
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


def save_fig(fig, path_base: Path) -> None:
    """Save a plotly figure as both PNG and HTML."""
    fig.write_image(str(path_base.with_suffix(".png")))
    fig.write_html(str(path_base.with_suffix(".html")))


def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def remap_category(raw_category: Any) -> Optional[str]:
    if raw_category is None:
        return None

    if isinstance(raw_category, str):
        raw_category = raw_category.strip()
        if not raw_category:
            return None
        return ENTITY_TYPE_MAP.get(raw_category, raw_category)

    if isinstance(raw_category, dict):
        for k in ("category", "label", "type", "name", "value"):
            v = raw_category.get(k)
            if isinstance(v, str) and v.strip():
                v = v.strip()
                return ENTITY_TYPE_MAP.get(v, v)
        return "UNKNOWN"

    if isinstance(raw_category, (list, tuple)):
        for item in raw_category:
            mapped = remap_category(item)
            if mapped is not None:
                return mapped
        return "UNKNOWN"

    return ENTITY_TYPE_MAP.get(str(raw_category), str(raw_category))


def raw_data_dict(
    user_id: Optional[str],
    conversation_id: Optional[str],
    message_id: Optional[str],
    message_content: Optional[str],
) -> Dict[str, Optional[str]]:
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "message_content": message_content,
    }

# -----------------------------------------------------------------------------
# CSV cache helpers
# -----------------------------------------------------------------------------

def load_entity_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load cached entity CSV (requires entity_text, category, count). Extra columns are allowed."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except OSError:
        return None

    required = {"entity_text", "category", "count"}
    if not required.issubset(df.columns):
        logger.warning(f"Cached CSV missing required columns {required}: {csv_path}")
        return None

    df["entity_text"] = df["entity_text"].astype(str)
    df["category"] = df["category"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df[df["count"] > 0].copy()
    return df


def plot_counter_dict(
    counter_dict: Dict[str, int],
    title: str,
    filename: str,
    stats_dir: Path,
) -> None:
    if not counter_dict:
        return
    items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)
    x, y = zip(*items)
    fig = px.bar(
        x=x,
        y=y,
        labels={"x": "Entity Type Category", "y": "Count"},
        title=title,
    )
    save_fig(fig, stats_dir / filename)


def compute_stats_and_plots_from_entity_csv(
    df: pd.DataFrame,
    role_label: str,
    stats_dir: Path,
) -> Dict[str, object]:
    """Compute totals + distributions + plots directly from cached CSV."""
    dist_all: Dict[str, int] = (
        df.groupby("category")["count"].sum().astype(int).to_dict()
    )
    dist_unique: Dict[str, int] = (
        df.groupby("category").size().astype(int).to_dict()
    )

    total_entities = int(df["count"].sum())
    unique_entities = int(len(df))

    logger.info(f"[{role_label.upper()}] (FROM CSV) Total entities: {total_entities}")
    logger.info(f"[{role_label.upper()}] (FROM CSV) Unique entities: {unique_entities}")

    plot_counter_dict(
        dist_all,
        f"{role_label.capitalize()} Entity Type Distribution (All)",
        f"{role_label}_entity_type_distribution_all",
        stats_dir,
    )
    plot_counter_dict(
        dist_unique,
        f"{role_label.capitalize()} Entity Type Distribution (Unique)",
        f"{role_label}_entity_type_distribution_unique",
        stats_dir,
    )

    return {
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "verification_accuracy": None,
        "entity_type_distribution_all": dist_all,
        "entity_type_distribution_unique": dist_unique,
        "verified_entity_type_distribution": {},
    }

# -----------------------------------------------------------------------------
# conversation CSV loading
# -----------------------------------------------------------------------------

def load_message_info(csv_path: Path) -> Tuple[List[MessageInfo], List[MessageInfo]]:
    """
    Read the conversation CSV and return structured message info.

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
    df["message_content"] = df["message_content"].astype(str)
    df = df[df["message_content"].str.strip() != ""]
    df = df[df["message_role"].isin(["user", "assistant"])]

    user_info: List[MessageInfo] = []
    assistant_info: List[MessageInfo] = []

    for _, row in df.iterrows():
        info: MessageInfo = (
            str(row["user_id"]),
            str(row["conversation_id"]),
            str(row["message_id"]),
        )
        if row["message_role"] == "user":
            user_info.append(info)
        else:
            assistant_info.append(info)

    logger.info(f"Loaded {len(user_info)} user messages and {len(assistant_info)} assistant messages")
    return user_info, assistant_info

# -----------------------------------------------------------------------------
# entity statistics
# -----------------------------------------------------------------------------

def compute_entity_statistics(
    info_list: List[MessageInfo],
    role_label: str,
    verified: bool,
    entity_dir: Path,
) -> Dict[str, Optional[Dict]]:
    """
    Compute statistics over extracted entities, generate plots,
    and save entity pair counts with remapped category labels to CSV.

    Caching behavior:
    - If statistics/{role_label}_unique_entities.csv exists, skip JSON processing and
      compute plots/stats directly from that CSV.
    - Otherwise, process all JSONs and write that CSV.

    Provenance saving:
    - raw_data_0, raw_data_1, ... columns store message-level info for each
      unique (entity_text, category) pair.
    """
    stats_dir = entity_dir / CONFIG.stats_dir_name
    ensure_dir(stats_dir)

    cached_csv = stats_dir / f"{role_label}_unique_entities.csv"
    # print(f'csv path:{cached_csv}')
    # exit()
    df_cached = load_entity_csv(cached_csv)
    if df_cached is not None:
        logger.info(f"[{role_label.upper()}] Using cached entity CSV: {cached_csv}")
        return compute_stats_and_plots_from_entity_csv(
            df=df_cached,
            role_label=role_label,
            stats_dir=stats_dir,
        )

    # --- fallback: process JSONs to build cache ---
    all_entity_texts: List[str] = []
    category_counter: Counter[str] = Counter()
    unique_category_counter: Counter[str] = Counter()
    verified_type_counter: Counter[str] = Counter()

    # count occurrences per (text, category)
    entity_pair_counter: Counter[Tuple[str, str]] = Counter()
    seen_pairs: set[Tuple[str, str]] = set()

    # message provenance per (text, category), de-duplicated by message_id
    pair_to_rawdata: Dict[Tuple[str, str], List[Dict[str, Optional[str]]]] = defaultdict(list)
    pair_to_seen_msg: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    verified_count = 0
    total_verified_checked = 0

    for user_id_csv, conversation_id_csv, message_id_csv in tqdm(
        info_list, desc=f"Processing {role_label} entity JSONs"
    ):
        json_path = entity_dir / f"user-{user_id_csv}_msg-{message_id_csv}.json"

        if not json_path.exists():
            logger.warning(f"Missing JSON: {json_path.name}")
            continue

        data = safe_load_json(json_path)
        if not data:
            continue

        # prefer values inside json if present, otherwise fallback to CSV tuple
        user_id = str(data.get("user_id") or user_id_csv)
        conversation_id = str(data.get("conversation_id") or conversation_id_csv)
        message_id = str(data.get("message_id") or message_id_csv)
        message_content_val = data.get("message_content")
        message_content = str(message_content_val) if message_content_val is not None else None

        entities = data.get("entities")
        if not isinstance(entities, list):
            continue

        for ent in entities:
            if not isinstance(ent, dict):
                continue

            text_val = ent.get("entity_text")
            text = str(text_val).strip() if isinstance(text_val, str) else (str(text_val).strip() if text_val is not None else "")
            if not text:
                continue

            raw_category = ent.get("entity_type_category")
            category = remap_category(raw_category) or "UNKNOWN"

            all_entity_texts.append(text)
            category_counter[category] += 1

            key = (text, category)
            entity_pair_counter[key] += 1

            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_category_counter[category] += 1

            # store provenance once per message for this entity-pair
            if message_id not in pair_to_seen_msg[key]:
                pair_to_seen_msg[key].add(message_id)
                pair_to_rawdata[key].append(
                    raw_data_dict(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_id=message_id,
                        message_content=message_content,
                    )
                )

            if verified and isinstance(ent.get("verification"), dict):
                total_verified_checked += 1
                verification = ent["verification"]
                if verification.get("verified") is True:
                    verified_count += 1
                    correct_cls = verification.get("correct_entity_type_classification")
                    if correct_cls:
                        verified_type_counter[str(correct_cls)] += 1

    total_entities = len(all_entity_texts)
    unique_entities = len(set(all_entity_texts))

    logger.info(f"[{role_label.upper()}] Total entities: {total_entities}")
    logger.info(f"[{role_label.upper()}] Unique entities: {unique_entities}")

    accuracy: Optional[float]
    if verified and total_verified_checked > 0:
        accuracy = verified_count / total_verified_checked
        logger.info(f"[{role_label.upper()}] Verification accuracy: {accuracy:.4f}")
    else:
        accuracy = None

    # Save cache CSV: one row per unique (entity_text, category) with count + raw_data_i columns
    if entity_pair_counter:
        # compute max provenance width
        max_k = 0
        for k in entity_pair_counter.keys():
            max_k = max(max_k, len(pair_to_rawdata.get(k, [])))

        rows: List[Dict[str, Any]] = []
        for (t, c), n in entity_pair_counter.items():
            row: Dict[str, Any] = {"entity_text": t, "category": c, "count": int(n)}
            raw_list = pair_to_rawdata.get((t, c), [])
            for i, rd in enumerate(raw_list):
                # store as JSON string so CSV round-trips reliably
                row[f"raw_data_{i}"] = json.dumps(rd, ensure_ascii=False)
            # ensure all raw_data columns exist (optional but makes schema stable)
            for i in range(len(raw_list), max_k):
                row[f"raw_data_{i}"] = ""
            rows.append(row)

        df_out = pd.DataFrame(rows).sort_values(
            by=["category", "count", "entity_text"],
            ascending=[True, False, True],
        )
        df_out.to_csv(cached_csv, index=False)
        logger.info(f"[{role_label.upper()}] Saved entity CSV: {cached_csv}")

    # Plot from counters
    plot_counter_dict(
        dict(category_counter),
        f"{role_label.capitalize()} Entity Type Distribution (All)",
        f"{role_label}_entity_type_distribution_all",
        stats_dir,
    )
    plot_counter_dict(
        dict(unique_category_counter),
        f"{role_label.capitalize()} Entity Type Distribution (Unique)",
        f"{role_label}_entity_type_distribution_unique",
        stats_dir,
    )
    if verified:
        plot_counter_dict(
            dict(verified_type_counter),
            f"{role_label.capitalize()} Verified Entity Type Distribution",
            f"{role_label}_verified_entity_type_distribution",
            stats_dir,
        )

    return {
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "verification_accuracy": accuracy,
        "entity_type_distribution_all": dict(category_counter),
        "entity_type_distribution_unique": dict(unique_category_counter),
        "verified_entity_type_distribution": dict(verified_type_counter),
    }

# -----------------------------------------------------------------------------
# WildHallucinations statistics
# -----------------------------------------------------------------------------

def compute_wildhallucinations_statistics(
    split: str = "train",
    dataset_name: str = "wentingzhao/WildHallucinations",
    output_dir: Path = CONFIG.entity_data_dir,
) -> Dict[str, object]:
    """Load the WildHallucinations dataset, compute and plot category distributions."""
    stats_dir = output_dir / CONFIG.stats_dir_name
    ensure_dir(stats_dir)

    logger.info(f"Loading WildHallucinations: {dataset_name} ({split})")
    ds = load_dataset(dataset_name, split=split)

    counter = Counter(ds["category"])
    total_entities = int(sum(counter.values()))

    logger.info(f"[WILDHALLUCINATIONS] Total entities: {total_entities}")
    logger.info(f"[WILDHALLUCINATIONS] Unique categories: {len(counter)}")

    if counter:
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        x, y = zip(*items)
        fig = px.bar(
            x=x,
            y=y,
            labels={"x": "Category", "y": "Count"},
            title="WildHallucinations Category Distribution",
        )
        save_fig(fig, stats_dir / "wildhallucinations_category_distribution")

    return {
        "total_entities": total_entities,
        "unique_categories": len(counter),
        "category_distribution": dict(counter),
    }

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    user_info, assistant_info = load_message_info(CONFIG.raw_data_csv)

    user_stats = compute_entity_statistics(
        info_list=user_info,
        role_label="user",
        verified=False,
        entity_dir=CONFIG.entity_data_dir,
    )

    assistant_stats = compute_entity_statistics(
        info_list=assistant_info,
        role_label="assistant",
        verified=False,
        entity_dir=CONFIG.entity_data_dir,
    )

    wild_stats = compute_wildhallucinations_statistics()

    logger.info("User stats summary:")
    logger.info(user_stats)

    logger.info("Assistant stats summary:")
    logger.info(assistant_stats)

    logger.info("WildHallucinations stats summary:")
    logger.info(wild_stats)


if __name__ == "__main__":
    main()