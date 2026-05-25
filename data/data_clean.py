from __future__ import annotations

import argparse
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CSV = Path("/NS/chatgpt/work/qwu/Tool_Call_Code/data/InvivoQuery.csv")
DEFAULT_MODEL = "openai/privacy-filter"
MASK_TEMPLATE = "[PII:{label}]"

REGEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "mac_address",
        re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    ),
    (
        "ipv4_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
    ),
    (
        "ipv6_address",
        re.compile(
            r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b"
        ),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "url",
        re.compile(r"\b(?:https?://|www\.)[^\s,\"'<>]+", re.IGNORECASE),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?"
            r"(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask PII in InvivoQuery.csv using OpenAI Privacy Filter plus regex rules."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV. Defaults to overwriting --input after making a .bak copy.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        default=-1,
        help="Transformers pipeline device. Use -1 for CPU, 0 for first GPU.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Columns to sanitize. Defaults to every string/object column.",
    )
    parser.add_argument(
        "--regex-only",
        action="store_true",
        help="Skip the model and apply only deterministic regex masking.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when overwriting the input.",
    )
    return parser.parse_args()


def mask_for(label: str) -> str:
    return MASK_TEMPLATE.format(label=label)


def regex_mask(text: str) -> str:
    masked = text
    for label, pattern in REGEX_PATTERNS:
        masked = pattern.sub(mask_for(label), masked)
    return masked


def load_privacy_filter(model_name: str, device: str | int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install the OpenAI Privacy Filter runtime "
            "dependencies first, for example:\n"
            "  pip install torch transformers\n"
            "Then rerun this script."
        ) from exc

    try:
        device_value = int(device)
    except (TypeError, ValueError):
        device_value = device

    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=device_value,
    )


def entity_label(entity: dict) -> str:
    raw = entity.get("entity_group") or entity.get("entity") or "pii"
    return str(raw).removeprefix("B-").removeprefix("I-").removeprefix("E-").removeprefix("S-")


def spans_from_entities(entities: Iterable[dict], text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    search_from = 0

    for entity in entities:
        label = entity_label(entity)
        start = entity.get("start")
        end = entity.get("end")

        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            spans.append((start, end, label))
            continue

        word = str(entity.get("word") or "").strip()
        if not word:
            continue

        found_at = text.find(word, search_from)
        if found_at == -1:
            found_at = text.find(word)
        if found_at != -1:
            spans.append((found_at, found_at + len(word), label))
            search_from = found_at + len(word)

    return spans


def apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    if not spans:
        return text

    masked = text
    for start, end, label in sorted(spans, key=lambda span: span[0], reverse=True):
        masked = masked[:start] + mask_for(label) + masked[end:]
    return masked


def build_sanitizer(classifier):
    @lru_cache(maxsize=20_000)
    def sanitize_value(value: str) -> str:
        if not value:
            return value

        model_masked = value
        if classifier is not None:
            entities = classifier(value)
            model_masked = apply_spans(value, spans_from_entities(entities, value))

        return regex_mask(model_masked)

    return sanitize_value


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path

    df = pd.read_csv(input_path)

    columns = args.columns
    if columns is None:
        columns = [
            column
            for column in df.columns
            if pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ]

    missing_columns = sorted(set(columns) - set(df.columns))
    if missing_columns:
        raise SystemExit(f"Missing columns in CSV: {missing_columns}")

    classifier = None if args.regex_only else load_privacy_filter(args.model, args.device)
    sanitize_value = build_sanitizer(classifier)

    changed_cells = 0
    for column in columns:
        original = df[column].copy()
        df[column] = df[column].map(
            lambda value: sanitize_value(str(value)) if pd.notna(value) else value
        )
        changed_cells += int((original != df[column]).sum())

    if output_path == input_path and not args.no_backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        print(f"Backup written to {backup_path}")

    df.to_csv(output_path, index=False)
    print(f"Sanitized {changed_cells} cells across {len(columns)} columns.")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
