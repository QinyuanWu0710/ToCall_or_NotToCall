"""
Re-score an existing results JSON file using EntityFactualityScorer.
Writes a CSV with columns matching the standard summary format.

Usage:
    python run_scoring.py \
        --input  /path/to/results.json \
        --output /path/to/scores.csv \
        [--extraction-model gpt-4o] \
        [--verification-model gpt-4o] \
        [--sleep 0.0]

Input JSON: list of objects with at least:
    entity, query (or question), response
    Optional passthrough fields: model, search_called, yes_no_decision

Output CSV columns:
    entity, query, model, search_called, yes_no_decision,
    score, correct_claims, total_claims
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scorer import EntityFactualityScorer

CSV_COLUMNS = [
    "entity",
    "query",
    "model",
    "search_called",
    "yes_no_decision",
    "score",
    "correct_claims",
    "total_claims",
]


def score_row(scorer: EntityFactualityScorer, row: dict) -> dict:
    """Run scoring on one row and return a flat dict ready for CSV."""
    entity   = row.get("entity", "")
    question = row.get("query") or row.get("question", "")
    response = row.get("response", "")

    correct_claims = None
    total_claims   = None
    score          = None

    if not response.strip():
        score = 0.0
    else:
        # Reach into scorer internals to also capture claim counts
        try:
            claims_payload = scorer._extract_claims(question, entity, response)
            claims_list    = claims_payload.get("claims", [])

            if not claims_list:
                score = 0.0
                total_claims = 0
                correct_claims = 0
            else:
                verify_payload = scorer._verify_claims(question, entity, response, claims_list)
                verifications  = verify_payload.get("verifications", [])

                total_claims   = len(verifications)
                correct_claims = sum(1 for v in verifications if v.get("is_correct") is True)
                score = round(correct_claims / total_claims, 6) if total_claims else 0.0
        except Exception as e:
            print(f"  [WARN] Scoring failed: {e}", file=sys.stderr)

    return {
        "entity":          entity,
        "query":           question,
        "model":           row.get("model", ""),
        "search_called":   row.get("search_called", ""),
        "yes_no_decision": row.get("yes_no_decision", ""),
        "score":           score,
        "correct_claims":  correct_claims,
        "total_claims":    total_claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a results JSON → summary CSV")
    parser.add_argument("--input",  required=True, help="Path to input results JSON file")
    parser.add_argument("--output", required=True, help="Path to write output CSV file")
    parser.add_argument("--extraction-model",   default="gpt-4o")
    parser.add_argument("--verification-model", default="gpt-4o")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between samples (default: 0)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Support both JSON array and JSONL (one object per line)
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    scorer = EntityFactualityScorer(
        extraction_model=args.extraction_model,
        verification_model=args.verification_model,
        sleep=args.sleep,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, row in enumerate(data):
            entity = row.get("entity", "")
            print(f"[{i+1}/{len(data)}] Scoring entity={entity!r} ...", flush=True)

            result = score_row(scorer, row)
            writer.writerow(result)

    print(f"\nDone. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
