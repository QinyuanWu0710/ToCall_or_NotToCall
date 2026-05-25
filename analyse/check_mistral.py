import csv
import json
import argparse
from pathlib import Path

TASK_CONFIGS = {
    "entity": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/"),
        cost_setups=[10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0],
        # no budget-aware dirs for entity
        csv_suffix="",
    ),
    "bfcl": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/"),
        cost_setups=[],   # no budget-aware dirs for bfcl
        csv_suffix="",
    ),
    "invivo": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/"),
        cost_setups=[10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0],
        csv_suffix="",
    ),
    "perplexity": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/perplexity"),
        cost_setups=[10000, 1000, 500, 250, 222, 200, 100, 67, 50, 40, 33, 29, 25, 20, 10, 0],
        csv_suffix="",
    ),
}


def extract_needs_tool(raw_response):
    """
    Return bool(needs_tool) parsed from the raw_response string.
    Handles Mistral's {{ }} double-brace escaping.
    """
    if raw_response is None:
        return False

    if isinstance(raw_response, dict):
        return bool(raw_response.get("needs_tool", False))

    if not isinstance(raw_response, str):
        return False

    raw_response = raw_response.strip()
    if not raw_response:
        return False

    # normalize double braces: {{ ... }} -> { ... }
    if raw_response.startswith("{{") and raw_response.endswith("}}"):
        raw_response = raw_response[1:-1].strip()

    # try full-string JSON parse first
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return bool(parsed.get("needs_tool", False))
        return False
    except (json.JSONDecodeError, TypeError):
        pass

    # fallback: try to find the first JSON object substring
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return False

    candidate = raw_response[start:end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return bool(parsed.get("needs_tool", False))
    except (json.JSONDecodeError, TypeError):
        pass

    return False


def patch_csv(jsonl_path: Path, csv_path: Path):
    if not jsonl_path.exists():
        print(f"  SKIP (no JSONL): {jsonl_path.name}")
        return
    if not csv_path.exists():
        print(f"  SKIP (no CSV):   {csv_path.name}")
        return

    # Parse yes_no_decision from JSONL raw_response
    search_called_values = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                search_called_values.append(False)
                continue
            try:
                tool_calls = item.get("tool_calls") or []
                if not tool_calls:
                    search_called_values.append(False)
                    continue
                raw_response = tool_calls[0].get("raw_response")
                search_called_values.append(extract_needs_tool(raw_response))
            except Exception:
                search_called_values.append(False)

    # Read CSV and patch search_called column
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "search_called" not in fieldnames:
        fieldnames.append("search_called")

    for i, row in enumerate(rows):
        row["search_called"] = str(search_called_values[i]) if i < len(search_called_values) else "False"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_true = sum(search_called_values)
    print(f"  OK  search_called=True: {n_true}/{len(rows)}  — {csv_path.parent.name}/{csv_path.name}")


def get_mistral_stems(directory: Path, csv_suffix: str) -> list[str]:
    """Return base stems (without _summary{csv_suffix}.csv) for Mistral with_search CSVs."""
    pattern = f"vllm_mistralai_*_with_search_summary{csv_suffix}.csv"
    suffix_to_strip = f"_summary{csv_suffix}"
    return [
        p.stem.replace(suffix_to_strip, "")
        for p in directory.glob(pattern)
    ]


def run_task(task: str):
    cfg = TASK_CONFIGS[task]
    base: Path = cfg["base"]
    cost_setups: list = cfg["cost_setups"]
    csv_suffix: str = cfg["csv_suffix"]

    print(f"\n{'='*60}")
    print(f"  Task: {task}  (base: {base},  csv_suffix='{csv_suffix}')")
    print(f"{'='*60}")

    # main/ directory
    print("\n=== main ===")
    main_dir = base / "main"
    if not main_dir.exists():
        print(f"  [NOT FOUND, skipping] {main_dir}")
    else:
        for stem in get_mistral_stems(main_dir, csv_suffix):
            patch_csv(
                jsonl_path=main_dir / f"{stem}.jsonl",
                csv_path=main_dir / f"{stem}_summary{csv_suffix}.csv",
            )

    # budget-aware directories 
    for cost in cost_setups:
        dir_name = f"tool-cost-{cost}-budget-aware"
        cost_dir = base / dir_name
        if not cost_dir.exists():
            print(f"\n=== {dir_name}  [NOT FOUND, skipping] ===")
            continue
        print(f"\n=== {dir_name} ===")
        for stem in get_mistral_stems(cost_dir, csv_suffix):
            patch_csv(
                jsonl_path=cost_dir / f"{stem}.jsonl",
                csv_path=cost_dir / f"{stem}_summary{csv_suffix}.csv",
            )
    for cost in cost_setups:
        dir_name = f"tool-cost-{cost}-budget-aware-v2"
        cost_dir = base / dir_name
        if not cost_dir.exists():
            print(f"\n=== {cost_dir}  [NOT FOUND, skipping] ===")
            continue
        print(f"\n=== {dir_name} ===")
        for stem in get_mistral_stems(cost_dir, csv_suffix):
            patch_csv(
                jsonl_path=cost_dir / f"{stem}.jsonl",
                csv_path=cost_dir / f"{stem}_summary{csv_suffix}.csv",
            )

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check/patch Mistral search_called parsing across tasks"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=list(TASK_CONFIGS) + ["all"],
        help="Which task to check (entity / bfcl / invivo / all)",
    )
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS) if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task)

    print("\nDone.")
