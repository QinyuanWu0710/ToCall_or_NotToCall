"""
Re-parse yes_no_decision from raw_response for main-perceived-need-v1 and v2.

Parses item["tool_calls"][0]["raw_response"] and determines whether the model
perceives a need for search (yes_no_decision=True) or not (False).

v1 prompt: "Do you need help to answer the question?"
    yes / needs help / uncertainty  → True  (perceives need)
    no  / confident / just answers  → False (no need)

v2 prompt: "Do you know the answer of the tool?"
    yes / confident / just answers  → False (knows = no need)
    no  / uncertainty               → True  (doesn't know = needs help)

Preprocessing handles:
  - <think>...</think> blocks (Qwen base thinking model)
  - "analysis...assistantfinal..." pattern (GPT)
  - Capability phrases: "I can help/answer/provide" → confident
  - Uncertainty phrases: "I don't know/have/..." → not confident

Special case v1: "Yes, I can help answer that" (Qwen IT quirk) is treated as
confident (False) because the model is saying it CAN answer, not that it NEEDS help.
"""

import csv
import json
import argparse
import re
from pathlib import Path

# ── Task configs ──────────────────────────────────────────────────────────────
# versions: list of (dir_name, invert)
#   invert=False  → yes/true → True   (v1: "Do you need help?" yes=needs help)
#   invert=True   → yes/true → False  (v2: "Do you know the answer?" yes=knows=no need)
TASK_CONFIGS = {
    "entity": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/entity_hallucination/temp=0/"),
        versions=[("main-perceived-need-v1", False), ("main-perceived-need-v2", True)],
        csv_suffix="",
    ),
    "bfcl": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/bfcl_raw/tool_result/"),
        versions=[("main-perceived-need", False)],
        csv_suffix="",
    ),
    "invivo": dict(
        base=Path("/NS/chatgpt/work/qwu/hallucinations_detection/results/real_query/temp=0/"),
        versions=[("main-perceived-need-v1", False), ("main-perceived-need-v2", True)],
        csv_suffix="",
    ),
}


# ── Parsing ───────────────────────────────────────────────────────────────────

# Phrases indicating the model is UNCERTAIN / doesn't know the answer
_UNCERTAINTY = re.compile(
    r"i\s+(don'?t|do\s+not|have\s+no|haven'?t|can'?t|cannot|couldn'?t)\s+"
    r"(know|have|find|recall|remember|access|provide|answer|tell|confirm|verify)"
    r"|i'?m\s+(not\s+sure|uncertain|unable|unaware|unsure|unfamiliar)"
    r"|i\s+lack\s+(access|information|knowledge|data)"
    r"|no\s+(specific|particular|detailed|direct|available)\s+(information|knowledge|details|data)"
    r"|unfortunately[,\s]+i\s+(don'?t|do\s+not|can'?t|cannot|have\s+no)"
    r"|i\s+have\s+no\s+(information|knowledge|details|data|specific)"
    r"|not\s+(widely\s+)?known\b"
    r"|i\s+don'?t\s+have\s+(any\s+)?(specific|particular|detailed)?\s*(information|knowledge|data)",
    re.IGNORECASE,
)

# Phrases indicating the model IS CONFIDENT / can answer
_CAPABLE = re.compile(
    r"i\s+can\s+(help|answer|provide|tell|explain|describe|give|assist)"
    r"|here\s+(is|are|'?s)\s+(a\s+|what|my|the\s+)?"
    r"|let\s+me\s+(help|explain|answer|tell|describe|provide|give)"
    r"|i\s+(know|recall|remember)\s+(about|that|this|what|the)"
    r"|certainly[,!\s]|sure[,!\s]|of\s+course[,!\s]",
    re.IGNORECASE,
)

# Capability phrase right after yes (e.g. "Yes, I can help answer that")
_YES_BUT_CAPABLE = re.compile(
    r"^yes\b.{0,30}i\s+can\s+(help|answer|provide|tell|explain)",
    re.IGNORECASE | re.DOTALL,
)


def _preprocess(text: str) -> str:
    """Strip think tags and analysis prefixes to get the actionable response text."""
    text = text.strip()

    # Strip closed <think>...</think> blocks (Qwen thinking model)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # GPT pattern: "analysis...assistantfinal[actual answer]"
    if text.lower().startswith("analysis"):
        m = re.search(r"assistantfinal", text, re.IGNORECASE)
        if m:
            text = text[m.end():].strip()
        # else: keep full analysis text (yes/no signals may be inside)

    return text


def perceived_need(raw_response, invert: bool = False) -> bool:
    """
    Return True if the model perceives a need for search/help, False otherwise.

    invert=False  →  v1 "Do you need help?"  yes=needs_help=True
    invert=True   →  v2 "Do you know the answer?"  yes=knows=no_need → False
    """
    if raw_response is None:
        return False

    # Handle dict (already-parsed JSON)
    if isinstance(raw_response, dict):
        for key in ("yes_no_decision", "answer", "needs_tool"):
            val = raw_response.get(key)
            if val is not None:
                raw_positive = bool(val) if isinstance(val, bool) else \
                    str(val).strip().lower() in ("yes", "true", "1")
                return (not raw_positive) if invert else raw_positive
        return False

    if not isinstance(raw_response, str):
        return False

    text = raw_response.strip()
    if not text:
        return False

    # Try JSON parse (handles Mistral {{ }} escaping)
    candidate = text[1:-1].strip() if (text.startswith("{{") and text.endswith("}}")) else text
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return perceived_need(parsed, invert=invert)
    except (json.JSONDecodeError, ValueError):
        pass

    text = _preprocess(text)
    low = text.lower()

    # ── Explicit yes/no at start ──────────────────────────────────────────────
    starts_yes = re.match(r"^(yes\b|true\b)", low)
    starts_no  = re.match(r"^(no\b|false\b)", low)

    if starts_yes:
        # Special case: "Yes, I can help answer that" (Qwen IT v1 quirk)
        # Model means "yes I CAN answer" not "yes I NEED help" → treat as confident
        if _YES_BUT_CAPABLE.match(text):
            confident = True
        else:
            confident = False   # genuine "yes I need help" or "yes I know"
        # confident=False  → raw_positive=True  (yes to the question)
        # confident=True   → raw_positive=False (quirk: yes-but-capable = no-need)
        raw_positive = not confident
        return (not raw_positive) if invert else raw_positive

    if starts_no:
        raw_positive = False   # explicit "no"
        return (not raw_positive) if invert else raw_positive

    # ── Contextual signals in first 400 chars ─────────────────────────────────
    window = text[:400]
    is_uncertain = bool(_UNCERTAINTY.search(window))
    is_capable   = bool(_CAPABLE.search(window))

    if is_uncertain and not is_capable:
        # Doesn't know → needs help → raw_positive=False (no to confidence)
        raw_positive = False
        return (not raw_positive) if invert else raw_positive

    if is_capable and not is_uncertain:
        # Confident, can answer → raw_positive=True
        raw_positive = True
        return (not raw_positive) if invert else raw_positive

    # ── Default: model just answers without hedging → confident ───────────────
    # v1: confident = doesn't need help = False
    # v2: confident = knows = invert(True) = False  ← same result either way
    raw_positive = True
    return (not raw_positive) if invert else raw_positive


# ── CSV patching ──────────────────────────────────────────────────────────────

def patch_csv(jsonl_path: Path, csv_path: Path, invert: bool = False):
    if not jsonl_path.exists():
        print(f"  SKIP (no JSONL): {jsonl_path.name}")
        return
    if not csv_path.exists():
        print(f"  SKIP (no CSV):   {csv_path.name}")
        return

    # Parse yes_no_decision from JSONL raw_response
    decisions = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                decisions.append(False)
                continue
            try:
                tool_calls = item.get("tool_calls") or []
                if not tool_calls:
                    decisions.append(False)
                    continue
                raw = tool_calls[0].get("raw_response")
                decisions.append(perceived_need(raw, invert=invert))
            except Exception:
                decisions.append(False)

    # Read CSV and patch yes_no_decision column
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "yes_no_decision" not in fieldnames:
        fieldnames.append("yes_no_decision")

    if len(rows) != len(decisions):
        print(f"  WARN row count mismatch: CSV={len(rows)}, JSONL={len(decisions)} — {csv_path.name}")

    for i, row in enumerate(rows):
        row["yes_no_decision"] = str(decisions[i]) if i < len(decisions) else "False"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_true = sum(decisions)
    print(f"  OK  yes_no_decision=True: {n_true:4d}/{len(rows)}  — {csv_path.parent.name}/{csv_path.name}")


# ── Per-directory runner ──────────────────────────────────────────────────────

def run_directory(directory: Path, csv_suffix: str, invert: bool = False):
    """Patch all model CSVs in a perceived-need directory."""
    if not directory.exists():
        print(f"  [NOT FOUND, skipping] {directory}")
        return

    # Find all with_search JSONL files — one per model
    jsonl_files = sorted(directory.glob("vllm_*_with_search.jsonl"))
    if not jsonl_files:
        print(f"  (no JSONL files found)")
        return

    for jsonl_path in jsonl_files:
        # Derive CSV name: replace .jsonl → _summary{csv_suffix}.csv
        stem = jsonl_path.stem  # e.g. vllm_google_gemma-3-27b-it_with_search
        csv_name = f"{stem}_summary{csv_suffix}.csv"
        csv_path = directory / csv_name
        patch_csv(jsonl_path, csv_path, invert=invert)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task: str):
    cfg = TASK_CONFIGS[task]
    base: Path = cfg["base"]
    versions: list = cfg["versions"]
    csv_suffix: str = cfg["csv_suffix"]

    print(f"\n{'='*60}")
    print(f"  Task: {task}  (base: {base})")
    print(f"{'='*60}")

    for version, invert in versions:
        directory = base / version
        label = f"{version}  [invert={'yes→False' if invert else 'yes→True'}]"
        print(f"\n=== {label} ===")
        run_directory(directory, csv_suffix, invert=invert)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-parse yes_no_decision from raw_response in perceived-need JSONL files"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=list(TASK_CONFIGS) + ["all"],
        help="Which task to process (entity / bfcl / invivo / all)",
    )
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS) if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task)

    print("\nDone.")
