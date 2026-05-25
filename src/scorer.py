"""
Task-specific scoring functions.

Supported tasks:
  - entity   : Factuality scoring via GPT claim extraction + web verification
                (wraps gpt_factuality_nobatch logic)
  - bfcl     : Exact-match scoring against BFCL ground-truth dataset
  - Add more tasks by subclassing BaseScorer.

Usage:
    scorer = get_scorer("entity", model="gpt-4o")
    score  = scorer.score(entity="...", question="...", response="...")
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class BaseScorer(ABC):
    """Abstract base class for all task scorers."""

    @abstractmethod
    def score(self, **kwargs) -> float:
        """
        Compute a score in [0, 1] for a single sample.

        Subclasses define which kwargs they consume.
        Returns float in [0, 1].
        """

    def score_batch(self, rows: List[Dict[str, Any]]) -> List[float]:
        """Score a list of result rows.  Override for batched efficiency."""
        return [self.score(**row) for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Entity / Factuality scorer  (uses OpenAI responses API + web_search)
# ──────────────────────────────────────────────────────────────────────────────

# Inline schemas (same as gpt_factuality_nobatch.py so we have no import dep)
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entity": {"type": "string"},
        "question": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["factual", "numerical", "historical", "definition", "other"],
                    },
                    "span": {"type": "string"},
                },
                "required": ["id", "claim", "type", "span"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entity", "question", "claims"],
    "additionalProperties": False,
}

_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "entity": {"type": "string"},
        "question": {"type": "string"},
        "verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "is_correct": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "claim", "is_correct", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entity", "question", "verifications"],
    "additionalProperties": False,
}

_EXTRACTION_SYSTEM = (
    "You are a claim extraction engine.\n"
    "Extract all distinct, checkable claims from the RESPONSE.\n"
    "- A claim is an assertion that could be true or false.\n"
    "- Split compound sentences into atomic claims.\n"
    "- Do NOT add new claims.\n"
    "- Prefer recall over precision.\n"
    "Return ONLY valid JSON matching the provided schema."
)

_VERIFICATION_SYSTEM = (
    "You are a claim verification engine.\n"
    "Verify each claim using the web_search tool when needed, prioritizing reliable sources.\n"
    "- Preserve the original id and claim text exactly.\n"
    "- Set is_correct=true only when the claim is clearly correct.\n"
    "- If uncertain, set is_correct=false and explain briefly in reason.\n"
    "- In reason, include 1–3 plain-text source URLs you used.\n"
    "Return ONLY valid JSON matching the provided schema."
)


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if isinstance(t, str):
                chunks.append(t)
    return "\n".join(chunks).strip()


def _call_with_retry(client, body: Dict, max_retries: int = 5, base_sleep: float = 1.0) -> Any:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.responses.create(**body)
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** attempt))
    raise RuntimeError(f"Failed after {max_retries} retries: {last}") from last


class EntityFactualityScorer(BaseScorer):
    """
    Scores entity responses by extracting claims and verifying them via web search.

    Returns correctness_score = correct_claims / total_claims.
    """

    def __init__(
        self,
        extraction_model: str = "gpt-4o",
        verification_model: str = "gpt-4o",
        sleep: float = 0.0,
    ):
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except ImportError:
            raise ImportError("openai package is required for EntityFactualityScorer")

        self.extraction_model = extraction_model
        self.verification_model = verification_model
        self.sleep = sleep

    def _extract_claims(self, question: str, entity: str, response: str) -> Dict:
        body = {
            "model": self.extraction_model,
            "input": [
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"question: {question}\n"
                        f"entity: {entity}\n\n"
                        f"RESPONSE:\n<<<\n{response}\n>>>"
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "claim_list",
                    "strict": True,
                    "schema": _EXTRACTION_SCHEMA,
                }
            },
        }
        resp = _call_with_retry(self._client, body)
        return json.loads(_extract_text(resp))

    def _verify_claims(
        self, question: str, entity: str, response: str, claims: List[Dict]
    ) -> Dict:
        body = {
            "model": self.verification_model,
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "input": [
                {"role": "system", "content": _VERIFICATION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"question: {question}\n"
                        f"entity: {entity}\n\n"
                        f"ORIGINAL_RESPONSE:\n<<<\n{response}\n>>>\n\n"
                        f"CLAIMS_TO_VERIFY (JSON):\n{json.dumps(claims, ensure_ascii=False)}"
                    ),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "claim_verification",
                    "strict": True,
                    "schema": _VERIFICATION_SCHEMA,
                }
            },
        }
        resp = _call_with_retry(self._client, body)
        return json.loads(_extract_text(resp))

    def score(
        self,
        entity: str = "",
        question: str = "",
        response: str = "",
        **_kwargs,
    ) -> float:
        if not response.strip():
            return 0.0

        claims_payload = self._extract_claims(question, entity, response)
        claims_list = claims_payload.get("claims", [])

        if not claims_list:
            return 0.0

        verify_payload = self._verify_claims(question, entity, response, claims_list)
        verifications = verify_payload.get("verifications", [])

        total = len(verifications)
        correct = sum(1 for v in verifications if v.get("is_correct") is True)

        if self.sleep > 0:
            time.sleep(self.sleep)

        return round(correct / total, 6) if total else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BFCL scorer  (exact match against ground-truth function calls)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_bfcl(value: Any) -> Any:
    """
    Recursively normalise a parsed function-call value for comparison:
    - Sort dict keys
    - Sort lists of primitives / single-key dicts (where order is irrelevant)
    - Strip surrounding whitespace from strings
    """
    if isinstance(value, dict):
        return {k: _normalize_bfcl(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_normalize_bfcl(v) for v in value]
        # Sort only if all elements are comparable primitives or single-key dicts
        try:
            return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError:
            return normalized
    if isinstance(value, str):
        return value.strip()
    return value


def _parse_function_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a model response as a function-call dict.
    Accepts:
      - Plain JSON object  {"name": ..., "arguments": {...}}
      - JSON wrapped in ```json ... ```
      - Python-style  func_name(arg=val, ...)  (best-effort)
    Returns None if parsing fails.
    """
    text = text.strip()

    # 1. Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 2. Try JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass

    # 3. Python-style call:  func_name(key=val, key2="val2")
    m = re.match(r"^(\w+)\s*\((.*)?\)\s*$", text, re.DOTALL)
    if m:
        func_name = m.group(1)
        args_str = m.group(2).strip()
        try:
            import ast
            # Wrap in a dummy function call so ast can parse it
            parsed = ast.parse(f"_f({args_str})", mode="eval")
            kwargs = {}
            call = parsed.body
            for kw in call.keywords:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            return {"name": func_name, "arguments": kwargs}
        except Exception:
            pass

    return None


class BFCLScorer(BaseScorer):
    """
    Scores BFCL (Berkeley Function-Calling Leaderboard) responses by exact match
    against ground-truth function calls.

    Ground truth can be supplied:
      - Per-call via the `ground_truth` kwarg in score()
      - As a pre-loaded dict keyed by sample id via load_ground_truth()

    Score is 1.0 (exact match) or 0.0 (mismatch / parse error).
    """

    def __init__(self, ground_truth_file: Optional[str] = None):
        """
        Args:
            ground_truth_file: Optional path to a JSONL file where each line has
                               {"id": ..., "ground_truth": <function-call dict or list>}
        """
        self._gt_map: Dict[str, Any] = {}
        if ground_truth_file:
            self.load_ground_truth(ground_truth_file)

    def load_ground_truth(self, path: str) -> None:
        """Load a JSONL ground-truth file into an id→gt mapping."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"BFCL ground truth file not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = str(obj.get("id", ""))
                if key:
                    self._gt_map[key] = obj.get("ground_truth")

    def score(
        self,
        response: str = "",
        ground_truth: Any = None,
        sample_id: str = "",
        **_kwargs,
    ) -> float:
        """
        Args:
            response:     Raw model response text.
            ground_truth: Expected function-call dict/list.  If None, looked up
                          by sample_id from the pre-loaded map.
            sample_id:    Key for ground-truth lookup when ground_truth is None.
        """
        # Resolve ground truth
        gt = ground_truth
        if gt is None and sample_id:
            gt = self._gt_map.get(str(sample_id))
        if gt is None:
            # No ground truth available → cannot score
            return 0.0

        # Parse model response
        pred = _parse_function_call(response)
        if pred is None:
            return 0.0

        # Normalise both sides and compare
        try:
            return 1.0 if _normalize_bfcl(pred) == _normalize_bfcl(gt) else 0.0
        except Exception:
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_SCORER_REGISTRY: Dict[str, type] = {
    "entity": EntityFactualityScorer,
    "bfcl": BFCLScorer,
}


def get_scorer(task: str, **kwargs) -> BaseScorer:
    """
    Return an instantiated scorer for the given task name.

    Args:
        task:    Task identifier, e.g. "entity" or "bfcl".
        **kwargs: Passed to the scorer constructor.

    Raises:
        ValueError: If the task is not recognised.
    """
    task = task.lower().strip()
    if task not in _SCORER_REGISTRY:
        available = ", ".join(_SCORER_REGISTRY)
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available}"
        )
    return _SCORER_REGISTRY[task](**kwargs)