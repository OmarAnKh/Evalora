from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', flags=re.IGNORECASE)


def _extract_json_object(text: str) -> str | None:
    match = _JSON_OBJECT_RE.search(text.strip())
    return match.group(0) if match else None


def _coerce_score(value: Any, min_score: float | None, max_score: float | None) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if min_score is not None:
        score = max(min_score, score)
    if max_score is not None:
        score = min(max_score, score)
    return score


def parse_prediction(
    text: str,
    min_score: float | None = 0.0,
    max_score: float | None = None,
) -> dict[str, Any]:
    """Parse model output into score/reasoning, tolerating common malformed generations."""
    raw = text.strip()
    parsed: dict[str, Any] | None = None
    json_blob = _extract_json_object(raw)

    if json_blob:
        try:
            payload = json.loads(json_blob)
            if isinstance(payload, dict):
                parsed = payload
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        score_match = _SCORE_RE.search(raw)
        score = score_match.group(1) if score_match else None
        parsed = {"score": score, "reasoning": raw}

    reasoning = parsed.get("reasoning", parsed.get("rationale", ""))
    score = _coerce_score(parsed.get("score"), min_score, max_score)

    return {
        "score": score if score is not None else min_score,
        "reasoning": str(reasoning).strip(),
        "raw_output": raw,
        "parse_error": score is None,
    }
