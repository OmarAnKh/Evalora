import json
from typing import Any, Dict


def build_assistant_json(score: Any, reasoning: Any) -> str:
    """Build a deterministic JSON string for assistant outputs."""
    payload: Dict[str, Any] = {
        "score": score,
        "reasoning": reasoning,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ": "))
