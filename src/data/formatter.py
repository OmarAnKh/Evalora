from typing import Any, Dict

from src.utils.json_utils import build_assistant_json


def format_messages(sample: dict[str, Any], include_assistant: bool = True) -> dict[str, Any]:
    """Build chat-style messages for a single evaluation sample."""
    existing_messages = sample.get("messages")
    if existing_messages is not None:
        messages = list(existing_messages)
        if not include_assistant:
            messages = [message for message in messages if message.get("role") != "assistant"]
        return {
            "messages": messages,
            "score": sample.get("score"),
            "reasoning": sample.get("reasoning"),
        }

    required_fields = ["task", "reference_answer", "answer", "rubric"]
    missing = [field for field in required_fields if field not in sample]
    if missing:
        raise ValueError(
            "Cannot format sample; missing raw fields "
            f"{missing} and no preformatted 'messages' field was provided."
        )

    system_prompt = (
        "You are an automated evaluation system.\n"
        "You must always return valid JSON only."
    )  
    user_prompt = (
        "Task:\n"
        f"{sample['task']}\n\n"
        "Reference Answer:\n"
        f"{sample['reference_answer']}\n\n"
        "Student Answer:\n"
        f"{sample['answer']}\n\n"
        "Rubric:\n"
        f"{sample['rubric']}\n\n"
        "You are an evaluation system.\n"
        "Return ONLY valid JSON.\n\n"
        "Output format:\n"
        "{\n"
        '  "score": 0.0,\n'
        '  "reasoning": "short explanation"\n'
        "}\n\n"
        "Rules:\n"
        "- output ONLY JSON\n"
        "- no markdown\n"
        "- no extra text\n"
        "- score must be a numeric grade consistent with the rubric and training labels\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if include_assistant:
        assistant_prompt = build_assistant_json(
            score=sample.get("score"),
            reasoning=sample.get("reasoning"),
        )
        messages.append({"role": "assistant", "content": assistant_prompt})

    return {
        "messages": messages,
        "score": sample.get("score"),
        "reasoning": sample.get("reasoning"),
    }


def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Format a single evaluation sample into a chat-style message list."""
    return format_messages(sample, include_assistant=True)

def format_sample_no_assistant(sample: dict[str, Any]) -> dict[str, Any]:
    """Format a single evaluation sample into a chat-style message list without assistant response."""
    return format_messages(sample, include_assistant=False)

class Formatter:
    """Builds chat-style message prompts from structured evaluation samples."""

    def format(self, example: dict[str, Any]) -> dict[str, Any]:
        return format_sample(example)

    def format_no_assistant(self, example: dict[str, Any]) -> dict[str, Any]:
        return format_sample_no_assistant(example)
    
    def reformat(self, dataset) -> Any:
        """Reformat an entire dataset into chat-style message prompts."""
        return dataset.map(self.format, remove_columns=dataset.column_names)
    
