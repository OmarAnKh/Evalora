from typing import Any, Dict

from src.utils.json_utils import build_assistant_json


def format_messages(sample: dict[str, Any], include_assistant: bool = True) -> dict[str, Any]:
    """Build chat-style messages for a single evaluation sample."""
    system_prompt = "You are an automated evaluation system."
    user_prompt = (
        "Task:\n"
        f"{sample['task']}\n\n"
        "Reference Answer:\n"
        f"{sample['reference_answer']}\n\n"
        "Student Answer:\n"
        f"{sample['answer']}\n\n"
        "Rubric:\n"
        f"{sample['rubric']}\n\n"
        "Evaluate the student answer and return:\n"
        "- score\n"
        "- reasoning\n"
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


class Formatter:
    """Builds chat-style message prompts from structured evaluation samples."""

    def format(self, example: dict[str, Any]) -> dict[str, Any]:
        return format_sample(example)

    def reformat(self, dataset) -> Any:
        """Reformat an entire dataset into chat-style message prompts."""
        return dataset.map(self.format, remove_columns=dataset.column_names)
