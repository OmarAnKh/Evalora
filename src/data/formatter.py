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
        "You are an automated evaluation model.\n\n"
        "Your task is to evaluate a candidate response based on the provided "
        "task, reference response, and rubric.\n\n"
        "Evaluate the candidate response according to the rubric and assign an "
        "appropriate score. The reference response should be used as a guide for "
        "correctness and completeness. Equivalent wording or valid alternative "
        "formulations should be accepted when they satisfy the rubric.\n\n"
        "Your evaluation must:\n\n"
        "* Follow the provided rubric.\n"
        "* Consider correctness, completeness, and relevance as specified by the rubric.\n"
        "* Distinguish between fully correct, partially correct, and incorrect responses.\n"
        "* Avoid making assumptions beyond the provided information.\n"
        "* Provide a concise justification for the assigned score.\n"
        "* Return only valid JSON.\n\n"
        "The output must follow this format:\n\n"
        "{\"score\": <score>,\n"
        "\"reasoning\": \"<brief justification>\"}"
    )
    user_prompt = (
        "Task: "
        f"{sample['task']}\n"
        "Reference response: "
        f"{sample['reference_answer']}\n"
        "Candidate response: "
        f"{sample['answer']}\n"
        "Rubric: "
        f"{sample['rubric']}"
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
    
