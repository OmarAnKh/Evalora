import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas.dataset import EvaluationSample


class Formatter:
    """Encodes evaluation samples using tokenizer chat templates."""

    def build_messages(self, record: EvaluationSample) -> list[dict[str, str]]:
        """Builds chat messages for a single EvaluationSample record.
        Args:
            record (EvaluationSample): The evaluation sample record to format.
        Returns:
            list[dict[str, str]]: A list of role/content chat messages.
        """
        rubric_text = self._format_rubric(record.rubric)
        user_content = (
            "Task:\n"
            f"{record.task}\n\n"
            "Reference Answer:\n"
            f"{record.reference_answer}\n\n"
            "Student Answer:\n"
            f"{record.answer}\n\n"
            "Rubric:\n"
            f"{rubric_text}\n\n"
            "Evaluate the student answer and return:\n"
            "- score\n"
            "- reasoning"
        )
        assistant_content = json.dumps(
            {"score": record.score, "reasoning": record.reasoning},
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": "You are an automated evaluation system."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    def _format_rubric(
        self,
        rubric,
    ) -> str:
        """Formats the rubric criteria into a readable string format.
        Args:
            rubric (list[Criterion]): A list of Criterion objects representing the evaluation rubric.
        Returns:
            str: A formatted string representation of the rubric criteria.
        """
        return "\n".join(
            (f"- {item.criterion} " f"(weight: {item.weight}): " f"{item.description}")
            for item in rubric
        )

    def load_records_from_file(self, file_id: str) -> list[EvaluationSample]:
        """Loads and validates evaluation sample records from a specified file.
        Args:
            file_id (str): The identifier of the file containing evaluation sample records to load.
        Returns:
            list[EvaluationSample]: A list of validated evaluation sample records.
        """

        sub_path = "processed" if file_id.startswith("processed_") else "raw"
        normalized_file_id = (
            file_id if file_id.endswith(".jsonl") else f"{file_id}.jsonl"
        )

        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / sub_path / normalized_file_id

        if not path.exists():
            fallback_path = repo_root / "data" / sub_path / file_id
            if fallback_path.exists():
                path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Input file not found: {file_id} (also tried: {normalized_file_id})"
                )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            records: list[EvaluationSample] = []

            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} in {file_id}: {exc}"
                    ) from exc

                try:
                    records.append(EvaluationSample.model_validate(payload))
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid record at line {line_number} in {file_id}: {exc}"
                    ) from exc

            return records

    def encode(
        self,
        example: dict[str, Any],
        tokenizer: Any,
        inference: bool = False,
        device: str | None = None,
    ) -> Any:
        """Encodes a prompt using apply_chat_template.

        Args:
            example (dict[str, Any]): Dict with a "prompt" list of chat messages.
            tokenizer (Any): Hugging Face tokenizer with apply_chat_template.
            inference (bool): Whether to prepare tensors with generation prompt.
            device (str | None): Device to move tensors to when inference is True.

        Returns:
            Any: Tokenized output (dict for inference, tokenizer output otherwise).
        """

        if device is None:
            device = "cpu"

        if inference:
            encoded = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)

            return encoded

        encoded = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=False,
        )

        return tokenizer(encoded, truncation=True)

    def apply_chat_template(self, tokenizer: Any, record: EvaluationSample) -> dict[str, list[int]]:
        """Formats and tokenizes a record via tokenizer.apply_chat_template.
        Args:
            tokenizer (Any): A Hugging Face tokenizer with apply_chat_template.
            record (EvaluationSample): The evaluation sample record to format.
        Returns:
            dict[str, list[int]]: Tokenized fields including input_ids and attention_mask.
        """
        example = {"prompt": self.build_messages(record)}
        encoded = self.encode(example, tokenizer, inference=False)

        input_ids = encoded.get("input_ids") or []
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
