import hashlib
import json
import unicodedata
from typing import Any
from datasets import Dataset


class Preprocessor:

    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocesses the input dataset by normalizing text fields and removing duplicates.
        Args:
            dataset (datasets.Dataset): A Hugging Face Dataset containing the evaluation samples to be preprocessed.
        Returns:
            datasets.Dataset: A new Dataset where each example has been normalized and duplicates have been removed.
        """
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []

        for row in dataset:
            normalized_row = self._normalize(row)
            payload = {
                "task": normalized_row.get("task", ""),
                "reference_answer": normalized_row.get("reference_answer", ""),
                "answer": normalized_row.get("answer", ""),
                "rubric": normalized_row.get("rubric", ""),
            }
            signature = self._build_signature(payload)

            if signature in seen:
                continue

            seen.add(signature)
            rows.append(normalized_row)

        return Dataset.from_list(rows)

    def _build_signature(self, row: dict[str, Any]) -> str:
        """Builds a unique signature for a given row by hashing its normalized content.
        Args:
            row (dict): A dictionary representing a single evaluation sample, containing fields like 'task', 'reference_answer', 'answer', and 'rubric'.
        Returns:
            str: A unique hash string that serves as a signature for the input row, allowing for duplicate detection.
        """
        payload = json.dumps(
            row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _normalize(self, value: Any) -> Any:
        """Normalizes a given value by stripping whitespace and normalizing Unicode characters.
        Args:
            value (Any): The value to be normalized.
        Returns:
            Any: The normalized value.
        """
        if isinstance(value, dict):
            return {key: self._normalize(val) for key, val in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]

        if isinstance(value, str):
            text = unicodedata.normalize("NFKC", value)
            return " ".join(text.strip().split())

        return value
