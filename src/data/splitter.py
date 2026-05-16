import json
import random
from pathlib import Path
from typing import Any

from src.schemas.dataset import EvaluationSample


class Splitter:
    """Splitter creates train, validation, and test JSONL splits from a dataset file."""

    def __init__(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio, validation_ratio, and test_ratio must sum to 1.0"
            )

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, file_path: str) -> dict[str, list[EvaluationSample]]:
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        records = self._load_records(source_path)
        return self.split_records(records)

    def split_records(self, records: list[Any]) -> dict[str, list[Any]]:
        """Splits preloaded records into train/validation/test sets."""
        shuffled = list(records)
        random.Random(self.seed).shuffle(shuffled)
        train_records, validation_records, test_records = self._partition_records(
            shuffled
        )

        return {
            "train": train_records,
            "validation": validation_records,
            "test": test_records,
        }

    def _load_records(self, source_path: Path) -> list[EvaluationSample]:
        with source_path.open("r", encoding="utf-8") as handle:
            return [
                EvaluationSample.model_validate(json.loads(line))
                for line in handle
                if line.strip()
            ]

    def _partition_records(
        self,
        records: list[EvaluationSample],
    ) -> tuple[list[EvaluationSample], list[EvaluationSample], list[EvaluationSample]]:
        total = len(records)
        if total == 0:
            return [], [], []

        counts = [
            int(total * ratio)
            for ratio in (self.train_ratio, self.validation_ratio, self.test_ratio)
        ]
        if total >= 3:
            counts = [max(1, count) for count in counts]
            while sum(counts) > total:
                counts[counts.index(max(counts))] -= 1
            while sum(counts) < total:
                counts[0] += 1

        train_end = counts[0]
        validation_end = train_end + counts[1]
        return (
            records[:train_end],
            records[train_end:validation_end],
            records[validation_end : validation_end + counts[2]],
        )
