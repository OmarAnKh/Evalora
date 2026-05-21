from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


def _load_jsonl(path: str | None) -> Dataset | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return None
    return load_dataset("json", data_files=str(file_path), split="train")


def load_sft_splits(
    train_file: str,
    validation_file: str | None = None,
    test_file: str | None = None,
) -> DatasetDict:
    """Load formatted chat JSONL splits for supervised fine-tuning."""
    train = _load_jsonl(train_file)
    if train is None:
        raise FileNotFoundError(f"Training file not found or empty: {train_file}")

    splits: dict[str, Dataset] = {"train": train}
    validation = _load_jsonl(validation_file)
    test = _load_jsonl(test_file)
    if validation is not None:
        splits["validation"] = validation
    if test is not None:
        splits["test"] = test
    return DatasetDict(splits)


def add_chat_template_text(dataset: Dataset, tokenizer) -> Dataset:
    """Convert message lists to model-ready text while preserving labels for evaluation."""
    def render(example: dict[str, Any]) -> dict[str, str]:
        messages = example.get("messages")
        if not messages:
            raise ValueError("Expected a 'messages' list in each formatted training sample.")
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    return dataset.map(render)


def prepare_sft_datasets(splits: DatasetDict, tokenizer) -> DatasetDict:
    """Render chat-template text lazily enough for large JSONL-backed experiments."""
    return DatasetDict(
        {name: add_chat_template_text(split, tokenizer) for name, split in splits.items()}
    )
