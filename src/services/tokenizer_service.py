from typing import Any

from src.services.model_loader import (
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_NAME,
    get_tokenizer,
)


class TokenizerService:
    """Apply chat templates and tokenize formatted message prompts."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    ) -> None:
        self._tokenizer = get_tokenizer(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

    def encode(self, example: dict[str, Any]) -> dict[str, Any]:
        """Encode a single chat message prompt into token IDs."""
        messages = example.get("messages")
        if messages is None:
            raise ValueError("Expected 'messages' in formatted sample for tokenization.")

        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return self._tokenizer(encoded, truncation=True)

    def tokenize(self, dataset) -> Any:
        """Tokenize a dataset of formatted prompts."""
        return dataset.map(self.encode, remove_columns=dataset.column_names)
