from typing import Any

from src.data.formatter import Formatter
from src.schemas.dataset import EvaluationSample


class DataFormattingService:
    """Service for formatting and tokenizing evaluation samples via chat templates."""

    def __init__(self):
        self.formatter = Formatter()

    def format_sample(self, record: EvaluationSample, tokenizer: Any) -> dict[str, list[int]]:
        """Formats and tokenizes a single evaluation sample record.
        Args:
            record: An evaluation sample record to format.
            tokenizer: A Hugging Face tokenizer with apply_chat_template.
        Returns:
            dict[str, list[int]]: Tokenized output with input_ids and attention_mask.
        """

        return self.formatter.apply_chat_template(tokenizer, record)

    def format_batch(
        self,
        records: list[EvaluationSample],
        tokenizer: Any,
    ) -> list[dict[str, list[int]]]:
        """Formats and tokenizes a batch of evaluation sample records.
        Args:
            records (list[EvaluationSample]): A list of evaluation sample records to format.
            tokenizer: A Hugging Face tokenizer with apply_chat_template.
        Returns:
            list[dict[str, list[int]]]: Tokenized outputs for each record.
        """

        return [self.format_sample(record, tokenizer) for record in records]

    def format_file(
        self,
        file_id: str,
        tokenizer: Any,
    ) -> list[dict[str, list[int]]]:
        """Formats and tokenizes evaluation sample records from a specified file.
        Args:
            file_id (str): The identifier of the file containing evaluation sample records to format.
            tokenizer: A Hugging Face tokenizer with apply_chat_template.
        Returns:
            list[dict[str, list[int]]]: Tokenized outputs for each record in the file.
        """

        records = self.formatter.load_records_from_file(file_id)

        if not records:
            raise FileNotFoundError(f"No records found for file_id: {file_id}")

        return self.format_batch(records, tokenizer)
