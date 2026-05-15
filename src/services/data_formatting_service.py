import json
from pathlib import Path
from pydantic import ValidationError

from src.data.formatter import Formatter


class DataFormattingService:
    """
    Service for formatting evaluation samples into Mistral instruction-tuning text samples.
    """

    def __init__(self):
        self.formatter = Formatter()

    def format_sample(self, record) -> dict[str, str]:
        """
        Formats a single evaluation sample record.
        Args:
            record: An evaluation sample record to format.
        Returns:
            dict[str, str]: A dictionary containing the formatted text."""

        return self.formatter.format(record)

    def format_batch(self, records: list) -> list[dict[str, str]]:
        """Formats a batch of evaluation sample records.
        Args:
            records (list): A list of evaluation sample records to format.
        Returns:
            list[dict[str, str]]: A list of dictionaries containing the formatted text for each record.
        """

        return self.formatter.format_batch(records)

    def format_file(self, file_id: str) -> list[dict[str, str]]:
        """Formats evaluation sample records from a specified file.
        Args:
            file_id (str): The identifier of the file containing evaluation sample records to format.
        Returns:
            list[dict[str, str]]: A list of dictionaries containing the formatted text for each record in the file.
        """

        formatted_records = self.formatter.format_file(file_id)

        if not formatted_records:
            raise FileNotFoundError(f"No records found for file_id: {file_id}")

        output_path = self._save_formatted(file_id, formatted_records)

        return {
            "status": "ok",
            "formatted_records": len(formatted_records),
            "path": output_path,
        }

    def _save_formatted(self, file_path: str, records: list[dict[str, str]]) -> str:
        """_save_formatted saves formatted records to a file at the specified path.
        Args:
            file_path (str): The full path where formatted records should be saved.
            records (list[dict[str, str]]): The list of formatted records to save.
        Returns:
            str: The path where formatted records were saved."""

        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / "data" / "formatted" / file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                json_record = json.dumps(record, ensure_ascii=False)
                handle.write(json_record + "\n")

        return str(output_path)
