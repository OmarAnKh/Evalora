import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.preprocessor import Preprocessor
from src.data.splitter import Splitter
from src.schemas.dataset import EvaluationSample


class PreprocessingService:
    """PreprocessingService is responsible for managing the preprocessing of uploaded dataset files. It utilizes the Preprocessor class to read, validate, and normalize the records in the dataset, and then saves the processed records to a new file. The service provides a method to preprocess a given dataset file and returns a summary of the preprocessing results, including the number of accepted and rejected records, any errors encountered, and the path to the processed file if applicable."""

    def __init__(self) -> None:
        self._preprocessor = Preprocessor()
        self._splitter = Splitter()

    def preprocess(self, file_id: str) -> dict[str, Any]:
        """preprocess executes the preprocessing steps for the specified dataset file. It validates and normalizes the records, removes duplicates, and saves the processed records to a new file. The method returns a summary of the preprocessing results, including the status, counts of accepted and rejected records, any errors encountered, and the path to the processed file if applicable.

        Args:
            file_id (str): The identifier of the dataset file to preprocess.

        Returns:
            dict[str, Any]: A summary of the preprocessing results.
        """
        if not file_id.endswith(".jsonl"):
            file_id = f"{file_id}.jsonl"

        records, errors = self._preprocessor.run(file_id)

        output_path = None
        if records:
            repo_root = Path(__file__).resolve().parents[2]
            file_path = str(repo_root / "data" / "processed" / f"processed_{file_id}")
            output_path = self._save_processed(file_path, records)

        return {
            "status": "ok" if records else "empty",
            "accepted": len(records),
            "rejected": len(errors),
            "errors": errors,
            "path": output_path,
        }

    def preprocess_records(
        self,
        records: list[EvaluationSample],
    ) -> dict[str, Any]:
        """preprocess_records runs preprocessing for in-memory records.

        Args:
            records (list[EvaluationSample]): Parsed evaluation samples.

        Returns:
            dict[str, Any]: A summary of preprocessing results and normalized records.
        """

        processed, errors = self._preprocessor.run_records(records)

        return {
            "status": "ok" if processed else "empty",
            "accepted": len(processed),
            "rejected": len(errors),
            "errors": errors,
            "records": processed,
        }

    def _save_processed(self, file_path: str, records: list[EvaluationSample]) -> str:
        """_save_processed saves records to a file at the specified path.

        Args:
            file_path (str): The full path where records should be saved.
            records (list[EvaluationSample]): The list of evaluation samples.

        Returns:
            str: The path where records were saved.
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                line = json.dumps(record.model_dump(), ensure_ascii=True)
                handle.write(line + "\n")

        return str(output_path)

    def split(self, file_id: str) -> dict[str, Any]:
        """split executes the dataset splitting process for the specified preprocessed dataset file."""
        if not file_id.endswith(".jsonl"):
            file_id = f"{file_id}.jsonl"

        repo_root = Path(__file__).resolve().parents[2]
        subdir = "processed" if file_id.startswith("processed_") else "raw"
        source_path = repo_root / "data" / subdir / file_id

        if not source_path.exists():
            raise FileNotFoundError(f"file not found: {file_id}")

        splits = self._splitter.split(str(source_path))
        output_dir = repo_root / "data" / "splits" / Path(source_path).stem

        saved_splits = {
            name: self._save_processed(str(output_dir / f"{name}.jsonl"), split_records)
            for name, split_records in splits.items()
        }

        return {
            "status": "ok",
            "source_path": str(source_path),
            "output_dir": str(output_dir),
            "total": sum(len(split_records) for split_records in splits.values()),
            "splits": {
                name: {"count": len(split_records), "path": saved_splits[name]}
                for name, split_records in splits.items()
            },
        }
