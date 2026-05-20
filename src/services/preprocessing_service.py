from datasets import load_dataset
from pathlib import Path
from typing import Dict, Optional


from src.data.preprocessor import Preprocessor
from src.data.formatter import Formatter


class PreprocessingService:
    def __init__(self, formatter: Optional[Formatter] = None) -> None:
        """Initializes the PreprocessingService with instances of the Preprocessor and Formatter classes."""
        self._preprocessor = Preprocessor()
        self._formatter = formatter or Formatter()

    def _build_path(self, file_id: str, stage: str) -> Path:
        """Builds a file path for a given file ID and processing stage.
        Args:
            file_id (str): The unique identifier for the dataset file.
            stage (str): The processing stage, which can be 'raw', 'processed', '
            'splits', 'formatted', or 'tokenized'.           
        Returns:
            Path: The constructed file path.
        """
        repo_path = Path(__file__).resolve().parents[2]
        base_path = repo_path / "data"
        if stage == "raw":
            return base_path / "raw" / f"{file_id}.jsonl"
        elif stage == "processed":
            return base_path / "processed" / f"{file_id}_preprocessed.jsonl"
        elif stage == "splits":
            return base_path / "splits" / file_id
        elif stage == "formatted":
            return base_path / "formatted" / f"{file_id}_formatted.jsonl"
        elif stage == "tokenized":
            return base_path / "tokenized" / file_id
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def format(self, file_id: str) -> Dict:
        """Formats the dataset corresponding to the given file ID by applying preprocessing and reformatting steps.
        Args:
            file_id (str): The unique identifier for the dataset file to be formatted.
        Returns:
            dict: A dictionary containing the results of the formatting process, including counts of accepted and rejected records and the path to the formatted file.
        Raises:
            FileNotFoundError: If the raw dataset file corresponding to the given file ID does not exist.
        """
        file_path = self._build_path(file_id, "raw")
        dataset = load_dataset("json", data_files=str(file_path), split="train")

        preprocessed = self._preprocessor.preprocess(dataset)

        formatted = self._formatter.reformat(preprocessed)
        formatted_path = self._build_path(file_id, "formatted")
        # ensure parent dir exists
        formatted_path.parent.mkdir(parents=True, exist_ok=True)

        # allow both pandas.DataFrame and datasets.Dataset
        try:
            formatted.to_json(str(formatted_path), orient="records", lines=True)
        except TypeError:
            formatted.to_json(str(formatted_path))

        return {
            "accepted_records": len(formatted),
            "rejected_records": len(dataset) - len(formatted),
            "formatted_file_path": str(formatted_path),
        }

    def split(self, file_id: str) -> Dict:
        """Splits the formatted dataset corresponding to the given file ID into training, validation, and test sets.
        Args:
            file_id (str): The unique identifier for the dataset file to be split.
        Returns:
            dict: A dictionary containing the results of the splitting process, including counts of records in each split and the paths to the split files.
        Raises:
            FileNotFoundError: If the formatted dataset file corresponding to the given file ID does not exist.
        """
        file_path = self._build_path(file_id, "formatted")
        dataset = load_dataset("json", data_files=str(file_path), split="train")

        if len(dataset) < 2:
            raise ValueError("Need at least 2 records to split the dataset.")

        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        test_set = train_test["test"]

        if len(test_set) < 2:
            data = {
                "train": train_test["train"],
                "validation": test_set,
                "test": test_set.select([]),
            }
        else:
            test_valid = test_set.train_test_split(test_size=0.5, seed=42)
            data = {
                "train": train_test["train"],
                "validation": test_valid["train"],
                "test": test_valid["test"],
            }

        splits_dir = self._build_path(file_id, "splits")
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_output_path = splits_dir / f"{file_id}_train.jsonl"
        test_output_path = splits_dir / f"{file_id}_test.jsonl"
        validation_output_path = splits_dir / f"{file_id}_validation.jsonl"

        data["train"].to_json(str(train_output_path), orient="records", lines=True)
        data["test"].to_json(str(test_output_path), orient="records", lines=True)
        data["validation"].to_json(
            str(validation_output_path), orient="records", lines=True
        )

        return {
            "train_records": len(data["train"]),
            "test_records": len(data["test"]),
            "validation_records": len(data["validation"]),
            "train_file_path": str(train_output_path),
            "test_file_path": str(test_output_path),
            "validation_file_path": str(validation_output_path),
        }

    def tokenize(self, file_id: str) -> Dict:
        """Tokenizes the split datasets corresponding to the given file ID using the formatter's tokenizer.
        Args:
            file_id (str): The unique identifier for the dataset file to be tokenized.
        Returns:
            dict: A dictionary containing the results of the tokenization process, including counts of accepted and rejected records and the paths to the tokenized files.
        Raises:
            FileNotFoundError: If the split dataset files corresponding to the given file ID do not exist.
        """
        splits_dir = self._build_path(file_id, "splits")
        train_path = splits_dir / f"{file_id}_train.jsonl"
        test_path = splits_dir / f"{file_id}_test.jsonl"
        validation_path = splits_dir / f"{file_id}_validation.jsonl"

        train = load_dataset("json", data_files=str(train_path), split="train")

        def _load_or_empty(path: Path, reference):
            if not path.exists() or path.stat().st_size == 0:
                return reference.select([])
            return load_dataset("json", data_files=str(path), split="train")

        test = _load_or_empty(test_path, train)
        validation = _load_or_empty(validation_path, train)

        train_tokenized = self._formatter.tokenize(train)
        test_tokenized = self._formatter.tokenize(test)
        validation_tokenized = self._formatter.tokenize(validation)

        tokenized_dir = self._build_path(file_id, "tokenized")
        tokenized_dir.mkdir(parents=True, exist_ok=True)

        train_out = tokenized_dir / f"{file_id}_train_tokenized.jsonl"
        test_out = tokenized_dir / f"{file_id}_test_tokenized.jsonl"
        validation_out = tokenized_dir / f"{file_id}_validation_tokenized.jsonl"

        train_tokenized.to_json(str(train_out), orient="records", lines=True)
        test_tokenized.to_json(str(test_out), orient="records", lines=True)
        validation_tokenized.to_json(str(validation_out), orient="records", lines=True)

        accepted = (
            len(train_tokenized) + len(test_tokenized) + len(validation_tokenized)
        )
        rejected = (
            (len(train) - len(train_tokenized))
            + (len(test) - len(test_tokenized))
            + (len(validation) - len(validation_tokenized))
        )

        return {
            "accepted_records": accepted,
            "rejected_records": rejected,
            "tokenized_file_path": str(tokenized_dir),
            "train_tokenized_file_path": str(train_out),
            "test_tokenized_file_path": str(test_out),
            "validation_tokenized_file_path": str(validation_out),
        }
