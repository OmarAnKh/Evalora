import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas.dataset import EvaluationSample


class Preprocessor:
    """Preprocessor is responsible for reading raw dataset files, validating and normalizing the records,
    and removing duplicates based on a content signature.
    """

    def run(
        self,
        file_id: str,
    ) -> tuple[list[EvaluationSample], list[dict[str, Any]]]:
        """run executes the preprocessing steps on the specified dataset file. It reads the file, validates each record against the EvaluationSample schema,
            normalizes the content, and builds a signature to filter out duplicates.

        Args:
            file_id (str): The identifier of the dataset file to preprocess, typically the filename.

        Raises:
            FileNotFoundError: Raised if the specified file is not found.

        Returns:
            tuple[list[EvaluationSample], list[dict[str, Any]]]: A tuple containing the list of preprocessed records and the list of errors.
        """

        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / "raw" / file_id

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_id}")

        records: list[EvaluationSample] = []
        errors: list[dict[str, Any]] = []

        seen: set[str] = set()

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    payload = json.loads(line)

                except json.JSONDecodeError as exc:
                    errors.append(
                        {
                            "line": line_number,
                            "error": (f"json decode error: {exc}"),
                        }
                    )
                    continue

                try:
                    record = EvaluationSample.model_validate(self._normalize(payload))

                except ValidationError as exc:
                    errors.append(
                        {
                            "line": line_number,
                            "error": exc.errors(),
                        }
                    )
                    continue

                signature = self._build_signature(record)

                if signature in seen:
                    continue

                seen.add(signature)

                records.append(record)

        return records, errors

    def run_records(
        self,
        records: list[EvaluationSample],
    ) -> tuple[list[EvaluationSample], list[dict[str, Any]]]:
        """run_records preprocesses already-loaded records in memory.

        Args:
            records (list[EvaluationSample]): Incoming evaluation samples.

        Returns:
            tuple[list[EvaluationSample], list[dict[str, Any]]]: The normalized records and errors.
        """

        processed: list[EvaluationSample] = []
        errors: list[dict[str, Any]] = []
        seen: set[str] = set()

        for index, record in enumerate(records, start=1):
            try:
                payload = self._normalize(record.model_dump())
                normalized = EvaluationSample.model_validate(payload)
            except ValidationError as exc:
                errors.append({"line": index, "error": exc.errors()})
                continue

            signature = self._build_signature(normalized)
            if signature in seen:
                continue

            seen.add(signature)
            processed.append(normalized)

        return processed, errors

    def _build_signature(
        self,
        record: EvaluationSample,
    ) -> str:
        """_build_signature generates a unique signature for a given EvaluationSample record by serializing its content and computing an MD5 hash. This signature is used to identify and filter out duplicate records during preprocessing.

        Args:
            record (EvaluationSample): The evaluation sample for which to build a signature.

        Returns:
            str: The unique signature for the evaluation sample.
        """

        payload = {
            "task": record.task,
            "reference_answer": (record.reference_answer),
            "answer": record.answer,
            "rubric": [rubric.model_dump(mode="json") for rubric in record.rubric],
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def _normalize(
        self,
        value: Any,
    ) -> Any:
        """_normalize normalizes the input value by stripping whitespace from strings and recursively normalizing nested structures.

        Args:
            value (Any): The value to normalize.

        Returns:
            Any: The normalized value, with strings stripped of leading and trailing whitespace and nested structures normalized recursively.
        """

        if isinstance(value, dict):
            return {key: self._normalize(val) for key, val in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]

        if isinstance(value, str):

            text = unicodedata.normalize(
                "NFKC",
                value,
            )

            return " ".join(text.strip().split())

        return value
