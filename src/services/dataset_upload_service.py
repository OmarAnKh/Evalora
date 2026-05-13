import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from pydantic import ValidationError

from src.schemas.dataset import EvaluationSample


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_jsonl_upload(upload) -> tuple[list[EvaluationSample], list[dict[str, Any]]]:
    """Parses a JSONL file upload and validates each line against the EvaluationSample schema.
    Args:
        upload: The uploaded file object containing the JSONL data.
    Returns:
        A tuple containing a list of valid EvaluationSample records and a list of errors encountered during parsing.
    """
    records: list[EvaluationSample] = []
    errors: list[dict[str, Any]] = []

    upload.file.seek(0)
    for line_number, raw_line in enumerate(upload.file, start=1):
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            errors.append({"line": line_number, "error": f"utf-8 decode error: {exc}"})
            continue

        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": f"json decode error: {exc}"})
            continue

        try:
            record = EvaluationSample.model_validate(payload)
        except ValidationError as exc:
            errors.append({"line": line_number, "error": exc.errors()})
            continue

        records.append(record)

    return records, errors


def save_jsonl_records(records: list[EvaluationSample]) -> str:
    """Saves a list of EvaluationSample records to a JSONL file in the data/raw directory.
    Args:
        records: A list of EvaluationSample instances to be saved.
    Returns:
        The file path of the saved JSONL file.
    """
    base_dir = _repo_root() / "data" / "raw"
    base_dir.mkdir(parents=True, exist_ok=True)

    dataset_uuid = str(uuid.uuid4())
    output_path = base_dir / f"{dataset_uuid}.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record.model_dump(), ensure_ascii=True)
            handle.write(line + "\n")

    return str(output_path)
