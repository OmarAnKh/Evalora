from pathlib import Path
from typing import Any

from fastapi import UploadFile

from src.data.formatter import Formatter
from src.schemas.pipeline import PipelineModel
from src.services.dataset_upload_service import parse_jsonl_upload, save_jsonl_records
from src.services.preprocessing_service import PreprocessingService


def _extract_upload_id(upload_path: str) -> str:
    return Path(upload_path).stem


class PipelineService:
    def run(self, file: UploadFile, model: PipelineModel) -> dict[str, Any]:
        records, errors = parse_jsonl_upload(file)
        if errors:
            return {
                "status": "failed",
                "accepted": len(records),
                "rejected": len(errors),
                "errors": errors,
            }

        upload_path = save_jsonl_records(records)
        upload_id = _extract_upload_id(upload_path)

        formatter = Formatter(model_name=model.value)
        preprocessing_service = PreprocessingService(formatter=formatter)

        try:
            preprocessing_service.format(upload_id)
            preprocessing_service.split(upload_id)
            tokenize_result = preprocessing_service.tokenize(upload_id)
        except ValueError as exc:
            return {
                "status": "failed",
                "accepted": len(records),
                "rejected": 0,
                "errors": [{"error": str(exc)}],
            }

        return {
            "status": "ok",
            "upload_id": upload_id,
            "result_file_path": tokenize_result["tokenized_file_path"],
        }
