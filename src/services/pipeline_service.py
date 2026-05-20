from pathlib import Path
from typing import Any

from fastapi import UploadFile

from src.schemas.pipeline import PipelineModel
from src.services.dataset_upload_service import parse_jsonl_upload, save_jsonl_records
from src.services.preprocessing_service import PreprocessingService


def _extract_upload_id(upload_path: str) -> str:
    return Path(upload_path).stem


class PipelineService:
    def run(
        self,
        file: UploadFile,
        model: PipelineModel,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, Any]:
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

        preprocessing_service = PreprocessingService(model_name=model.value)

        try:
            preprocessing_service.format(upload_id)
            preprocessing_service.split(
                upload_id,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
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
