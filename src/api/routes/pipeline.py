from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from transformers import AutoTokenizer

from src.data.splitter import Splitter
from src.schemas.pipeline import PipelineRequest, PipelineResponse, SplitPayload
from src.services.data_formatting_service import DataFormattingService
from src.services.dataset_preprocessing_service import PreprocessingService
from src.services.dataset_upload_service import parse_jsonl_upload

router = APIRouter()
preprocessing_service = PreprocessingService()
formatting_service = DataFormattingService()


@router.post("/run", response_model=PipelineResponse)
async def run_pipeline(
    file: UploadFile = File(...),
    request: PipelineRequest = Depends(PipelineRequest.as_form),
) -> PipelineResponse:
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")

    records, parse_errors = parse_jsonl_upload(file)

    preprocess_result = preprocessing_service.preprocess_records(records)
    processed_records = preprocess_result["records"]

    if not processed_records:
        return PipelineResponse(
            status="empty",
            accepted=0,
            rejected=len(parse_errors) + preprocess_result["rejected"],
            errors=parse_errors + preprocess_result["errors"],
            total=0,
            splits={
                "train": SplitPayload(count=0, records=[]),
                "validation": SplitPayload(count=0, records=[]),
                "test": SplitPayload(count=0, records=[]),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_model.value)
    tokenized_records = formatting_service.format_batch(processed_records, tokenizer)

    try:
        splitter = Splitter(
            train_ratio=request.train_ratio,
            validation_ratio=request.validation_ratio,
            test_ratio=request.test_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    splits = splitter.split_records(tokenized_records)

    split_payload = {
        name: SplitPayload(count=len(split_records), records=split_records)
        for name, split_records in splits.items()
    }

    return PipelineResponse(
        status="ok",
        accepted=len(processed_records),
        rejected=len(parse_errors) + preprocess_result["rejected"],
        errors=parse_errors + preprocess_result["errors"],
        total=len(processed_records),
        splits=split_payload,
    )
