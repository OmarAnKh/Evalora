from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from src.schemas.pipeline import FormatRequest, FormatResponse
from src.services.dataset_upload_service import parse_jsonl_upload, save_jsonl_records
from src.services.dataset_preprocessing_service import PreprocessingService
from src.services.data_formatting_service import DataFormattingService

router = APIRouter()
preprocessing_service = PreprocessingService()
data_formatting_service = DataFormattingService()


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")

    records, errors = parse_jsonl_upload(file)

    if errors:
        return JSONResponse(
            status_code=422,
            content={
                "status": "failed",
                "accepted": len(records),
                "rejected": len(errors),
                "errors": errors,
            },
        )

    output_path = save_jsonl_records(records)
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "records": len(records), "path": output_path},
    )


@router.post("/preprocess/{file_id}")
def preprocess_dataset(file_id: str) -> JSONResponse:
    result = preprocessing_service.preprocess(file_id)

    if result["status"] == "empty":
        return JSONResponse(
            status_code=200,
            content={
                "status": "empty",
                "accepted": 0,
                "rejected": len(result["errors"]),
                "errors": result["errors"],
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": result["status"],
            "accepted": result["accepted"],
            "rejected": result["rejected"],
            "errors": result["errors"],
            "path": result["path"],
        },
    )


@router.get("/split/{file_id}")
def split_dataset(file_id: str) -> JSONResponse:
    try:
        result = preprocessing_service.split(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JSONResponse(status_code=200, content=result)


@router.post("/format/{file_id}", response_model=FormatResponse)
def format_dataset(file_id: str, request: FormatRequest) -> FormatResponse:
    try:
        tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_model.value)
        records = data_formatting_service.format_file(file_id, tokenizer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return FormatResponse(status="ok", formatted_records=len(records), records=records)
