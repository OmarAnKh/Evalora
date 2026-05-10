from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.dataset import EvaluationSample
from src.services.dataset_upload_service import parse_jsonl_upload, save_jsonl_records

router = APIRouter()


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
