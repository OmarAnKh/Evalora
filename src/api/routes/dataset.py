from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.pipeline import PipelineModel
from src.services.dataset_upload_service import parse_jsonl_upload, save_jsonl_records
from src.services.preprocessing_service import PreprocessingService

router = APIRouter()


def _get_preprocessing_service(model: PipelineModel | None = None) -> PreprocessingService:
    if model is None:
        return PreprocessingService()
    return PreprocessingService(model_name=model.value)


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


@router.get("/format/{file_id}")
def format_dataset(file_id: str) -> JSONResponse:
    try:
        result = _get_preprocessing_service().format(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JSONResponse(status_code=200, content=result)


@router.get("/split/{file_id}")
def split_dataset(
    file_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> JSONResponse:
    try:
        result = _get_preprocessing_service().split(
            file_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse(status_code=200, content=result)

@router.get("/tokenize/{file_id}")
def tokenize_dataset(
    file_id: str,
    model: PipelineModel = PipelineModel.MISTRAL_7B_INSTRUCT_BNB_4BIT,
) -> JSONResponse:
    try:
        result = _get_preprocessing_service(model).tokenize(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse(status_code=200, content=result)

