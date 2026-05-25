from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse

from src.schemas.pipeline import PipelineRequest, PipelineResponse
from src.services.pipeline_service import PipelineService

router = APIRouter()
pipeline_service = PipelineService()


@router.post("/run", response_model=PipelineResponse)
async def run_pipeline(
    file: UploadFile = File(...),
    payload: PipelineRequest = Depends(PipelineRequest.as_form),
) -> JSONResponse:
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")

    result = pipeline_service.run(
        file,
        payload.model,
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        test_ratio=payload.test_ratio,
        seed=payload.seed,
    )
    if result.get("status") != "ok":
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(
        status_code=200,
        content={
            "upload_id": result["upload_id"],
            "result_file_path": result["result_file_path"],
        },
    )
