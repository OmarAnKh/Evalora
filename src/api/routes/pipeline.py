from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse

from src.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineTrainEvaluateRequest,
    PipelineTrainEvaluateResponse,
)
from src.services.pipeline_service import PipelineService
from src.services.training_service import TrainingService
from src.services.evaluation import EvaluationService
from src.schemas.training import TrainingRequest

router = APIRouter()
pipeline_service = PipelineService()
training_service = TrainingService()
evaluation_service = EvaluationService()


def _comparison_dir(upload_id: str, split: str) -> str:
    return str(Path("experiments") / upload_id / "evaluation" / split)


def _remove_upload_id(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "upload_id"}


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


@router.post("/full-run", response_model=PipelineTrainEvaluateResponse)
async def run_pipeline_train_evaluate(
    file: UploadFile = File(...),
    training_config: UploadFile | None = File(None),
    payload: PipelineTrainEvaluateRequest = Depends(
        PipelineTrainEvaluateRequest.as_form
    ),
) -> JSONResponse:
    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")

    pipeline_result = pipeline_service.run(
        file,
        payload.base_model_name,
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        test_ratio=payload.test_ratio,
        seed=payload.seed,
    )
    if pipeline_result.get("status") != "ok":
        return JSONResponse(status_code=422, content=pipeline_result)

    upload_id = pipeline_result["upload_id"]
    training_request = TrainingRequest(
        upload_id=upload_id,
        model_name=payload.base_model_name,
        experiment_name=payload.experiment_name,
        num_train_epochs=payload.num_train_epochs,
        learning_rate=payload.learning_rate,
        per_device_train_batch_size=payload.per_device_train_batch_size,
        gradient_accumulation_steps=payload.gradient_accumulation_steps,
    )

    try:
        training_result = training_service.train(training_config, training_request)
        evaluation_result = evaluation_service.compare_baseline_and_finetuned(
            upload_id=upload_id,
            model_name=payload.base_model_name,
            split=payload.evaluation_split,
            use_kappa=payload.use_cohen_kappa,
            use_bertscore=payload.use_bertscore,
            output_dir=_comparison_dir(upload_id, payload.evaluation_split),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evaluation_result = _remove_upload_id(evaluation_result)
    if "baseline" in evaluation_result:
        evaluation_result["baseline"] = _remove_upload_id(evaluation_result["baseline"])
    if "finetuned" in evaluation_result:
        evaluation_result["finetuned"] = _remove_upload_id(
            evaluation_result["finetuned"]
        )

    response_payload = {
        "upload_id": upload_id,
        "pipeline": {
            "upload_id": upload_id,
            "result_file_path": pipeline_result["result_file_path"],
        },
        "training": training_result,
        "evaluation": evaluation_result,
    }

    return JSONResponse(status_code=200, content=response_payload)
