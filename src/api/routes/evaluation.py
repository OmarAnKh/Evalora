from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.schemas.evaluation import (
    BaselineEvaluationRequest,
    ComparisonEvaluationRequest,
    EvaluationResultRequest,
    FinetunedEvaluationRequest,
)
from src.services.evaluation import EvaluationService

router = APIRouter()
service = EvaluationService()


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _prediction_file(upload_id: str, split: str, name: str) -> str:
    return str(Path("experiments") / upload_id / "evaluation" / split / f"{name}_predictions.jsonl")


def _comparison_dir(upload_id: str, split: str) -> str:
    return str(Path("experiments") / upload_id / "evaluation" / split)


@router.post("/baseline")
def evaluate_baseline(request: BaselineEvaluationRequest):
    """Evaluate the base model before fine-tuning."""
    try:
        return service.evaluate_model_on_split(
            upload_id=request.upload_id,
            model_name=request.model_name,
            adapter_path=None,
            split=request.split,
            use_kappa=request.use_cohen_kappa,
            use_bertscore=request.use_bertscore,
            output_file=_prediction_file(request.upload_id, request.split, "baseline"),
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/finetuned")
def evaluate_finetuned(request: FinetunedEvaluationRequest):
    """Evaluate the LoRA adapter stored under models/{upload_id}/lora."""
    try:
        return service.evaluate_finetuned_on_split(
            upload_id=request.upload_id,
            model_name=request.model_name,
            split=request.split,
            use_kappa=request.use_cohen_kappa,
            use_bertscore=request.use_bertscore,
            output_file=_prediction_file(request.upload_id, request.split, "finetuned"),
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/compare")
def compare_baseline_and_finetuned(request: ComparisonEvaluationRequest):
    """Evaluate baseline and models/{upload_id}/lora on the same split."""
    try:
        return service.compare_baseline_and_finetuned(
            upload_id=request.upload_id,
            model_name=request.model_name,
            split=request.split,
            use_kappa=request.use_cohen_kappa,
            use_bertscore=request.use_bertscore,
            output_dir=_comparison_dir(request.upload_id, request.split),
        )
    except Exception as exc:
        _raise_http(exc)


@router.post("/evaluate")
def evaluate_models(request: EvaluationResultRequest):
    """Generic endpoint: set use_finetuned=true to load models/{upload_id}/lora."""
    try:
        if request.use_finetuned:
            return service.evaluate_finetuned_on_split(
                upload_id=request.upload_id,
                model_name=request.model_name,
                split=request.split,
                use_kappa=request.use_cohen_kappa,
                use_bertscore=request.use_bertscore,
                output_file=_prediction_file(request.upload_id, request.split, "finetuned"),
            )
        return service.evaluate_model_on_split(
            upload_id=request.upload_id,
            model_name=request.model_name,
            adapter_path=None,
            split=request.split,
            use_kappa=request.use_cohen_kappa,
            use_bertscore=request.use_bertscore,
            output_file=_prediction_file(request.upload_id, request.split, "baseline"),
        )
    except Exception as exc:
        _raise_http(exc)
