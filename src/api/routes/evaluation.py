from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from schemas.evaluation import EvaluationResultRequest
from src.evaluation.evaluators.evaluator import evaluate
from src.evaluation.outputs.save_results import save_evaluation_results
from src.services.evaluation import evaluation_service
router = APIRouter()

service = evaluation_service()


@router.post("/evaluate")
def evaluate_models(request: EvaluationResultRequest):
    try:
         service.evaluate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
