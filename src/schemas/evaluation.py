from __future__ import annotations

from pydantic import BaseModel, Field

from src.training.config import DEFAULT_MODEL_NAME


class BaselineEvaluationRequest(BaseModel):
    """Evaluate the base instruction model on a held-out split."""

    upload_id: str = Field(..., description="Unique identifier for the uploaded dataset/experiment")
    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    split: str = Field(default="test")
    use_cohen_kappa: bool = Field(default=True)
    use_bertscore: bool = Field(default=True)


class FinetunedEvaluationRequest(BaselineEvaluationRequest):
    """Evaluate the LoRA adapter saved for this upload_id."""


class ComparisonEvaluationRequest(BaselineEvaluationRequest):
    """Evaluate baseline and fine-tuned models on the same held-out split."""


class EvaluationResultRequest(BaselineEvaluationRequest):
    """Generic model evaluation request for a saved dataset split."""

    use_finetuned: bool = Field(
        default=False,
        description="When true, load models/{upload_id}/lora; otherwise evaluate the baseline model.",
    )
