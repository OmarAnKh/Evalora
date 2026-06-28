from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.dataset import RubricItem
from src.training.config import DEFAULT_MODEL_NAME


class ModelInferenceRequest(BaseModel):
    upload_id: str = Field(
        ..., description="Trained model id stored under models/{upload_id}/lora"
    )
    reference_answer: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rubric: list[RubricItem] = Field(..., min_length=1)
    task: str | None = Field(
        default=None,
        description="Optional task prompt; a default evaluation prompt is used when omitted.",
    )
    model_name: str = Field(default=DEFAULT_MODEL_NAME)


class ModelInferenceResponse(BaseModel):
    upload_id: str
    score: float | int | None
    reasoning: str
