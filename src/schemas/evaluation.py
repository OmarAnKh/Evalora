from typing import List

from pydantic import BaseModel, Field, field_validator


class EvaluationResultRequest(BaseModel):
    """Represents the evaluation results for a given model.
        """
    upload_id: str = Field(..., description="Unique identifier for the uploaded dataset")
    use_cohen_kappa: bool = Field(default=True, description="Whether to compute Cohen's Kappa for score evaluation")
    