from typing import List

from pydantic import BaseModel, Field, field_validator


class RubricItem(BaseModel):
    """Represents a single criterion in the evaluation rubric.

    Attributes:
        criterion (str): The name of the evaluation criterion.
        description (str): A detailed description of the criterion.
        weight (float): The weight of the criterion in the overall score (must be non-negative).

    Returns:
        RubricItem: An instance of the RubricItem class.

    """

    criterion: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(ge=0)

    @field_validator("criterion", "description", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if value is None:
            raise ValueError("value is required")
        text = str(value).strip()
        if not text:
            raise ValueError("value cannot be empty")
        return text


class EvaluationSample(BaseModel):
    """Represents a single evaluation sample for a given task.

    Attributes:
        task (str): The description of the task being evaluated.
        reference_answer (str): The correct answer for the task.
        answer (str): The answer provided by the model being evaluated.
        rubric (List[RubricItem]): A list of RubricItem instances representing the evaluation criteria.
        score (int): The overall score for the evaluation sample (must be non-negative).
        reasoning (str): The reasoning behind the evaluation score.

    Returns:
        EvaluationSample: An instance of the EvaluationSample class.
    """

    task: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rubric: List[RubricItem] = Field(min_length=1)
    score: int = Field(ge=0)
    reasoning: str = Field(min_length=1)

    @field_validator(
        "task",
        "reference_answer",
        "answer",
        "reasoning",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if value is None:
            raise ValueError("value is required")
        text = str(value).strip()
        if not text:
            raise ValueError("value cannot be empty")
        return text
