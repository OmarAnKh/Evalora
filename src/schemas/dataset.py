from typing import List

from pydantic import BaseModel, Field, field_validator


class RubricItem(BaseModel):
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
