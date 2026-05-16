from enum import Enum
from typing import Any

from fastapi import Form
from pydantic import BaseModel, Field


class TokenizerModel(str, Enum):
    mistral_7b_instruct_v0_2 = "mistralai/Mistral-7B-Instruct-v0.2"


class PipelineRequest(BaseModel):
    tokenizer_model: TokenizerModel = Field(...)
    train_ratio: float = Field(0.8, ge=0.0, le=1.0)
    validation_ratio: float = Field(0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(0.1, ge=0.0, le=1.0)

    @classmethod
    def as_form(
        cls,
        tokenizer_model: TokenizerModel = Form(...),
        train_ratio: float = Form(0.8),
        validation_ratio: float = Form(0.1),
        test_ratio: float = Form(0.1),
    ) -> "PipelineRequest":
        return cls(
            tokenizer_model=tokenizer_model,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )


class TokenizedSample(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]


class SplitPayload(BaseModel):
    count: int
    records: list[TokenizedSample]


class PipelineResponse(BaseModel):
    status: str
    accepted: int
    rejected: int
    errors: list[dict[str, Any]]
    total: int
    splits: dict[str, SplitPayload]


class FormatRequest(BaseModel):
    tokenizer_model: TokenizerModel = Field(...)


class FormatResponse(BaseModel):
    status: str
    formatted_records: int
    records: list[TokenizedSample]
