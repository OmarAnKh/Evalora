from enum import Enum

from fastapi import Form
from pydantic import BaseModel, Field


class PipelineModel(str, Enum):
    MISTRAL_7B_INSTRUCT_BNB_4BIT = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    LLAMA_3_1_8B_INSTRUCT_BNB_4BIT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


class PipelineRequest(BaseModel):
    model: PipelineModel = Field(
        default=PipelineModel.MISTRAL_7B_INSTRUCT_BNB_4BIT,
        description="Model identifier to use for tokenization.",
    )
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)

    @classmethod
    def as_form(
        cls,
        model: PipelineModel = Form(
            default=PipelineModel.MISTRAL_7B_INSTRUCT_BNB_4BIT
        ),
        train_ratio: float = Form(default=0.8),
        val_ratio: float = Form(default=0.1),
        test_ratio: float = Form(default=0.1),
        seed: int = Form(default=42),
    ) -> "PipelineRequest":
        return cls(
            model=model,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )


class PipelineResponse(BaseModel):
    upload_id: str
    result_file_path: str
