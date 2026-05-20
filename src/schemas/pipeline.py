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

    @classmethod
    def as_form(
        cls,
        model: PipelineModel = Form(
            default=PipelineModel.MISTRAL_7B_INSTRUCT_BNB_4BIT
        ),
    ) -> "PipelineRequest":
        return cls(model=model)


class PipelineResponse(BaseModel):
    upload_id: str
    result_file_path: str
