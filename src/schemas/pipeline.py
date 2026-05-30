from enum import Enum

from fastapi import Form
from pydantic import BaseModel, Field

from src.training.config import DEFAULT_MODEL_NAME


class PipelineModel(str, Enum):
    MISTRAL_7B_INSTRUCT_BNB_4BIT = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    LLAMA_3_1_8B_INSTRUCT_BNB_4BIT = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


class PipelineRequest(BaseModel):
    model: str = Field(default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit")
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)

    @classmethod
    def as_form(
        cls,
        model: str = Form(default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit"),
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


class PipelineTrainEvaluateRequest(BaseModel):
    base_model_name: str = Field(default=DEFAULT_MODEL_NAME)
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    experiment_name: str | None = None
    num_train_epochs: float = 3
    learning_rate: float = 0.0002
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    evaluation_split: str = "test"
    use_cohen_kappa: bool = True
    use_bertscore: bool = True

    @classmethod
    def as_form(
        cls,
        base_model_name: str = Form(DEFAULT_MODEL_NAME),
        train_ratio: float = Form(0.8),
        val_ratio: float = Form(0.1),
        test_ratio: float = Form(0.1),
        seed: int = Form(42),
        experiment_name: str | None = Form(None),
        num_train_epochs: float = Form(3),
        learning_rate: float = Form(0.0002),
        per_device_train_batch_size: int = Form(1),
        gradient_accumulation_steps: int = Form(4),
        use_cohen_kappa: bool = Form(True),
        use_bertscore: bool = Form(True),
    ) -> "PipelineTrainEvaluateRequest":
        return cls(
            base_model_name=base_model_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            experiment_name=experiment_name,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_cohen_kappa=use_cohen_kappa,
            use_bertscore=use_bertscore,
        )


class PipelineTrainEvaluateResponse(BaseModel):
    upload_id: str
    pipeline: PipelineResponse
    training: dict
    evaluation: dict
