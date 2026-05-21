from __future__ import annotations

from pydantic import BaseModel, Field

from src.training.config import DEFAULT_MODEL_NAME


class TrainingRequest(BaseModel):
    """Start a LoRA fine-tuning run for an uploaded dataset id."""

    upload_id: str = Field(..., description="Dataset/experiment id produced by /pipeline/run")
    config_path: str = Field(default="configs/train.yaml")
    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    experiment_name: str | None = Field(default=None)
    num_train_epochs: float = Field(default=3.0, gt=0, le=20)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    per_device_train_batch_size: int = Field(default=1, gt=0, le=16)
    gradient_accumulation_steps: int = Field(default=4, gt=0, le=64)


class TrainingResponse(BaseModel):
    status: str
    output_dir: str
    config_path: str
    metrics: dict


class TrainingModelDefaults(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    config_path: str = "configs/train.yaml"
    output_dir_pattern: str = "models/{upload_id}/lora"
