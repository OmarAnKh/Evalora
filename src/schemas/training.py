from __future__ import annotations

from pydantic import BaseModel

from src.training.config import DEFAULT_MODEL_NAME

from fastapi import Form

class TrainingRequest(BaseModel):
    upload_id: str
    model_name: str
    experiment_name: str | None = None
    num_train_epochs: float = 3
    learning_rate: float = 0.0002
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    @classmethod
    def as_form(
        cls,
        upload_id: str = Form(...),
        model_name: str = Form(...),
        experiment_name: str | None = Form(None),
        num_train_epochs: float = Form(3),
        learning_rate: float = Form(0.0002),
        per_device_train_batch_size: int = Form(1),
        gradient_accumulation_steps: int = Form(4),
    ):
        return cls(
            upload_id=upload_id,
            model_name=model_name,
            experiment_name=experiment_name,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


class TrainingResponse(BaseModel):
    status: str
    output_dir: str
    config_path: str
    metrics: dict


class TrainingModelDefaults(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    config_path: str = "configs/train.yaml"
    output_dir_pattern: str = "models/{upload_id}/lora"
