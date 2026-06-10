from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from io import StringIO
from fastapi import UploadFile

from src.schemas.training import TrainingRequest
from src.training.config import TrainConfig


class TrainingService:
    """Build and run reproducible LoRA fine-tuning experiments."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.repo_root / path

    def _split_file(self, upload_id: str, split: str) -> str:
        path = (
            self.repo_root
            / "data"
            / "splits"
            / upload_id
            / f"{upload_id}_{split}.jsonl"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split file: {path}")
        return str(path)

    def _apply_request_overrides(
        self, config: TrainConfig, request: TrainingRequest
    ) -> TrainConfig:
        """Override config fields with values from the training request."""
        config.train_file = self._split_file(request.upload_id, "train")
        config.validation_file = self._split_file(request.upload_id, "validation")
        config.test_file = self._split_file(request.upload_id, "test")
        config.output_dir = str(self.repo_root / "models" / request.upload_id / "lora")
        config.experiment_name = (
            request.experiment_name or f"Evalora-{request.upload_id}"
        )
        config.logging_dir = str(
            self.repo_root
            / "experiments"
            / request.upload_id
            / config.experiment_name
            / "logs"
        )

        override_fields = [
            "model_name",
            "num_train_epochs",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]
        for field in override_fields:
            setattr(config, field, getattr(request, field))

        if config.validation_file:
            validation_path = self._resolve(config.validation_file)
            if not validation_path.exists() or validation_path.stat().st_size == 0:
                config.validation_file = None
                config.eval_strategy = "no"

        # Avoid multiprocessing pickling failures with Unsloth/Torch config objects.
        config.dataloader_num_workers = 0
        config.dataset_num_proc = 1
        return config

    def train(
        self, file: UploadFile | None, request: TrainingRequest
    ) -> dict[str, Any]:

        print(f"Received training request: {request}")
        if file is not None:
            # Read YAML config from uploaded file
            file_content = file.file.read().decode("utf-8")
            config_dict = yaml.safe_load(StringIO(file_content))
            config = TrainConfig(**config_dict)
            print(f"Loaded training config from uploaded file: {config}")
        else:
            print("No config file uploaded, using defaults and request overrides.")
            # Use default config path as fallback
            default_config_path = self.repo_root / "configs" / "train.yaml"
            if not default_config_path.exists():
                raise FileNotFoundError(
                    f"Default training config not found: {default_config_path}"
                )
            config = TrainConfig.from_yaml(default_config_path)

        config = self._apply_request_overrides(config, request)

        # Import lazily so FastAPI can start without loading training/GPU libraries.
        from src.training.trainer import run_training

        result = run_training(config)
        return {"status": "ok", **result}
