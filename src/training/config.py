from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"


@dataclass(slots=True)
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str | bool = "unsloth"


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass(slots=True)
class TrainConfig:
    experiment_name: str = "autoeval-lora"
    seed: int = 3407
    model_name: str = DEFAULT_MODEL_NAME
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    train_file: str = ""
    validation_file: str | None = None
    test_file: str | None = None
    output_dir: str = "models/autoeval-lora"
    logging_dir: str = "experiments/autoeval-lora/logs"
    report_to: list[str] = field(default_factory=lambda: ["none"])
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    packing: bool = False
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    dataloader_num_workers: int = 0
    dataset_num_proc: int = 1
    remove_unused_columns: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        data = dict(payload)
        if "lora" in data and isinstance(data["lora"], dict):
            data["lora"] = LoraConfig(**data["lora"])
        if "generation" in data and isinstance(data["generation"], dict):
            data["generation"] = GenerationConfig(**data["generation"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
