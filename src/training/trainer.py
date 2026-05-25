from __future__ import annotations

import inspect
import json

import unsloth  # noqa: F401
from pathlib import Path
from typing import Any

from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:  # Older TRL versions use TrainingArguments with SFTTrainer kwargs.
    SFTConfig = None

from src.training.config import TrainConfig
from src.training.data import load_sft_splits, prepare_sft_datasets
from src.training.modeling import attach_lora_for_training, load_base_model_and_tokenizer


def _resolve_report_to(report_to: list[str]) -> list[str]:
    normalized = [target.lower() for target in report_to]
    if "tensorboard" not in normalized:
        return report_to

    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return [target for target in report_to if target.lower() != "tensorboard"] or ["none"]
    return report_to


def _common_training_kwargs(config: TrainConfig) -> dict[str, Any]:
    return {
        "output_dir": config.output_dir,
        "logging_dir": config.logging_dir,
        "report_to": _resolve_report_to(config.report_to),
        "seed": config.seed,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "eval_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        "max_grad_norm": config.max_grad_norm,
        "dataloader_num_workers": config.dataloader_num_workers,
        "remove_unused_columns": config.remove_unused_columns,
        "load_best_model_at_end": bool(config.validation_file),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


def build_training_arguments(config: TrainConfig):
    """Translate project config into the installed TRL/HF training argument class."""
    kwargs = _common_training_kwargs(config)
    if SFTConfig is None:
        return TrainingArguments(**kwargs)

    kwargs.update(
        {
            "dataset_text_field": "text",
            "max_length": config.max_seq_length,
            "packing": config.packing,
            "dataset_num_proc": config.dataset_num_proc,
        }
    )
    return SFTConfig(**kwargs)


def _build_sft_trainer_kwargs(config: TrainConfig, tokenizer, prepared, args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": None,
        "args": args,
        "train_dataset": prepared["train"],
        "eval_dataset": prepared.get("validation"),
    }

    signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer

    if SFTConfig is None:
        legacy_kwargs = {
            "dataset_text_field": "text",
            "max_seq_length": config.max_seq_length,
            "packing": config.packing,
        }
        if "dataset_num_proc" in signature:
            legacy_kwargs["dataset_num_proc"] = config.dataset_num_proc
        kwargs.update(legacy_kwargs)
    return kwargs


def build_trainer(config: TrainConfig) -> SFTTrainer:
    set_seed(config.seed)
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )
    model = attach_lora_for_training(model, config.lora, seed=config.seed)

    splits = load_sft_splits(
        train_file=config.train_file,
        validation_file=config.validation_file,
        test_file=config.test_file,
    )
    prepared = prepare_sft_datasets(splits, tokenizer)
    args = build_training_arguments(config)
    trainer_kwargs = _build_sft_trainer_kwargs(config, tokenizer, prepared, args)
    trainer_kwargs["model"] = model
    return SFTTrainer(**trainer_kwargs)


def run_training(config: TrainConfig) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(config)
    train_result = trainer.train()
    trainer.save_model(config.output_dir)

    tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(config.output_dir)

    config_path = Path(config.output_dir) / "training_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if trainer.eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        metrics.update(eval_metrics)

    return {
        "output_dir": config.output_dir,
        "config_path": str(config_path),
        "metrics": metrics,
    }
