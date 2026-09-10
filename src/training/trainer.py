from __future__ import annotations

import inspect
import json
import platform
import shutil
import tempfile
import time
from dataclasses import replace

import unsloth  # noqa: F401
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold
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
    seeds = config.training_seeds or [config.seed]
    if len(seeds) == 1:
        return _run_training_for_seed(replace(config, seed=seeds[0]))

    root_output = Path(config.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = replace(
            config,
            seed=seed,
            output_dir=str(root_output / "seeds" / f"seed-{seed}"),
            logging_dir=str(root_output / "seeds" / f"seed-{seed}" / "logs"),
            training_seeds=[seed],
        )
        seed_results.append(_run_training_for_seed(seed_config))

    first_seed_output = Path(seed_results[0]["output_dir"])
    for item in first_seed_output.iterdir():
        target = root_output / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    config_path = root_output / "training_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    return {
        "output_dir": str(root_output),
        "config_path": str(config_path),
        "training_seeds": seeds,
        "seed_runs": seed_results,
        "config": config.to_dict(),
        "metrics": {
            "seed_train_loss": [
                result["metrics"].get("train_loss") for result in seed_results
            ]
        },
    }


def _run_training_for_seed(config: TrainConfig) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

    if config.cross_validation_folds < 1:
        raise ValueError("cross_validation_folds must be at least 1.")
    if config.cross_validation_folds == 1:
        return _run_single_training(config)

    from datasets import load_dataset

    source_dataset = load_dataset("json", data_files=config.train_file, split="train")
    if config.cross_validation_folds > len(source_dataset):
        raise ValueError(
            "cross_validation_folds cannot exceed the number of training examples."
        )

    fold_root = Path(config.output_dir) / "cv"
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="evalora_cv_") as temp_dir:
        score_labels = source_dataset["score"]
        min_class_count = min(score_labels.count(label) for label in set(score_labels))
        if config.cross_validation_folds > min_class_count:
            raise ValueError(
                "cross_validation_folds cannot exceed the number of examples in the "
                f"rarest score class ({min_class_count})."
            )
        splitter = StratifiedKFold(
            n_splits=config.cross_validation_folds,
            shuffle=True,
            random_state=config.seed,
        )
        for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(range(len(source_dataset)), score_labels), start=1
        ):
            fold_dir = fold_root / f"fold-{fold_number}"
            fold_train_file = Path(temp_dir) / f"fold-{fold_number}-train.jsonl"
            fold_validation_file = Path(temp_dir) / f"fold-{fold_number}-validation.jsonl"
            source_dataset.select(train_indices.tolist()).to_json(
                str(fold_train_file), orient="records", lines=True
            )
            source_dataset.select(validation_indices.tolist()).to_json(
                str(fold_validation_file), orient="records", lines=True
            )
            fold_config = replace(
                config,
                cross_validation_folds=1,
                train_file=str(fold_train_file),
                validation_file=str(fold_validation_file),
                test_file=None,
                output_dir=str(fold_dir),
                logging_dir=str(fold_dir / "logs"),
            )
            fold_results.append(_run_single_training(fold_config))

    first_fold_dir = fold_root / "fold-1"
    for item in first_fold_dir.iterdir():
        target = Path(config.output_dir) / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    config_path = Path(config.output_dir) / "training_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    return {
        "output_dir": config.output_dir,
        "config_path": str(config_path),
        "cross_validation": {
            "n_splits": config.cross_validation_folds,
            "folds": fold_results,
        },
        "metrics": {
            "fold_train_loss": [
                result["metrics"].get("train_loss") for result in fold_results
            ]
        },
    }


def _run_single_training(config: TrainConfig) -> dict[str, Any]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging_dir).mkdir(parents=True, exist_ok=True)

    started_at = time.time()
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

    duration_seconds = round(time.time() - started_at, 2)
    hardware = _hardware_metadata()
    run_metadata = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "seed": config.seed,
        "configured_training_seeds": config.training_seeds,
        "cross_validation_folds": config.cross_validation_folds,
        "lora": {
            "rank": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
        },
        "training": {
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "per_device_eval_batch_size": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
        },
        "hardware": hardware,
        "training_time_seconds": duration_seconds,
    }
    metadata_path = Path(config.output_dir) / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    return {
        "output_dir": config.output_dir,
        "config_path": str(config_path),
        "metrics": metrics,
        "run_metadata_path": str(metadata_path),
        "config": config.to_dict(),
        "reproducibility": run_metadata,
    }


def _hardware_metadata() -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return metadata
