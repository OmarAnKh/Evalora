from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.evaluation.evaluators.evaluator import evaluate
from src.training.config import DEFAULT_MODEL_NAME, GenerationConfig

DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LOAD_IN_4BIT = True


class EvaluationService:
    """Run baseline/fine-tuned inference and compute held-out metrics."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    def _split_path(self, upload_id: str, split: str) -> Path:
        return (
            self.repo_root
            / "data"
            / "splits"
            / upload_id
            / f"{upload_id}_{split}.jsonl"
        )

    def _adapter_path(self, upload_id: str) -> Path:
        path = self.repo_root / "models" / upload_id / "lora"
        if not path.exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found for upload_id '{upload_id}'. "
                f"Expected: {path}"
            )
        return path

    def _build_manual_inference_messages(
        self,
        reference_answer: str,
        answer: str,
        rubric: list[dict[str, Any]],
        task: str | None = None,
    ) -> list[dict[str, str]]:
        task_text = (
            task.strip()
            if task and task.strip()
            else "Evaluate the student's answer using the reference answer and rubric."
        )
        user_content = (
            "Task:\n"
            f"{task_text}\n\n"
            "Reference Answer:\n"
            f"{reference_answer.strip()}\n\n"
            "Student Answer:\n"
            f"{answer.strip()}\n\n"
            "Rubric:\n"
            f"{json.dumps(rubric, ensure_ascii=False, indent=2)}\n\n"
            "Return only valid JSON with keys score and reasoning."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an automated evaluation model. "
                    "Score the answer strictly using the rubric and return JSON only."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _load_expected(self, dataset) -> list[dict[str, Any]]:
        expected: list[dict[str, Any]] = []
        for row in dataset:
            if row.get("score") is None or row.get("reasoning") is None:
                expected_payload = row.get("expected") or {}
                expected.append(
                    {
                        "score": expected_payload.get("score"),
                        "reasoning": expected_payload.get("reasoning", ""),
                    }
                )
            else:
                expected.append({"score": row["score"], "reasoning": row["reasoning"]})
        return expected

    def evaluate_predictions(
        self,
        expected: list[dict[str, Any]],
        predicted: list[dict[str, Any]],
        use_kappa: bool = True,
        use_bertscore: bool = True,
    ) -> dict[str, Any]:
        return evaluate(
            expected, predicted, use_kappa=use_kappa, use_bertscore=use_bertscore
        )

    def evaluate_model_on_split(
        self,
        upload_id: str,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: str | None = None,
        split: str = "test",
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
        generation: GenerationConfig | None = None,
        use_kappa: bool = True,
        use_bertscore: bool = True,
        output_file: str | None = None,
    ) -> dict[str, Any]:
        split_path = self._split_path(upload_id, split)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        from src.inference import EvaloraPredictor

        dataset = load_dataset("json", data_files=str(split_path), split="train")
        predictor = EvaloraPredictor(
            model_name=model_name,
            adapter_path=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
        )
        predictions = predictor.predict_many([dict(row) for row in dataset])
        expected = self._load_expected(dataset)
        metrics = self.evaluate_predictions(
            expected,
            predictions,
            use_kappa=use_kappa,
            use_bertscore=use_bertscore,
        )

        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for prediction in predictions:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        return {
            "upload_id": upload_id,
            "split": split,
            "model_name": model_name,
            "model_variant": "finetuned" if adapter_path else "baseline",
            "num_examples": len(predictions),
            "metrics": metrics,
        }

    def evaluate_finetuned_on_split(
        self,
        upload_id: str,
        model_name: str = DEFAULT_MODEL_NAME,
        split: str = "test",
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
        generation: GenerationConfig | None = None,
        use_kappa: bool = True,
        use_bertscore: bool = True,
        output_file: str | None = None,
    ) -> dict[str, Any]:
        return self.evaluate_model_on_split(
            upload_id=upload_id,
            model_name=model_name,
            adapter_path=str(self._adapter_path(upload_id)),
            split=split,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
            use_kappa=use_kappa,
            use_bertscore=use_bertscore,
            output_file=output_file,
        )

    def compare_baseline_and_finetuned(
        self,
        upload_id: str,
        model_name: str = DEFAULT_MODEL_NAME,
        split: str = "test",
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
        generation: GenerationConfig | None = None,
        use_kappa: bool = True,
        use_bertscore: bool = True,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        return self.compare_baseline_and_adapter(
            upload_id=upload_id,
            adapter_path=str(self._adapter_path(upload_id)),
            model_name=model_name,
            split=split,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
            use_kappa=use_kappa,
            use_bertscore=use_bertscore,
            output_dir=output_dir,
        )

    def compare_baseline_and_adapter(
        self,
        upload_id: str,
        adapter_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        split: str = "test",
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
        generation: GenerationConfig | None = None,
        use_kappa: bool = True,
        use_bertscore: bool = True,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate the base model and a LoRA adapter on the same held-out split."""
        baseline_output = None
        adapter_output = None
        if output_dir:
            base_path = Path(output_dir) / "predictions"
            baseline_output = str(base_path / "baseline_predictions.jsonl")
            adapter_output = str(base_path / "finetuned_predictions.jsonl")

        baseline = self.evaluate_model_on_split(
            upload_id=upload_id,
            model_name=model_name,
            adapter_path=None,
            split=split,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
            use_kappa=use_kappa,
            use_bertscore=use_bertscore,
            output_file=baseline_output,
        )

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        finetuned = self.evaluate_model_on_split(
            upload_id=upload_id,
            model_name=model_name,
            adapter_path=adapter_path,
            split=split,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
            use_kappa=use_kappa,
            use_bertscore=use_bertscore,
            output_file=adapter_output,
        )

        result = {
            "upload_id": upload_id,
            "split": split,
            "baseline": baseline,
            "finetuned": finetuned,
        }
        if output_dir:
            path = Path(output_dir) / "metrics" / "comparison_metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
        return result

    def predict_with_finetuned_model(
        self,
        upload_id: str,
        reference_answer: str,
        answer: str,
        rubric: list[dict[str, Any]],
        task: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
        generation: GenerationConfig | None = None,
    ) -> dict[str, Any]:
        from src.inference import EvaloraPredictor

        predictor = EvaloraPredictor(
            model_name=model_name,
            adapter_path=str(self._adapter_path(upload_id)),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            generation=generation,
        )
        messages = self._build_manual_inference_messages(
            reference_answer=reference_answer,
            answer=answer,
            rubric=rubric,
            task=task,
        )
        prediction = predictor.predict({"messages": messages})
        return {
            "upload_id": upload_id,
            "score": prediction.get("score"),
            "reasoning": prediction.get("reasoning", ""),
        }


# Backward-compatible alias for the existing route import.
evaluation_service = EvaluationService
