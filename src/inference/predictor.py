from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from src.data.formatter import format_sample_no_assistant
from src.evaluation.parsers import parse_prediction
from src.training.config import GenerationConfig
from src.training.modeling import load_base_model_and_tokenizer, prepare_for_inference


class EvaloraPredictor:
    """Generate rubric-based score/rationale predictions from a base or LoRA model."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        generation: GenerationConfig | None = None,
        min_score: float = 0.0,
        max_score: float | None = None,
    ) -> None:
        self.model, self.tokenizer = load_base_model_and_tokenizer(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, Path(adapter_path))
        prepare_for_inference(self.model)
        self.generation = generation or GenerationConfig()
        self.min_score = min_score
        self.max_score = max_score

    def _prompt_text(self, sample: dict[str, Any]) -> str:
        messages = sample.get("messages")
        if messages is None:
            messages = format_sample_no_assistant(sample)["messages"]
        else:
            messages = [
                message for message in messages if message.get("role") != "assistant"
            ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict(self, sample: dict[str, Any]) -> dict[str, Any]:
        prompt = self._prompt_text(sample)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation.max_new_tokens,
                temperature=self.generation.temperature,
                top_p=self.generation.top_p,
                do_sample=self.generation.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return parse_prediction(
            text, min_score=self.min_score, max_score=self.max_score
        )

    def predict_many(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict(sample) for sample in samples]
