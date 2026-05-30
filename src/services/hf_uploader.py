from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


class HuggingFaceUploader:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(
        self,
        hf_token: str | None = None,
        hf_username: str = "OmarAnKh",
    ) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN not found. " "Set it as environment variable.")

        self.api = HfApi(token=self.hf_token)
        self.hf_username = hf_username

    def load_comparison_metrics(self, upload_id: str) -> dict:
        """
        Load metrics from comparison_metrics.json
        """

        metrics_path = (
            self.PROJECT_ROOT
            / "experiments"
            / upload_id
            / "evaluation"
            / "test"
            / "metrics"
            / "comparison_metrics.json"
        )

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        finetuned = data["finetuned"]
        score_metrics = finetuned["metrics"]["score"]
        rationale_metrics = finetuned["metrics"].get("rationale", {})

        return {
            "model_name": finetuned["model_name"],
            "accuracy": score_metrics["accuracy"],
            "mae": score_metrics["mae"],
            "qwk": score_metrics["quadratic_weighted_kappa"],
            "bertscore": rationale_metrics.get("bert_f1"),
        }

    def create_readme(
        self,
        experiment_path: Path,
        metadata: dict,
    ) -> None:
        """
        Automatically generate README.md
        """

        readme = readme = f"""
# Evalora LoRA Model

## Experiment Information

- Upload ID: {metadata.get("upload_id")}
- Base Model: [{metadata.get("model_name")}](https://huggingface.co/{metadata.get("model_name")})
- Dataset: {metadata.get("dataset_name")}
- Created At: {metadata.get("timestamp")}

---

## Metrics

| Metric | Value |
|---|---|
| Accuracy | {metadata.get("accuracy")} |
| MAE | {metadata.get("mae")} |
| QWK | {metadata.get("qwk")} |
| BERTScore | {metadata.get("bertscore")} |

---

## Training Configuration

- Epochs: {metadata.get("epochs")}
- Learning Rate: {metadata.get("learning_rate")}
- LoRA Rank: {metadata.get("lora_r")}
- LoRA Alpha: {metadata.get("lora_alpha")}

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{metadata.get("model_name")}"
)

model = PeftModel.from_pretrained(
    base_model,
    "{metadata.get("repo_id")}"
)

tokenizer = AutoTokenizer.from_pretrained("{metadata.get("model_name")}")

# Replace the placeholders below with your actual input
user_content = (
    "Provide both a score and a rationale by evaluating the student's answer strictly within the mark scheme range, "
    "grading based on how well it meets the question's requirements by comparing the student answer to the reference answer.\\n"
    "Question: <YOUR_QUESTION>\\n"
    "Reference Answer: <YOUR_REFERENCE_ANSWER>\\n"
    "Student Answer: <YOUR_STUDENT_ANSWER>\\n"
    "Mark Scheme: <YOUR_MARK_SCHEME>"  # e.g. {{'1': 'Mentions X', '2': 'Explains Y'}}
)

messages = [
    {{
        "role": "system",
        "content": "You are a grading assistant. Evaluate student answers based on the mark scheme. Respond only in JSON format with keys 'score' (int) and 'rationale' (string)."
    }},
    {{
        "role": "user",
        "content": user_content
    }},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.2,
    top_k=5,
    do_sample=False,
)[0]

new_generated_ids = generated_ids[inputs["input_ids"].shape[1]:]
generated_text = tokenizer.decode(new_generated_ids, skip_special_tokens=True)
print(generated_text)
# Output example: {{"score": 4, "rationale": "The student correctly identified..."}}
```
"""

        readme_path = experiment_path / "README.md"

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme.strip())

    def save_metadata(
        self,
        experiment_path: Path,
        metadata: dict,
    ) -> None:
        """
        Save metadata json
        """

        metadata_path = experiment_path / "experiment_metadata.json"

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_adapter_config(self, upload_id: str) -> dict:
        """
        Load LoRA config from adapter_config.json and trainer_state.json
        """

        adapter_config_path = (
            self.PROJECT_ROOT / "models" / upload_id / "lora" / "adapter_config.json"
        )

        trainer_state_path = (
            self.PROJECT_ROOT / "models" / upload_id / "lora" / "trainer_state.json"
        )

        if not adapter_config_path.exists():
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        result = {
            "lora_r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "epochs": None,
            "learning_rate": None,
        }

        # trainer_state.json has epochs and learning rate
        if trainer_state_path.exists():
            with trainer_state_path.open("r", encoding="utf-8") as f:
                trainer_state = json.load(f)

            result["epochs"] = trainer_state.get("num_train_epochs")
            result["learning_rate"] = trainer_state.get(
                "best_model_checkpoint"
            ) and trainer_state.get("log_history", [{}])[-1].get("learning_rate")

        return result

    def upload_experiment(
        self,
        upload_id: str,
        dataset_name: str | None = None,
        private: bool = False,
    ) -> str:
        """
        Upload trained LoRA experiment to HuggingFace Hub
        """

        experiment_path = self.PROJECT_ROOT / "models" / upload_id / "lora"

        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

        comparison_metrics = self.load_comparison_metrics(upload_id)
        adapter_config = self.load_adapter_config(upload_id)

        repo_name = f"Evalora-{upload_id}"
        repo_id = f"{self.hf_username}/{repo_name}"

        metadata = {
            "upload_id": upload_id,
            "dataset_name": dataset_name or "N/A",
            "model_name": comparison_metrics["model_name"],
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": comparison_metrics["accuracy"],
            "mae": comparison_metrics["mae"],
            "qwk": comparison_metrics["qwk"],
            "bertscore": comparison_metrics["bertscore"],
            "epochs": adapter_config["epochs"],
            "learning_rate": adapter_config["learning_rate"],
            "lora_r": adapter_config["lora_r"],
            "lora_alpha": adapter_config["lora_alpha"],
            "repo_id": repo_id,
        }

        self.save_metadata(
            experiment_path=experiment_path,
            metadata=metadata,
        )

        self.create_readme(
            experiment_path=experiment_path,
            metadata=metadata,
        )

        create_repo(
            repo_id=repo_id,
            token=self.hf_token,
            repo_type="model",
            exist_ok=True,
            private=private,  # ← private/public flag
        )

        upload_folder(
            repo_id=repo_id,
            folder_path=str(experiment_path),
            token=self.hf_token,
            repo_type="model",
            ignore_patterns=[
                "checkpoint-*",
                "*.pth",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ],
        )

        print(f"Uploaded successfully to: {repo_id}")

        return repo_id
