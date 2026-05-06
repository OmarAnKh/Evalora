# AutoEval

## Project Description
AutoEval is a domain-agnostic, instruction-tuned evaluation system that turns rubric-based scoring into a reusable ML pipeline. Users upload a dataset (CSV or JSON) with six fields—`task`, `reference`, `submission`, `rubric`, `score`, and `rationale`—and AutoEval fine-tunes a Mistral-7B-Instruct model using LoRA so it can act as a scoring evaluator. The trained evaluator outputs both a numeric score (0–4) and a natural language rationale.

Manual rubric scoring is expensive, slow, and highly domain-specific. AutoEval addresses this by instruction-tuning a general-purpose LLM to follow rubric constraints and generate consistent scores with explanations. Rationales provide transparency and make the model useful for audit and review.

## Key Features
- Domain-agnostic evaluation across tasks and industries
- Automatic dataset validation for the 6-field schema
- Instruction-tuned LLM scoring with rubric conditioning
- Combined score + rationale output
- LoRA-based efficient fine-tuning
- End-to-end pipeline from dataset to trained evaluator

## Project Structure
```
AUTOEVAL/
├─ data/                   # Raw, processed, and split datasets
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ src/
│  ├─ data/                 # Loading, formatting, and splitting utilities
│  ├─ training/             # train.py, LoRA setup, training configs
│  ├─ evaluation/           # Metrics and evaluation scripts
│  ├─ inference/            # Prediction and scoring logic
│  └─ utils/                # Shared helpers (IO, logging, validation)
├─ configs/                 # YAML configs for training and evaluation
├─ models/                  # Base models, LoRA adapters, merged checkpoints
├─ experiments/             # Logs, runs, and artifacts
├─ prompts/                 # Instruction templates and prompt assets
├─ pipeline/                # End-to-end pipeline script(s)
└─ scripts/                 # CLI entry points and utility scripts
```

## How It Works (Pipeline)
1. Dataset upload in CSV or JSON format.
2. Validation against the required 6-field schema.
3. Prompt formatting in `[INST]` style for instruction tuning.
4. Train/validation/test split generation.
5. Baseline evaluation with the base model.
6. Fine-tuning with LoRA (Unsloth-backed).
7. Post-training evaluation on held-out data.
8. Model export (LoRA adapters and optional merged model).

## Model Details
- Base model: Mistral-7B-Instruct
- Fine-tuning method: LoRA (PEFT)
- Training framework: Hugging Face Transformers + TRL + Unsloth
- Metrics: Accuracy, MAE, Quadratic Weighted Kappa (QWK)

## Example Input/Output
**Input (one row)**
```json
{
	"task": "Grade the response to the customer email.",
	"reference": "We apologize for the delay and offer a replacement within 3 business days.",
	"submission": "Sorry for the wait. We can send a replacement this week.",
	"rubric": "Score 4: apologizes, gives clear replacement timeline. Score 2: apology but vague timeline. Score 0: no apology or resolution.",
	"score": 3,
	"rationale": "Includes an apology and a replacement, but the timeline is less specific than the rubric's 3 business days."
}
```

**Model Output**
```json
{
	"score": 3,
	"rationale": "The response apologizes and offers a replacement, but the timeline is vague compared to the rubric's 3 business days."
}
```

## Setup & Installation
**Python**: 3.10+ recommended

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
**Training**
```bash
python src/training/train.py --config configs/train.yaml
```

**Evaluation**
```bash
python src/evaluation/evaluator.py
```

**Inference**
```bash
python src/inference/predict.py
```

## Team & Contributions
This is a two-person project:
- Model & Training: fine-tuning, metrics, and evaluation design
- Pipeline & System: data validation, orchestration, and infrastructure

## Future Improvements
- Automated dataset generation and augmentation
- Multi-domain evaluator ensembles
- API deployment for real-time scoring
- Larger base models and domain-specific adapters
