# Evalora

## Project Description
Evalora is a domain-agnostic, instruction-tuned evaluation system that turns rubric-based scoring into a reusable ML pipeline. Users upload a JSONL dataset with six fields: `task`, `reference_answer`, `answer`, `rubric` (a list of weighted criteria), `score`, and `reasoning`. Evalora fine-tunes a Mistral-7B-Instruct model using LoRA so it can act as a scoring evaluator. The trained evaluator outputs both a numeric score and a natural language rationale.

Manual rubric scoring is expensive, slow, and highly domain-specific. Evalora addresses this by instruction-tuning a general-purpose LLM to follow rubric constraints and generate consistent scores with explanations. Rationales provide transparency and make the model useful for audit and review.

## Key Features
- Domain-agnostic evaluation across tasks and industries
- Automatic dataset validation for the 6-field schema
- JSONL upload API with line-level validation errors
- Instruction-tuned LLM scoring with rubric conditioning
- Combined score + rationale output
- LoRA-based efficient fine-tuning
- End-to-end pipeline from dataset to trained evaluator

## Project Structure
```
Evalora/
├─ data/                   # Raw, processed, and split datasets
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ src/
│  ├─ data/                 # Loading, formatting, and splitting utilities
│  ├─ training/             # train.py, LoRA setup, training configs
│  ├─ evaluation/           # Metrics and evaluation scripts
│  ├─ inference/            # Prediction and scoring logic
│  ├─ api/                  # FastAPI entrypoint and routes
│  ├─ services/             # Service layer (validation + persistence)
│  ├─ schemas/              # Pydantic DTOs
│  └─ utils/                # Shared helpers (IO, logging, validation)
├─ configs/                 # YAML configs for training and evaluation
├─ models/                  # Base models, LoRA adapters, merged checkpoints
├─ experiments/             # Logs, runs, and artifacts
├─ prompts/                 # Instruction templates and prompt assets
├─ pipeline/                # End-to-end pipeline script(s)
└─ scripts/                 # CLI entry points and utility scripts
```

## How It Works (Pipeline)
1. Dataset upload as JSONL via API.
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
**Input (one JSONL record)**
```json
{
	"task": "Grade the response to the customer email.",
	"reference_answer": "We apologize for the delay and offer a replacement within 3 business days.",
	"answer": "Sorry for the wait. We can send a replacement this week.",
	"rubric": [
		{"criterion": "Apology", "description": "Must apologize clearly", "weight": 0.5},
		{"criterion": "Resolution", "description": "Offer replacement with a timeline", "weight": 0.5}
	],
	"score": 3,
	"reasoning": "Includes an apology and a replacement, but the timeline is less specific than 3 business days."
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
uv venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
uv sync
```

## Usage
**API (dataset upload)**
```bash
uvicorn src.api.main:app --reload
```

## Frontend
The UI lives in the interface/ folder and talks to the FastAPI backend.

**Install dependencies**
```bash
cd interface
npm install
```

**Run the dev server**
```bash
npm run dev
```

**Connect to the API**
- Keep the backend running (for example: http://localhost:8004).
- Set the API URL in the header input, or export VITE_API_BASE_URL before starting the dev server.

**Upload**
```
POST /datasets/upload
Content-Type: multipart/form-data
file: <dataset.jsonl>
```

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
- Partial-save mode for invalid JSONL uploads
- API deployment for real-time scoring
- Larger base models and domain-specific adapters
