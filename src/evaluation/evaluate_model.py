from __future__ import annotations

import argparse
import json

from src.services.evaluation import EvaluationService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and LoRA Evalora models."
    )
    parser.add_argument("--upload-id", required=True)
    parser.add_argument(
        "--model-name", default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    )
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bertscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = EvaluationService()
    if args.adapter_path:
        result = service.compare_baseline_and_adapter(
            upload_id=args.upload_id,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            split=args.split,
            use_bertscore=args.bertscore,
            output_dir=args.output_dir,
        )
    else:
        result = service.evaluate_model_on_split(
            upload_id=args.upload_id,
            model_name=args.model_name,
            split=args.split,
            use_bertscore=args.bertscore,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
