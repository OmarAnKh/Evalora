from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.inference import EvaloraPredictor
from src.training.config import GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Evalora model inference on JSONL data."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-score", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=args.input_file, split="train")
    predictor = EvaloraPredictor(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length,
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens),
        min_score=args.min_score,
        max_score=args.max_score,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in dataset:
            prediction = predictor.predict(dict(sample))
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
