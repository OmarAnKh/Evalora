from __future__ import annotations

import argparse
import json

from src.training.config import TrainConfig
from src.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Evalora with LoRA.")
    parser.add_argument(
        "--config", required=True, help="Path to a YAML training config."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(TrainConfig.from_yaml(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
