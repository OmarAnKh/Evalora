from __future__ import annotations

from typing import Any

import numpy as np
import torch
from bert_score import BERTScorer
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error


def _scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([float(row["score"]) for row in rows], dtype=float)


def _reasoning(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("reasoning", "")) for row in rows]


def _discrete(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(int)


def evaluate(
    actual: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    use_kappa: bool = True,
    use_bertscore: bool = True,
) -> dict[str, Any]:
    """Evaluate score accuracy and rationale similarity for rubric-grading outputs."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted result lists must have the same length.")
    if not actual:
        raise ValueError("Cannot evaluate an empty result set.")

    actual_score = _scores(actual)
    predicted_score = _scores(predicted)
    actual_cls = _discrete(actual_score)
    predicted_cls = _discrete(predicted_score)

    score_metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(actual_cls, predicted_cls)), 4),
        "rmse": round(float(mean_squared_error(actual_score, predicted_score) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(actual_score, predicted_score)), 4),
    }

    spearman_corr, _ = spearmanr(actual_score, predicted_score)
    score_metrics["spearman"] = (
        round(float(spearman_corr), 4) if not np.isnan(spearman_corr) else None
    )

    if use_kappa:
        score_metrics["quadratic_weighted_kappa"] = round(
            float(cohen_kappa_score(actual_cls, predicted_cls, weights="quadratic")), 4
        )

    rationale_metrics: dict[str, Any] = {}
    if use_bertscore:
        scorer = BERTScorer(
            model_type="bert-base-uncased",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        precision, recall, f1 = scorer.score(_reasoning(predicted), _reasoning(actual))
        rationale_metrics = {
            "bert_precision": round(precision.mean().item(), 4),
            "bert_recall": round(recall.mean().item(), 4),
            "bert_f1": round(f1.mean().item(), 4),
        }

    parse_errors = sum(1 for row in predicted if row.get("parse_error"))
    return {
        "score": score_metrics,
        "rationale": rationale_metrics,
        "generation": {
            "parse_error_rate": round(parse_errors / len(predicted), 4),
            "parse_errors": parse_errors,
        },
    }
