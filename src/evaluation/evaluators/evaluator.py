from __future__ import annotations

from typing import Any

import numpy as np
import torch
from bert_score import BERTScorer
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


def _scores(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([float(row["score"]) for row in rows], dtype=float)


def _reasoning(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("reasoning", "")) for row in rows]


def _discrete(values: np.ndarray) -> np.ndarray:
    return np.rint(values).astype(int)


def _classification_metrics(actual_cls: np.ndarray, predicted_cls: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(actual_cls.tolist()) | set(predicted_cls.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual_cls,
        predicted_cls,
        labels=labels,
        zero_division=0,
    )
    return {
        "per_class": {
            str(label): {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(actual_cls, predicted_cls, labels=labels).tolist(),
        },
    }


def _bootstrap_uncertainty(
    actual_score: np.ndarray,
    predicted_score: np.ndarray,
    actual_cls: np.ndarray,
    predicted_cls: np.ndarray,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate sampling uncertainty by resampling paired predictions."""
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"accuracy": [], "mae": [], "rmse": []}
    for _ in range(samples):
        indices = rng.integers(0, len(actual_score), len(actual_score))
        actual = actual_score[indices]
        predicted = predicted_score[indices]
        values["accuracy"].append(float(accuracy_score(actual_cls[indices], predicted_cls[indices])))
        values["mae"].append(float(mean_absolute_error(actual, predicted)))
        values["rmse"].append(float(mean_squared_error(actual, predicted) ** 0.5))

    return {
        metric: {
            "mean": round(float(np.mean(metric_values)), 4),
            "std": round(float(np.std(metric_values, ddof=1)), 4),
            "ci95": [
                round(float(np.percentile(metric_values, 2.5)), 4),
                round(float(np.percentile(metric_values, 97.5)), 4),
            ],
        }
        for metric, metric_values in values.items()
    }


def evaluate_human_ratings(
    human_ratings: list[list[int | float]],
) -> dict[str, Any]:
    """Summarize blinded human scores and pairwise inter-rater agreement."""
    if not human_ratings or any(len(ratings) < 2 for ratings in human_ratings):
        raise ValueError("At least two human ratings are required for every example.")

    ratings = np.asarray(human_ratings, dtype=float)
    rounded = _discrete(ratings.reshape(-1))
    human_mean = ratings.mean(axis=1)
    human_consensus = _discrete(human_mean)
    pairwise_kappa: list[float] = []
    for first in range(ratings.shape[1]):
        for second in range(first + 1, ratings.shape[1]):
            pairwise_kappa.append(
                float(cohen_kappa_score(_discrete(ratings[:, first]), _discrete(ratings[:, second])))
            )

    return {
        "num_examples": int(ratings.shape[0]),
        "num_raters": int(ratings.shape[1]),
        "mean_score": round(float(human_mean.mean()), 4),
        "std_score": round(float(human_mean.std(ddof=1)), 4),
        "consensus_scores": human_consensus.tolist(),
        "inter_rater_agreement": {
            "cohen_kappa_mean": round(float(np.mean(pairwise_kappa)), 4),
            "cohen_kappa_by_pair": [round(value, 4) for value in pairwise_kappa],
        },
        "ratings": ratings.tolist(),
        "score_distribution": {str(label): int(count) for label, count in zip(*np.unique(rounded, return_counts=True))},
    }


def cross_validate_predictions(
    actual: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate metric stability across shuffled K-fold evaluation splits.

    Predictions must already exist for every example. This evaluates fold stability;
    it does not retrain a model for each fold.
    """
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted result lists must have the same length.")
    if n_splits < 2 or n_splits > len(actual):
        raise ValueError("n_splits must be at least 2 and no greater than the number of examples.")

    actual_score = _scores(actual)
    predicted_score = _scores(predicted)
    actual_cls = _discrete(actual_score)
    predicted_cls = _discrete(predicted_score)
    folds: list[dict[str, Any]] = []
    _, class_counts = np.unique(actual_cls, return_counts=True)
    min_class_count = int(class_counts.min())
    if n_splits > min_class_count:
        raise ValueError(
            "n_splits cannot exceed the number of examples in the rarest score class "
            f"({min_class_count})."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_number, (_, test_indices) in enumerate(
        splitter.split(actual_score, actual_cls), start=1
    ):
        fold_actual = actual_score[test_indices]
        fold_predicted = predicted_score[test_indices]
        folds.append(
            {
                "fold": fold_number,
                "num_examples": int(len(test_indices)),
                "accuracy": round(
                    float(accuracy_score(actual_cls[test_indices], predicted_cls[test_indices])), 4
                ),
                "mae": round(float(mean_absolute_error(fold_actual, fold_predicted)), 4),
                "rmse": round(float(mean_squared_error(fold_actual, fold_predicted) ** 0.5), 4),
            }
        )

    aggregate: dict[str, dict[str, float | list[float]]] = {}
    for metric in ("accuracy", "mae", "rmse"):
        values = np.asarray([fold[metric] for fold in folds], dtype=float)
        aggregate[metric] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=1)), 4),
            "ci95": [
                round(float(np.percentile(values, 2.5)), 4),
                round(float(np.percentile(values, 97.5)), 4),
            ],
        }
    return {"n_splits": n_splits, "seed": seed, "folds": folds, "aggregate": aggregate}


def evaluate(
    actual: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    use_kappa: bool = True,
    use_bertscore: bool = True,
    human_ratings: list[list[int | float]] | None = None,
    baseline_predictions: list[dict[str, Any]] | None = None,
    cross_validation_folds: int | None = 5,
    _include_baselines: bool = True,
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
    score_metrics.update(_classification_metrics(actual_cls, predicted_cls))
    score_metrics["uncertainty"] = _bootstrap_uncertainty(
        actual_score, predicted_score, actual_cls, predicted_cls
    )

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
    result: dict[str, Any] = {
        "score": score_metrics,
        "rationale": rationale_metrics,
        "generation": {
            "parse_error_rate": round(parse_errors / len(predicted), 4),
            "parse_errors": parse_errors,
        },
    }

    if _include_baselines:
        majority_score = int(np.bincount(actual_cls).argmax())
        majority_predictions = [{"score": majority_score, "reasoning": ""} for _ in actual]
        result["baselines"] = {
            "majority_class": evaluate(
                actual,
                majority_predictions,
                use_kappa=use_kappa,
                use_bertscore=False,
                _include_baselines=False,
                cross_validation_folds=None,
            )
        }
        if baseline_predictions is not None:
            if len(baseline_predictions) != len(actual):
                raise ValueError("Baseline predictions must have the same length as actual results.")
            result["baselines"]["external"] = evaluate(
                actual,
                baseline_predictions,
                use_kappa=use_kappa,
                use_bertscore=False,
                _include_baselines=False,
                cross_validation_folds=None,
            )
    if human_ratings is not None:
        human_result = evaluate_human_ratings(human_ratings)
        if len(human_ratings) != len(predicted):
            raise ValueError("Human ratings must have the same length as predictions.")
        consensus = np.asarray(human_result["consensus_scores"], dtype=int)
        result["human_evaluation"] = {
            **human_result,
            "model_vs_human": {
                "accuracy": round(float(accuracy_score(consensus, predicted_cls)), 4),
                "mae": round(float(mean_absolute_error(consensus, predicted_cls)), 4),
            },
        }
    if cross_validation_folds is not None:
        result["cross_validation"] = cross_validate_predictions(
            actual,
            predicted,
            n_splits=cross_validation_folds,
        )
    return result
