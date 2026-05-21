import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from bert_score import BERTScorer
from scipy.stats import spearmanr
from typing import List, Dict, Any
import torch

def evaluate(actual: List[Dict[str, Any]], predicted: List[Dict[str, Any]], use_kappa: bool = True, num_bins: int = 5) -> Dict[str, Any]:
    """
    Evaluation for LLM-based rubric scoring systems.
    """
    # Extract values
    actual_score = np.array([float(x["score"]) for x in actual])
    predicted_score = np.array([float(x["score"]) for x in predicted])

    actual_rationale = [x["reasoning"] for x in actual]
    predicted_rationale = [x["reasoning"] for x in predicted]

    # Regression metrics
    rmse = mean_squared_error(actual_score, predicted_score) ** 0.5
    mae = mean_absolute_error(actual_score, predicted_score)
    spearman_corr, _ = spearmanr(actual_score, predicted_score)
    score_metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "spearman": round(float(spearman_corr), 4) if spearman_corr is not None else None,
    }

    # Cohen's Kappa (optional)
    if use_kappa:
        def to_bin(x):
            return min(num_bins - 1, max(0, int(x * num_bins)))
        actual_cls = [to_bin(x) for x in actual_score]
        predicted_cls = [to_bin(x) for x in predicted_score]
        kappa = cohen_kappa_score(actual_cls, predicted_cls)
        score_metrics["cohen_kappa"] = round(float(kappa), 4)

    # BERTScore (reasoning)
    scorer = BERTScorer(model_type="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu")
    P, R, F1 = scorer.score(predicted_rationale, actual_rationale)
    rationale_metrics = {
        "bert_precision": round(P.mean().item(), 4),
        "bert_recall": round(R.mean().item(), 4),
        "bert_f1": round(F1.mean().item(), 4),
    }

    return {
        "score": score_metrics,
        "rationale": rationale_metrics
    }
