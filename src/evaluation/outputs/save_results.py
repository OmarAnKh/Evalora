import os
import json
from typing import Dict, Any

def save_evaluation_results(experiment_id: str, baseline_results: Dict[str, Any], finetuned_results: Dict[str, Any]):
    """
    Save baseline and finetuned evaluation results in the experiments folder under a unique experiment_id.
    """
    base_path = os.path.join("experiments", experiment_id)
    os.makedirs(base_path, exist_ok=True)
    baseline_path = os.path.join(base_path, "baseline_eval.json")
    finetuned_path = os.path.join(base_path, "finetuned_eval.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)
    with open(finetuned_path, "w", encoding="utf-8") as f:
        json.dump(finetuned_results, f, indent=2)
