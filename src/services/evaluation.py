from src.evaluation.evaluators.evaluator import evaluate



class evaluation_service:
    def __init__(self):
        pass
    
    def evaluate(self, upload_id: str, use_cohen_kappa: bool = True) -> Dict[str, Any]:
        
        return evaluate(actual, predicted)