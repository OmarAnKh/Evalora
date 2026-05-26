from fastapi import APIRouter, File, HTTPException, UploadFile,Depends

from src.schemas.training import TrainingRequest, TrainingResponse
from src.services.training_service import TrainingService

router = APIRouter()
service = TrainingService()


@router.post("/train")
def train_model(
    request: TrainingRequest = Depends(TrainingRequest.as_form),
    file: UploadFile | None = File(None),
):
    """Start LoRA fine-tuning and save the adapter/model artifacts."""  

    try:
        return service.train(file, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
