from fastapi import APIRouter, HTTPException, Depends

from src.schemas.upload_model import UploadModelRequest, UploadModelResponse
from src.services.hf_uploader import HuggingFaceUploader


router = APIRouter()


@router.post("/upload", response_model=UploadModelResponse)
def upload_experiment(
    request: UploadModelRequest = Depends(UploadModelRequest.as_form),
):
    """Upload a trained LoRA experiment to HuggingFace Hub."""

    try:
        uploader = HuggingFaceUploader(
            hf_token=request.hf_token,
            hf_username=request.hf_username,
        )
        repo_id = uploader.upload_experiment(
            upload_id=request.upload_id,
            dataset_name=request.dataset_name,
            private=request.private,
        )
        repo_url = f"https://huggingface.co/{repo_id}"
        return UploadModelResponse(
            upload_id=request.upload_id,
            repo_id=repo_id,
            repo_url=repo_url,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc