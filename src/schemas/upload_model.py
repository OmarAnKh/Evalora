from __future__ import annotations

from pydantic import BaseModel

from fastapi import Form

class UploadModelRequest(BaseModel):
    upload_id: str
    hf_token: str
    hf_username: str
    dataset_name: str | None = None
    private: bool = False
    
    @classmethod
    def as_form(
        cls,
        upload_id: str = Form(...),
        hf_token: str = Form(...),
        hf_username: str = Form(...),
        dataset_name: str | None = Form(None),
        private: bool = Form(False),
    ):
        return cls(
            upload_id=upload_id,
            hf_token=hf_token,
            hf_username=hf_username,
            dataset_name=dataset_name,
            private=private,
        )
    


class UploadModelResponse(BaseModel):
    upload_id: str
    repo_id: str
    repo_url: str