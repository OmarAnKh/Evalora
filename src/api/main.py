from fastapi import FastAPI

from src.api.routes.dataset import router as dataset_router

app = FastAPI(title="AutoEval Dataset API")
app.include_router(dataset_router, prefix="/datasets", tags=["datasets"])
