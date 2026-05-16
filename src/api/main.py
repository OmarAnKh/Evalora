from fastapi import FastAPI

from src.api.routes.dataset import router as dataset_router
from src.api.routes.pipeline import router as pipeline_router

app = FastAPI(title="AutoEval Dataset API")
app.include_router(dataset_router, prefix="/datasets", tags=["datasets"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
