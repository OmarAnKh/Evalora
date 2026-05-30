from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.dataset import router as dataset_router
from src.api.routes.pipeline import router as pipeline_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.training import router as training_router
from src.api.routes.upload_model import router as upload_router

app = FastAPI(title="AutoEval Dataset API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(dataset_router, prefix="/datasets", tags=["datasets"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
app.include_router(training_router, prefix="/training", tags=["training"])
app.include_router(upload_router, prefix="/upload", tags=["upload"])
