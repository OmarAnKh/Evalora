from pathlib import Path
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.dataset import router as dataset_router
from src.api.routes.pipeline import router as pipeline_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.training import router as training_router
from src.api.routes.upload_model import router as upload_router


def _clear_data_directories() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

    for name in ("formatted", "splits", "raw", "tokenized"):
        target_dir = data_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _clear_data_directories()  # startup
    yield
    # shutdown logic here if needed


app = FastAPI(
    title="Evalora Dataset API",
    lifespan=lifespan,
)

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
