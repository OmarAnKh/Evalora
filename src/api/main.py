from pathlib import Path
import shutil

from fastapi import FastAPI

from src.api.routes.dataset import router as dataset_router
from src.api.routes.pipeline import router as pipeline_router


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clear_data_files() -> None:
    data_root = _repo_root() / "data"
    target_dirs = ("raw", "processed", "splits", "formatted")

    for dir_name in target_dirs:
        directory = data_root / dir_name
        directory.mkdir(parents=True, exist_ok=True)

        for path in directory.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


app = FastAPI(title="AutoEval Dataset API")


@app.on_event("startup")
def clear_data_directories_on_startup() -> None:
    _clear_data_files()


app.include_router(dataset_router, prefix="/datasets", tags=["datasets"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
