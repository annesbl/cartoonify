from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from routes import router as api_router


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Simpsonify (Clean)", version="1.0")

    app.include_router(api_router, prefix="/api")

    # Frontend direkt unter /
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
