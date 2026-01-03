# backend/app.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

load_dotenv()

from backend.routes import router as api_router  # <-- FIX: package import

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Simpsonify (Clean)", version="1.0")

    app.include_router(api_router, prefix="/api")

    # Frontend direkt unter /
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
