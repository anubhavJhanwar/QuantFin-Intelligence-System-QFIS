"""
QFIS FastAPI Application Entry Point.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.routes import query, voice, evaluate, logs
from app.config.settings import settings

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(LOG_DIR / "api.log", rotation="10 MB", level=settings.log_level)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QFIS — Financial Question Answering System",
    description="QLoRA fine-tuned Phi-2 + RAG for financial QA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(query.router)
app.include_router(voice.router)
app.include_router(evaluate.router)
app.include_router(logs.router)


@app.get("/")
async def root():
    return {
        "name": "QFIS API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health():
    from app.services.inference import get_status
    return {"status": "ok", "model": get_status()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=False,
        log_level="info",
    )
