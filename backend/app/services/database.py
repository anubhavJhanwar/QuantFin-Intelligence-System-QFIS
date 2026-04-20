"""
MongoDB service for QFIS.
Gracefully falls back to in-memory + JSON file storage if MongoDB is unavailable.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# ── Try MongoDB, fall back to file-based storage ──────────────────────────────
_mongo_available = False
_client = None

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import DESCENDING
    from app.config.settings import settings

    _test_client = AsyncIOMotorClient(
        settings.mongo_uri, serverSelectionTimeoutMS=500
    )
    _mongo_available = True
    logger.info("MongoDB client initialized")
except Exception as e:
    logger.warning(f"MongoDB unavailable — using file-based fallback: {e}")

# ── File-based fallback storage ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "logs").exists():
    ROOT = ROOT.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_FILE = LOG_DIR / "queries.json"
EVALS_FILE = LOG_DIR / "evaluations.json"


def _read_json(path: Path) -> list:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _append_json(path: Path, record: dict):
    data = _read_json(path)
    data.insert(0, record)
    data = data[:500]  # keep last 500
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── Public API (same interface regardless of backend) ─────────────────────────

def get_client():
    global _client
    if _client is None and _mongo_available:
        from motor.motor_asyncio import AsyncIOMotorClient
        from app.config.settings import settings
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=500,
            connectTimeoutMS=500,
            socketTimeoutMS=2000,
        )
    return _client


def get_db():
    c = get_client()
    if c is None:
        return None
    from app.config.settings import settings
    return c[settings.mongo_db]


async def log_query(
    query: str,
    answer: str,
    sources: list,
    model_type: str,
    rag_used: bool,
    latency_ms: float,
    voice_input: bool = False,
) -> str:
    doc = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "model_type": model_type,
        "rag_used": rag_used,
        "latency_ms": round(latency_ms, 2),
        "voice_input": voice_input,
        "timestamp": datetime.utcnow().isoformat(),
    }

    db = get_db()
    if db is not None:
        try:
            result = await db.queries.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.warning(f"MongoDB write failed, using file fallback: {e}")

    _append_json(QUERIES_FILE, doc)
    return f"file_{datetime.utcnow().timestamp()}"


async def get_query_history(limit: int = 50) -> list:
    db = get_db()
    if db is not None:
        try:
            from pymongo import DESCENDING
            cursor = db.queries.find({}, {"_id": 0}).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.warning(f"MongoDB read failed: {e}")

    data = _read_json(QUERIES_FILE)
    return data[:limit]


async def log_evaluation(model_name: str, metrics: dict) -> None:
    doc = {
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }
    db = get_db()
    if db is not None:
        try:
            await db.evaluations.insert_one(doc)
            return
        except Exception:
            pass
    _append_json(EVALS_FILE, doc)


async def get_evaluations() -> list:
    db = get_db()
    if db is not None:
        try:
            from pymongo import DESCENDING
            cursor = db.evaluations.find({}, {"_id": 0}).sort(
                "timestamp", DESCENDING
            )
            return await cursor.to_list(length=100)
        except Exception:
            pass
    return _read_json(EVALS_FILE)


async def get_stats() -> dict:
    history = await get_query_history(500)
    total = len(history)
    rag_count = sum(1 for h in history if h.get("rag_used"))
    voice_count = sum(1 for h in history if h.get("voice_input"))
    latencies = [h.get("latency_ms", 0) for h in history if h.get("latency_ms")]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

    return {
        "total_queries": total,
        "rag_queries": rag_count,
        "voice_queries": voice_count,
        "avg_latency_ms": avg_latency,
        "storage": "mongodb" if _mongo_available else "file",
    }
