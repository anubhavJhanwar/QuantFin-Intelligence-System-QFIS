"""
MongoDB service for QFIS.
Stores queries, responses, and logs for evaluation and debugging.
"""

from datetime import datetime
from typing import Optional

from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING

from app.config.settings import settings

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_db():
    return get_client()[settings.mongo_db]


async def log_query(
    query: str,
    answer: str,
    sources: list,
    model_type: str,
    rag_used: bool,
    latency_ms: float,
    voice_input: bool = False,
) -> str:
    """Store a query-response pair in MongoDB."""
    db = get_db()
    doc = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "model_type": model_type,
        "rag_used": rag_used,
        "latency_ms": round(latency_ms, 2),
        "voice_input": voice_input,
        "timestamp": datetime.utcnow(),
    }
    result = await db.queries.insert_one(doc)
    return str(result.inserted_id)


async def get_query_history(limit: int = 50) -> list:
    db = get_db()
    cursor = db.queries.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
    return await cursor.to_list(length=limit)


async def log_evaluation(model_name: str, metrics: dict) -> None:
    db = get_db()
    await db.evaluations.insert_one({
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": datetime.utcnow(),
    })


async def get_evaluations() -> list:
    db = get_db()
    cursor = db.evaluations.find({}, {"_id": 0}).sort("timestamp", DESCENDING)
    return await cursor.to_list(length=100)


async def get_stats() -> dict:
    db = get_db()
    total = await db.queries.count_documents({})
    rag_count = await db.queries.count_documents({"rag_used": True})
    voice_count = await db.queries.count_documents({"voice_input": True})

    pipeline = [{"$group": {"_id": None, "avg_latency": {"$avg": "$latency_ms"}}}]
    agg = await db.queries.aggregate(pipeline).to_list(1)
    avg_latency = agg[0]["avg_latency"] if agg else 0

    return {
        "total_queries": total,
        "rag_queries": rag_count,
        "voice_queries": voice_count,
        "avg_latency_ms": round(avg_latency, 2),
    }
