"""
/api/query — main financial QA endpoint.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from app.services.inference import answer_query, get_status
from app.services.database import log_query

router = APIRouter(prefix="/api/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    use_rag: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    model_type: str
    rag_used: bool
    latency_ms: float
    query_id: str


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Answer a financial question using RAG + fine-tuned Phi-2."""
    logger.info(f"Query: {req.question[:80]}...")
    start = time.time()

    try:
        result = answer_query(req.question, use_rag=req.use_rag)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start) * 1000

    query_id = await log_query(
        query=req.question,
        answer=result["answer"],
        sources=result["sources"],
        model_type=result["model_type"],
        rag_used=result["rag_used"],
        latency_ms=latency,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        model_type=result["model_type"],
        rag_used=result["rag_used"],
        latency_ms=round(latency, 2),
        query_id=query_id,
    )


@router.get("/status")
async def model_status():
    return get_status()
