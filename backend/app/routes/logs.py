"""
/api/logs — query history and system stats.
"""

from fastapi import APIRouter, Query
from app.services.database import get_query_history, get_stats

router = APIRouter(prefix="/api/logs", tags=["Logs"])


@router.get("")
async def get_logs(limit: int = Query(default=50, le=200)):
    """Return recent query-response logs."""
    return await get_query_history(limit)


@router.get("/stats")
async def system_stats():
    """Return aggregate system statistics."""
    return await get_stats()
