"""
/api/evaluate — returns evaluation results and comparison data.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.services.database import get_evaluations

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])

ROOT = Path(__file__).resolve().parents[4]
RESULTS_PATH = ROOT / "data" / "processed" / "evaluation_results.json"
GRAPH_PATH = ROOT / "data" / "processed" / "evaluation_comparison.png"


@router.get("")
async def get_evaluation_results():
    """Return BLEU/ROUGE/EM/F1 comparison across all three models."""
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run backend/training/evaluate.py first.",
        )
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


@router.get("/history")
async def evaluation_history():
    """Return evaluation runs stored in MongoDB."""
    return await get_evaluations()


@router.get("/graph/exists")
async def graph_exists():
    return {"exists": GRAPH_PATH.exists(), "path": str(GRAPH_PATH)}
