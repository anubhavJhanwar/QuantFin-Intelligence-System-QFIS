"""
RAG Retriever for QFIS.
Retrieves top-k relevant financial contexts from FAISS index
and augments the query before passing to the LLM.
"""

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
_candidate = ROOT / "data" / "processed"
if not _candidate.exists():
    ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "faiss_metadata.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

_index = None
_metadata = None
_embedder = None


def _load_resources():
    global _index, _metadata, _embedder
    if _index is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Run faiss_index.py first."
        )

    logger.info("Loading FAISS index and embedder...")
    _index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        _metadata = pickle.load(f)
    _embedder = SentenceTransformer(EMBED_MODEL)
    logger.info(f"FAISS index loaded: {_index.ntotal} vectors")


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Retrieve top-k relevant contexts for a query.
    Returns list of dicts with context, question, answer, score.
    """
    _load_resources()

    query_emb = _embedder.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = _index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        meta = _metadata[idx]
        results.append({
            "context": meta["context"],
            "question": meta["question"],
            "answer": meta["answer"],
            "score": float(score),
        })

    return results


def build_rag_prompt(query: str, retrieved: list) -> str:
    """Build augmented prompt with retrieved contexts."""
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(f"[Context {i}]\n{r['context']}")

    combined_context = "\n\n".join(context_parts)

    return (
        f"You are a financial analyst. Use the following retrieved contexts "
        f"to answer the question accurately and concisely.\n\n"
        f"{combined_context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def retrieve_and_build_prompt(query: str, top_k: int = TOP_K) -> tuple:
    """Retrieve contexts and build RAG prompt. Returns (prompt, retrieved_docs)."""
    retrieved = retrieve(query, top_k)
    prompt = build_rag_prompt(query, retrieved)
    return prompt, retrieved
