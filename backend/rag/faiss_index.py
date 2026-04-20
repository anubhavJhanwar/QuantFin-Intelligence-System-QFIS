"""
FAISS index builder for QFIS RAG pipeline.
Embeds FinQA contexts using sentence-transformers and stores in FAISS.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "faiss_metadata.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, good quality
EMBED_DIM = 384
BATCH_SIZE = 64


def build_index():
    """Build FAISS index from training + validation data."""
    logger.info("Building FAISS index...")

    if INDEX_PATH.exists() and METADATA_PATH.exists():
        logger.info("FAISS index already exists. Skipping build.")
        return

    # Load all records
    records = []
    for split in ["train", "val"]:
        path = DATA_DIR / f"{split}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                records.extend(json.load(f))

    logger.info(f"Indexing {len(records)} records...")

    # Load embedding model
    embedder = SentenceTransformer(EMBED_MODEL)

    # Build texts to embed (context + question)
    texts = [f"{r['context']} {r['question']}" for r in records]

    # Embed in batches
    logger.info("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index (Inner Product = cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))

    # Save metadata (context + answer for each record)
    metadata = [
        {
            "question": r["question"],
            "context": r["context"],
            "answer": r["completion"],
        }
        for r in records
    ]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"✓ FAISS index saved: {INDEX_PATH} ({index.ntotal} vectors)")


if __name__ == "__main__":
    build_index()
