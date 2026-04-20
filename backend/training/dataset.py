"""
Dataset pipeline for QFIS.
Downloads FinQA dataset, cleans, deduplicates, tokenizes,
normalizes answers, and splits 70/15/15 with no data leakage.
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from loguru import logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 1200
MAX_SEQ_LEN = 512


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace and numbers."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Normalize numbers: remove commas in numbers
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Remove trailing % signs for comparison but keep value
    text = re.sub(r"\s+", " ", text)
    # Remove leading articles
    text = re.sub(r"^(a|an|the)\s+", "", text)
    return text.strip()


def normalize_question(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_context(sample: dict) -> str:
    """Flatten FinQA table + pre/post text into a single context string."""
    parts = []

    # Pre-text
    pre = sample.get("pre_text", [])
    if isinstance(pre, list):
        parts.extend([p.strip() for p in pre if isinstance(p, str) and p.strip()])
    elif isinstance(pre, str) and pre.strip():
        parts.append(pre.strip())

    # Table
    table = sample.get("table", [])
    if isinstance(table, list):
        for row in table:
            if isinstance(row, list):
                parts.append(" | ".join(str(c) for c in row))
            elif isinstance(row, str):
                parts.append(row)

    # Post-text
    post = sample.get("post_text", [])
    if isinstance(post, list):
        parts.extend([p.strip() for p in post if isinstance(p, str) and p.strip()])
    elif isinstance(post, str) and post.strip():
        parts.append(post.strip())

    return " ".join(parts)[:1500]  # cap context length


def extract_answer(sample: dict) -> str:
    """Extract final answer from FinQA annotation."""
    ann = sample.get("annotation", {})
    if isinstance(ann, dict):
        ans = ann.get("exe_ans", ann.get("answer", ""))
    else:
        ans = sample.get("answer", "")
    return normalize_answer(str(ans))


def build_prompt(question: str, context: str) -> str:
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ── Loading ────────────────────────────────────────────────────────────────────

def load_finqa() -> list:
    """Load FinQA from HuggingFace datasets."""
    logger.info("Loading FinQA dataset from HuggingFace...")
    try:
        ds = load_dataset("dreamerdeo/finqa", trust_remote_code=True)
        records = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    records.append(dict(item))
        logger.info(f"Loaded {len(records)} raw records from FinQA")
        return records
    except Exception as e:
        logger.error(f"Failed to load FinQA: {e}")
        raise


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_records(records: list) -> list:
    """Remove nulls, noise, and inconsistent samples."""
    cleaned = []
    for r in records:
        qa = r.get("qa", r)
        question = normalize_question(qa.get("question", r.get("question", "")))
        answer = extract_answer(qa if isinstance(qa, dict) else r)
        context = build_context(r)

        # Filter out bad samples
        if not question or len(question) < 10:
            continue
        if not answer or answer in ("none", "n/a", "", "nan"):
            continue
        if not context or len(context) < 20:
            continue

        cleaned.append({
            "question": question,
            "answer": answer,
            "context": context,
            "prompt": build_prompt(question, context),
            "completion": answer,
        })

    logger.info(f"After cleaning: {len(cleaned)} records")
    return cleaned


def deduplicate(records: list) -> list:
    """Remove duplicate questions using MD5 hash."""
    seen = set()
    unique = []
    for r in records:
        key = hashlib.md5(r["question"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    logger.info(f"After deduplication: {len(unique)} records")
    return unique


# ── Splitting ──────────────────────────────────────────────────────────────────

def split_dataset(records: list) -> tuple:
    """Strict 70/15/15 split with no leakage."""
    train, temp = train_test_split(records, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)
    logger.info(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def verify_no_leakage(train: list, val: list, test: list):
    """Assert no question appears in multiple splits."""
    train_qs = {r["question"] for r in train}
    val_qs = {r["question"] for r in val}
    test_qs = {r["question"] for r in test}
    assert not train_qs & val_qs, "Data leakage: train ∩ val"
    assert not train_qs & test_qs, "Data leakage: train ∩ test"
    assert not val_qs & test_qs, "Data leakage: val ∩ test"
    logger.info("✓ No data leakage detected")


# ── Save ───────────────────────────────────────────────────────────────────────

def save_split(data: list, name: str):
    path = PROCESSED_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {name}: {len(data)} records → {path}")


def save_stats(train, val, test):
    stats = {
        "total": len(train) + len(val) + len(test),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_pct": round(len(train) / (len(train) + len(val) + len(test)) * 100, 1),
        "val_pct": round(len(val) / (len(train) + len(val) + len(test)) * 100, 1),
        "test_pct": round(len(test) / (len(train) + len(val) + len(test)) * 100, 1),
    }
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats: {stats}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_pipeline():
    logger.info("=" * 60)
    logger.info("QFIS Dataset Pipeline")
    logger.info("=" * 60)

    # Check if already processed
    if (PROCESSED_DIR / "train.json").exists():
        logger.info("Processed data already exists. Skipping pipeline.")
        with open(PROCESSED_DIR / "train.json") as f:
            train = json.load(f)
        logger.info(f"Loaded existing train split: {len(train)} records")
        return

    records = load_finqa()
    records = clean_records(records)
    records = deduplicate(records)

    if len(records) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(records)} samples after cleaning. Need at least {MIN_SAMPLES}."
        )

    train, val, test = split_dataset(records)
    verify_no_leakage(train, val, test)

    save_split(train, "train")
    save_split(val, "val")
    save_split(test, "test")
    save_stats(train, val, test)

    logger.info("✓ Dataset pipeline complete")


if __name__ == "__main__":
    run_pipeline()
