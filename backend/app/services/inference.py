"""
Inference service — loads fine-tuned Phi-2 once, serves forever.
Integrates RAG retrieval before generation.
"""

import sys
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from rag.retriever import retrieve_and_build_prompt

MODEL_SAVE_DIR = ROOT / "models" / "v1"
BASE_MODEL = "microsoft/phi-2"
MAX_NEW_TOKENS = 128

_model = None
_tokenizer = None
_device = None
_model_type = None


def _load():
    global _model, _tokenizer, _device, _model_type
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32

    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    if (MODEL_SAVE_DIR / "adapter_config.json").exists():
        logger.info("Loading fine-tuned Phi-2 (QLoRA adapter)...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto" if _device == "cuda" else None,
        )
        _model = PeftModel.from_pretrained(base, str(MODEL_SAVE_DIR))
        _model = _model.merge_and_unload()
        _model_type = "fine-tuned (QLoRA)"
    else:
        logger.warning("Fine-tuned model not found. Using base Phi-2.")
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, trust_remote_code=True,
            device_map="auto" if _device == "cuda" else None,
        )
        _model_type = "base"

    _model.eval()
    if _device == "cpu":
        _model.to(_device)
    logger.info(f"Model ready: {_model_type} on {_device}")


def answer_query(query: str, use_rag: bool = True) -> dict:
    """
    Answer a financial query using RAG + fine-tuned Phi-2.
    Returns answer, sources, model_type, prompt used.
    """
    _load()

    retrieved_docs = []
    if use_rag:
        try:
            prompt, retrieved_docs = retrieve_and_build_prompt(query)
        except FileNotFoundError:
            logger.warning("FAISS index not found. Falling back to direct query.")
            prompt = f"Question: {query}\n\nAnswer:"
    else:
        prompt = f"Question: {query}\n\nAnswer:"

    inputs = _tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=480
    ).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.15,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up answer
    for sep in ["\n\nQuestion", "\n\nContext", "\n\n"]:
        if sep in answer:
            answer = answer.split(sep)[0].strip()

    sources = [
        {"context": d["context"][:300], "score": round(d["score"], 3)}
        for d in retrieved_docs
    ]

    return {
        "answer": answer or "Unable to generate an answer. Please rephrase your question.",
        "sources": sources,
        "model_type": _model_type,
        "rag_used": use_rag and len(retrieved_docs) > 0,
    }


def get_status() -> dict:
    return {
        "model_loaded": _model is not None,
        "model_type": _model_type,
        "device": str(_device) if _device else "not loaded",
        "finetuned_available": (MODEL_SAVE_DIR / "adapter_config.json").exists(),
    }
