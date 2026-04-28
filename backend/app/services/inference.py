"""
Inference service — loads fine-tuned Phi-2 once, serves forever.
Uses CPU offloading to handle Phi-2 (5.5GB) on 4GB VRAM RTX 2050.
"""

import sys
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "models").exists():
    ROOT = ROOT.parent
MODEL_SAVE_DIR = ROOT / "models" / "v1"

sys.path.insert(0, str(ROOT / "backend"))

from rag.retriever import retrieve_and_build_prompt

BASE_MODEL = "microsoft/phi-2"
MAX_NEW_TOKENS = 100

_ft_model   = None
_tokenizer  = None
_device     = None


def _load():
    global _ft_model, _tokenizer, _device

    if _ft_model is not None:
        return

    has_cuda = torch.cuda.is_available()
    _device  = "cuda" if has_cuda else "cpu"
    logger.info(f"Loading Phi-2 | CUDA: {has_cuda}")

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    load_kwargs = dict(
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        trust_remote_code=True,
    ) if has_cuda else dict(torch_dtype=torch.float32, trust_remote_code=True)

    logger.info("Loading base Phi-2...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

    if (MODEL_SAVE_DIR / "adapter_config.json").exists():
        logger.info("Loading fine-tuned adapter (QLoRA)...")
        import json as _json
        cfg_path = MODEL_SAVE_DIR / "adapter_config.json"
        cfg = _json.loads(cfg_path.read_text())
        for k in ["eva_config","lora_bias","exclude_modules","layer_replication","use_dora","use_rslora"]:
            cfg.pop(k, None)
        cfg_path.write_text(_json.dumps(cfg, indent=2))
        _ft_model = PeftModel.from_pretrained(base, str(MODEL_SAVE_DIR))
    else:
        logger.warning("No fine-tuned adapter found — using base only.")
        _ft_model = base

    _ft_model.eval()
    logger.info("Model ready.")


def _generate(model, prompt: str) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
    if _device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for sep in ["\n\nQuestion", "\n\nContext", "\n\n"]:
        if sep in answer:
            answer = answer.split(sep)[0].strip()
    return answer or "Unable to generate an answer."


def answer_query(query: str, use_rag: bool = True) -> dict:
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

    # Generate fine-tuned answer first
    ft_answer = _generate(_ft_model, prompt)

    # For base answer: disable the adapter temporarily
    if hasattr(_ft_model, "disable_adapter"):
        with _ft_model.disable_adapter():
            base_answer = _generate(_ft_model, prompt)
    else:
        base_answer = _generate(_base_model, prompt)

    sources = [
        {"context": d["context"][:300], "score": round(d["score"], 3)}
        for d in retrieved_docs
    ]

    return {
        "answer":       ft_answer,
        "base_answer":  base_answer,
        "sources":      sources,
        "model_type":   "fine-tuned (QLoRA)",
        "rag_used":     use_rag and len(retrieved_docs) > 0,
    }


def get_status() -> dict:
    return {
        "model_loaded":        _ft_model is not None,
        "model_type":          "fine-tuned (QLoRA)",
        "device":              str(_device) if _device else "not loaded",
        "finetuned_available": (MODEL_SAVE_DIR / "adapter_config.json").exists(),
    }
