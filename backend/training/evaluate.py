"""
QFIS Evaluation — correct approach for FinQA dataset.

FinQA answers are SHORT: numbers, percentages, yes/no (e.g. "5.3", "12%", "yes").
All three models must generate short answers in the same format.
Fine-tuned model is loaded once on GPU and generates answers directly.

Models compared:
  1. Base Phi-2          — no fine-tuning, direct prompt
  2. Prompt-Engineered   — system prompt added, no fine-tuning
  3. Fine-tuned (QLoRA)  — LoRA adapter from Colab training
"""

import gc
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_score import rouge_scorer as rs
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_SAVE_DIR = ROOT.parent / "models" / "v1"
RESULTS_DIR    = DATA_DIR
LOG_DIR        = ROOT.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "evaluate.log", rotation="5 MB")

BASE_MODEL      = "microsoft/phi-2"
MAX_EVAL        = 40      # 40 samples × 3 models — fast on GPU
MAX_NEW_TOKENS  = 20      # FinQA answers are short numbers/percentages

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer using only the context. "
    "Give a short precise answer — a number, percentage, or yes/no.\n\n"
)


# ── Prompt builders ────────────────────────────────────────────────────────────

def make_base_prompt(r: dict) -> str:
    ctx = r["context"][:600]
    return f"Context: {ctx}\n\nQuestion: {r['question']}\n\nAnswer:"


def make_pe_prompt(r: dict) -> str:
    ctx = r["context"][:600]
    return (
        f"{SYSTEM_PROMPT}"
        f"Context: {ctx}\n\nQuestion: {r['question']}\n\nAnswer:"
    )


def make_ft_prompt(r: dict) -> str:
    # Fine-tuned model was trained on this exact format
    ctx = r["context"][:600]
    return f"Context: {ctx}\n\nQuestion: {r['question']}\n\nAnswer:"


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def extract_short_answer(raw: str) -> str:
    """
    Extract the first meaningful token from model output.
    FinQA answers are: numbers, percentages, yes/no, short phrases.
    """
    raw = raw.strip()
    # Take only first line
    raw = raw.split("\n")[0].strip()
    # Remove leading "Answer:" if model repeated it
    raw = re.sub(r"^answer\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    # Take first sentence
    raw = re.split(r"[.!?]", raw)[0].strip()
    # If very long, take first 8 words
    words = raw.split()
    if len(words) > 8:
        raw = " ".join(words[:8])
    return raw if raw else "N/A"


# ── Metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p = normalize(pred).split()
    r = normalize(ref).split()
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    num = sum(common.values())
    if num == 0:
        return 0.0
    return 2 * (num / len(p)) * (num / len(r)) / ((num / len(p)) + (num / len(r)))


def compute_metrics(preds: list, refs: list) -> dict:
    smoother  = SmoothingFunction().method1
    tok_refs  = [[normalize(r).split()] for r in refs]
    tok_preds = [normalize(p).split()   for p in preds]

    bleu1 = corpus_bleu(tok_refs, tok_preds, weights=(1,0,0,0),
                        smoothing_function=smoother)
    bleu4 = corpus_bleu(tok_refs, tok_preds, weights=(.25,.25,.25,.25),
                        smoothing_function=smoother)

    scorer = rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    r1, r2, rl, em_l, f1_l = [], [], [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        em_l.append(exact_match(p, r))
        f1_l.append(token_f1(p, r))

    n = len(preds)
    return {
        "bleu1":       round(bleu1, 4),
        "bleu4":       round(bleu4, 4),
        "rouge1":      round(sum(r1)/n, 4),
        "rouge2":      round(sum(r2)/n, 4),
        "rougeL":      round(sum(rl)/n, 4),
        "exact_match": round(sum(em_l)/n, 4),
        "f1":          round(sum(f1_l)/n, 4),
    }


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(adapter: bool = False):
    """Load Phi-2 with 4-bit quant on GPU. Optionally attach LoRA adapter."""
    has_cuda = torch.cuda.is_available()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if has_cuda else None

    kwargs = dict(trust_remote_code=True)
    if bnb:
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    if adapter:
        # Patch adapter_config for local PEFT version compatibility
        cfg_path = MODEL_SAVE_DIR / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        for key in ["eva_config","lora_bias","exclude_modules",
                    "layer_replication","use_dora","use_rslora"]:
            cfg.pop(key, None)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        model = PeftModel.from_pretrained(base, str(MODEL_SAVE_DIR))
    else:
        model = base

    model.eval()
    return model


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str) -> str:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_short_answer(raw)


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_model_eval(model, tokenizer, records, label, prompt_fn):
    preds, refs, examples = [], [], []
    for r in tqdm(records, desc=label):
        prompt = prompt_fn(r)
        pred   = generate_answer(model, tokenizer, prompt)
        ref    = r["completion"]
        preds.append(pred)
        refs.append(ref)
        if len(examples) < 8:
            examples.append({
                "question":    r["question"],
                "reference":   ref,
                "prediction":  pred,
                "exact_match": exact_match(pred, ref),
            })
    metrics = compute_metrics(preds, refs)
    logger.info(f"  {label}: {metrics}")
    return {"metrics": metrics, "examples": examples}


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(results: dict):
    keys   = ["bleu1","bleu4","rouge1","rouge2","rougeL","exact_match","f1"]
    models = list(results.keys())
    w      = 14 + 20 * len(models)
    print("\n" + "=" * w)
    print("  QFIS EVALUATION RESULTS")
    print("=" * w)
    print(f"{'Metric':<14}" + "".join(f"{m:>20}" for m in models))
    print("-" * w)
    for k in keys:
        row = f"{k:<14}"
        for m in models:
            row += f"{results[m]['metrics'][k]:>20.4f}"
        print(row)
    print("=" * w)

    base = results["Base Phi-2"]["metrics"]
    ft   = results["Fine-tuned (QLoRA)"]["metrics"]
    print("\n  IMPROVEMENT: Fine-tuned vs Base")
    print("-" * 45)
    for k in keys:
        d = ft[k] - base[k]
        bar = ("▲" if d > 0 else "▼") * max(1, int(abs(d) * 30))
        print(f"  {k:<14} {d:+.4f}  {bar}")
    print()


# ── Graph ─────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict):
    models = list(results.keys())
    keys   = ["bleu1","bleu4","rouge1","rouge2","rougeL","exact_match","f1"]
    labels = ["BLEU-1","BLEU-4","ROUGE-1","ROUGE-2","ROUGE-L","Exact Match","F1"]
    colors = ["#6c757d","#0d6efd","#198754"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1629")
    for ax in axes:
        ax.set_facecolor("#151e38")

    x, w = range(len(keys)), 0.25
    ax = axes[0]
    for i, (m, c) in enumerate(zip(models, colors)):
        vals = [results[m]["metrics"][k] for k in keys]
        ax.bar([xi + i*w for xi in x], vals, w, label=m, color=c, alpha=0.9)
    ax.set_xticks([xi + w for xi in x])
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Score", color="white")
    ax.set_title("QFIS: Base vs Prompt-Engineered vs Fine-tuned (QLoRA)",
                 color="white", fontweight="bold")
    ax.legend(facecolor="#1a2647", labelcolor="white")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2563eb")
    ax.grid(axis="y", alpha=0.2, color="white")

    base_vals = [results["Base Phi-2"]["metrics"][k] for k in keys]
    ft_vals   = [results["Fine-tuned (QLoRA)"]["metrics"][k] for k in keys]
    impr      = [round(f - b, 4) for f, b in zip(ft_vals, base_vals)]
    bar_cols  = ["#198754" if v >= 0 else "#dc3545" for v in impr]
    ax2 = axes[1]
    bars = ax2.barh(labels, impr, color=bar_cols, alpha=0.9)
    ax2.axvline(0, color="white", linewidth=0.8)
    ax2.set_xlabel("Improvement over Base Model", color="white")
    ax2.set_title("Fine-tuned vs Base: Score Improvement",
                  color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#2563eb")
    for bar, val in zip(bars, impr):
        ax2.text(val + (0.003 if val >= 0 else -0.003),
                 bar.get_y() + bar.get_height()/2,
                 f"{val:+.4f}", va="center", fontsize=9, color="white",
                 ha="left" if val >= 0 else "right")
    ax2.grid(axis="x", alpha=0.2, color="white")

    plt.tight_layout()
    out = RESULTS_DIR / "evaluation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1629")
    logger.info(f"Graph saved: {out}")
    plt.close()


# ── Hallucination analysis ────────────────────────────────────────────────────

def analyze_hallucinations(results: dict) -> list:
    out = []
    for ex in results.get("Fine-tuned (QLoRA)", {}).get("examples", []):
        pred_nums = set(re.findall(r"\d+\.?\d*", normalize(ex["prediction"])))
        ref_nums  = set(re.findall(r"\d+\.?\d*", normalize(ex["reference"])))
        hallucinated = pred_nums - ref_nums
        if hallucinated and ex["exact_match"] == 0:
            out.append({
                "question":            ex["question"],
                "reference":           ex["reference"],
                "prediction":          ex["prediction"],
                "hallucinated_values": list(hallucinated),
                "type":                "numerical_hallucination",
            })
    return out[:8]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    logger.info("=" * 60)
    logger.info("QFIS Evaluation — BLEU / ROUGE / EM / F1")
    logger.info("=" * 60)

    if not (MODEL_SAVE_DIR / "adapter_config.json").exists():
        logger.error("Fine-tuned model not found in models/v1/")
        sys.exit(1)

    test_path = DATA_DIR / "test.json"
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_EVAL]
    logger.info(f"Test samples: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ── 1. Base model ──────────────────────────────────────────────────────────
    logger.info("Loading Base Phi-2...")
    model = load_model(adapter=False)
    results["Base Phi-2"] = run_model_eval(
        model, tokenizer, test_data, "Base Phi-2", make_base_prompt
    )
    unload(model)

    # ── 2. Prompt-engineered ───────────────────────────────────────────────────
    logger.info("Loading Prompt-Engineered Phi-2...")
    model = load_model(adapter=False)
    results["Prompt-Engineered"] = run_model_eval(
        model, tokenizer, test_data, "Prompt-Engineered", make_pe_prompt
    )
    unload(model)

    # ── 3. Fine-tuned (QLoRA) ──────────────────────────────────────────────────
    logger.info("Loading Fine-tuned Phi-2 (QLoRA adapter)...")
    model = load_model(adapter=True)
    results["Fine-tuned (QLoRA)"] = run_model_eval(
        model, tokenizer, test_data, "Fine-tuned (QLoRA)", make_ft_prompt
    )
    unload(model)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_table(results)
    plot_comparison(results)
    hallucinations = analyze_hallucinations(results)

    final = {
        "results":              results,
        "hallucination_analysis": hallucinations,
        "improvement_over_base": {
            k: round(
                results["Fine-tuned (QLoRA)"]["metrics"][k]
                - results["Base Phi-2"]["metrics"][k], 4
            )
            for k in results["Base Phi-2"]["metrics"]
        },
        "eval_samples": MAX_EVAL,
    }

    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {out_path}")
    logger.info("Open http://localhost:3000/evaluate to view in UI")


if __name__ == "__main__":
    run_evaluation()
