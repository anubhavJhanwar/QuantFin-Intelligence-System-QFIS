"""
Evaluation script for QFIS.
Compares three models:
  1. Base Phi-2 (no fine-tuning)
  2. Prompt-engineered Phi-2 (system prompt only)
  3. Fine-tuned Phi-2 (QLoRA adapter)
Metrics: BLEU-1, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, Exact Match, F1
Saves results + generates comparison graphs.
"""

import json
import re
import string
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from loguru import logger
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from rouge_score import rouge_scorer as rs
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]  # backend/training/ → backend/
DATA_DIR = ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_SAVE_DIR = ROOT.parent / "models" / "v1"
RESULTS_DIR = DATA_DIR
LOG_DIR = ROOT.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "evaluate.log", rotation="5 MB")

BASE_MODEL = "microsoft/phi-2"
MAX_NEW_TOKENS = 64
MAX_EVAL_SAMPLES = 50   # 50 × 3 models on CPU ≈ 20-30 min

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question accurately "
    "using only the provided context. Be concise and precise.\n\n"
)


# ── Text normalization ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(predictions: list, references: list) -> dict:
    smoother = SmoothingFunction().method1
    tok_refs = [[normalize(r).split()] for r in references]
    tok_preds = [normalize(p).split() for p in predictions]

    bleu1 = corpus_bleu(tok_refs, tok_preds, weights=(1, 0, 0, 0),
                        smoothing_function=smoother)
    bleu4 = corpus_bleu(tok_refs, tok_preds, weights=(.25, .25, .25, .25),
                        smoothing_function=smoother)

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl, em_scores, f1_scores = [], [], [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        em_scores.append(exact_match(pred, ref))
        f1_scores.append(token_f1(pred, ref))

    n = len(predictions)
    return {
        "bleu1":  round(bleu1, 4),
        "bleu4":  round(bleu4, 4),
        "rouge1": round(sum(r1) / n, 4),
        "rouge2": round(sum(r2) / n, 4),
        "rougeL": round(sum(rl) / n, 4),
        "exact_match": round(sum(em_scores) / n, 4),
        "f1":     round(sum(f1_scores) / n, 4),
    }


# ── Generation ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=480
    )
    # Use the device of the first model parameter (handles device_map="auto")
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device(device)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True).strip()
    # Take only first sentence/line
    for sep in ["\n", "."]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break
    return text or "N/A"


def run_eval(model, tokenizer, records: list, device: str, label: str,
             use_system_prompt: bool = False) -> dict:
    logger.info(f"Evaluating: {label}")
    model.eval()
    # Don't call .to(device) — device_map="auto" already handles placement
    preds, refs = [], []
    examples = []

    for r in tqdm(records, desc=label):
        prompt = (SYSTEM_PROMPT + r["prompt"]) if use_system_prompt else r["prompt"]
        pred = generate(model, tokenizer, prompt, device)
        preds.append(pred)
        refs.append(r["completion"])
        if len(examples) < 10:
            examples.append({
                "question": r["question"],
                "reference": r["completion"],
                "prediction": pred,
                "exact_match": exact_match(pred, r["completion"]),
            })

    metrics = compute_all_metrics(preds, refs)
    logger.info(f"  {label}: {metrics}")
    return {"metrics": metrics, "examples": examples}


# ── Graphs ─────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict):
    models = list(results.keys())
    metric_keys = ["bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "exact_match", "f1"]
    values = {m: [results[m]["metrics"][k] for k in metric_keys] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("QFIS: Model Comparison — Base vs Prompt-Engineered vs Fine-tuned",
                 fontsize=14, fontweight="bold")

    # Bar chart
    x = range(len(metric_keys))
    width = 0.25
    colors = ["#6c757d", "#0d6efd", "#198754"]
    ax = axes[0]
    for i, (model, color) in enumerate(zip(models, colors)):
        ax.bar([xi + i * width for xi in x], values[model], width,
               label=model, color=color, alpha=0.85)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(metric_keys, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("All Metrics Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Improvement heatmap
    base_vals = values[models[0]]
    ft_vals = values[models[-1]]
    improvements = [round(ft - base, 4) for ft, base in zip(ft_vals, base_vals)]
    ax2 = axes[1]
    colors_heat = ["#dc3545" if v < 0 else "#198754" for v in improvements]
    bars = ax2.barh(metric_keys, improvements, color=colors_heat, alpha=0.85)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Improvement over Base Model")
    ax2.set_title("Fine-tuned vs Base: Improvement")
    for bar, val in zip(bars, improvements):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.4f}", va="center", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "evaluation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Graph saved: {out_path}")
    plt.close()


def print_table(results: dict):
    metric_keys = ["bleu1", "bleu4", "rouge1", "rouge2", "rougeL", "exact_match", "f1"]
    header = f"{'Metric':<14}" + "".join(f"{m:>22}" for m in results.keys())
    print("\n" + "=" * (14 + 22 * len(results)))
    print("EVALUATION RESULTS")
    print("=" * (14 + 22 * len(results)))
    print(header)
    print("-" * (14 + 22 * len(results)))
    for k in metric_keys:
        row = f"{k:<14}"
        for m in results.keys():
            row += f"{results[m]['metrics'][k]:>22.4f}"
        print(row)
    print("=" * (14 + 22 * len(results)))


# ── Hallucination analysis ─────────────────────────────────────────────────────

def analyze_hallucinations(results: dict) -> list:
    """Find cases where fine-tuned model hallucinated vs base."""
    ft_examples = results.get("Fine-tuned (QLoRA)", {}).get("examples", [])
    hallucinations = []
    for ex in ft_examples:
        pred_norm = normalize(ex["prediction"])
        ref_norm = normalize(ex["reference"])
        # Hallucination: prediction has numbers not in reference
        pred_nums = set(re.findall(r"\d+\.?\d*", pred_norm))
        ref_nums = set(re.findall(r"\d+\.?\d*", ref_norm))
        hallucinated_nums = pred_nums - ref_nums
        if hallucinated_nums and ex["exact_match"] == 0:
            hallucinations.append({
                "question": ex["question"],
                "reference": ex["reference"],
                "prediction": ex["prediction"],
                "hallucinated_values": list(hallucinated_nums),
                "type": "numerical_hallucination",
            })
    return hallucinations[:10]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    logger.info("=" * 60)
    logger.info("QFIS Evaluation: BLEU, ROUGE, EM, F1")
    logger.info("=" * 60)

    if not (MODEL_SAVE_DIR / "adapter_config.json").exists():
        logger.error("Fine-tuned model not found. Run finetune.py first.")
        sys.exit(1)

    # Use CPU for evaluation — avoids VRAM OOM when loading 3 models sequentially
    # CPU is slower but reliable on 4GB VRAM machines
    device = "cpu"
    logger.info(f"Device: {device} (CPU eval avoids VRAM OOM on 4GB GPU)")

    with open(DATA_DIR / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_EVAL_SAMPLES]
    logger.info(f"Test samples: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import gc

    def load_base_cpu():
        """Load Phi-2 in float32 on CPU — safe for sequential evaluation."""
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    def cleanup(model):
        del model
        gc.collect()

    results = {}

    # 1. Base model
    logger.info("Loading base Phi-2 on CPU...")
    base_model = load_base_cpu()
    results["Base Phi-2"] = run_eval(base_model, tokenizer, test_data, device, "Base Phi-2")
    cleanup(base_model)

    # 2. Prompt-engineered (same weights, different prompt)
    logger.info("Loading prompt-engineered Phi-2 on CPU...")
    pe_model = load_base_cpu()
    results["Prompt-Engineered"] = run_eval(
        pe_model, tokenizer, test_data, device, "Prompt-Engineered", use_system_prompt=True
    )
    cleanup(pe_model)

    # 3. Fine-tuned — strip unknown PEFT fields, load adapter on CPU base
    logger.info("Loading fine-tuned Phi-2 (QLoRA) on CPU...")
    import json as _json
    cfg_path = MODEL_SAVE_DIR / "adapter_config.json"
    cfg = _json.loads(cfg_path.read_text())
    for unknown in ["eva_config", "lora_bias", "exclude_modules",
                    "layer_replication", "use_dora", "use_rslora"]:
        cfg.pop(unknown, None)
    cfg_path.write_text(_json.dumps(cfg, indent=2))

    ft_base = load_base_cpu()
    ft_model = PeftModel.from_pretrained(ft_base, str(MODEL_SAVE_DIR))
    ft_model = ft_model.merge_and_unload()  # merge LoRA into base for clean inference
    results["Fine-tuned (QLoRA)"] = run_eval(
        ft_model, tokenizer, test_data, device, "Fine-tuned (QLoRA)"
    )
    cleanup(ft_model)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_table(results)
    plot_comparison(results)

    hallucinations = analyze_hallucinations(results)

    final = {
        "results": results,
        "hallucination_analysis": hallucinations,
        "improvement_over_base": {
            k: round(
                results["Fine-tuned (QLoRA)"]["metrics"][k] - results["Base Phi-2"]["metrics"][k], 4
            )
            for k in results["Base Phi-2"]["metrics"]
        },
    }

    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Results saved: {out_path}")


if __name__ == "__main__":
    run_evaluation()
