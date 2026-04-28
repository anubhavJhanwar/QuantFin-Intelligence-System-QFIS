"""
QFIS Evaluation — BLEU / ROUGE / EM / F1
Each model runs in its own subprocess → full VRAM release guaranteed.
"""

import gc, json, re, string, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer as rs

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_DIR   = ROOT.parent / "models" / "v1"
RESULTS_DIR = DATA_DIR
LOG_DIR     = ROOT.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "evaluate.log", rotation="5 MB")

BOOL_MAP = {
    "true":"yes","false":"no","correct":"yes","incorrect":"no",
    "higher":"yes","lower":"no","greater":"yes","less":"no",
    "increased":"yes","decreased":"no","more":"yes","fewer":"no",
}

# Unit multipliers for converting to a canonical base unit (millions)
UNIT_MAP = {
    "trillion": 1_000_000, "trillions": 1_000_000,
    "billion":  1_000,     "billions":  1_000,
    "million":  1,         "millions":  1,
    "thousand": 0.001,     "thousands": 0.001,
    "k": 0.001, "m": 1, "b": 1_000, "t": 1_000_000,
}

def _to_canonical_number(t: str) -> str:
    """
    Convert any numeric expression to a canonical float string.
    Handles: $5M, 5,000,000, USD 5 million, 5 billion, 5.3b, etc.
    Returns the original string if no numeric pattern found.
    """
    t = t.strip()
    # Strip currency symbols and codes
    t = re.sub(r"(?i)^(usd|eur|gbp|inr|cad)\s*", "", t)
    t = re.sub(r"\$\s*", "", t)
    # Remove commas in numbers
    t = re.sub(r"(\d),(\d)", r"\1\2", t)

    # Pattern: number followed by unit (e.g. "5.3 billion", "5b", "5M")
    m = re.match(
        r"^(-?\d+\.?\d*)\s*(trillion|trillions|billion|billions|million|millions|thousand|thousands|[kmbt])s?\b(.*)$",
        t, re.IGNORECASE
    )
    if m:
        num   = float(m.group(1))
        unit  = m.group(2).lower().rstrip("s")
        rest  = m.group(3).strip()
        mult  = UNIT_MAP.get(unit, 1)
        val   = num * mult
        # Format cleanly
        result = f"{val:.4f}".rstrip("0").rstrip(".")
        # Preserve % if it was in the rest
        if "%" in rest:
            result += "%"
        return result

    # Pattern: shorthand like $5M, $3.2B at end
    m = re.match(r"^(-?\d+\.?\d*)\s*([kmbt])%?$", t, re.IGNORECASE)
    if m:
        num  = float(m.group(1))
        unit = m.group(2).lower()
        mult = UNIT_MAP.get(unit, 1)
        val  = num * mult
        return f"{val:.4f}".rstrip("0").rstrip(".")

    return t


def normalize(text: str) -> str:
    # Step 1: clean escape sequences and whitespace
    t = str(text).lower().strip()
    t = re.sub(r"\\+n", " ", t)          # remove escaped newlines like \\n
    t = re.sub(r"\\+t", " ", t)          # remove escaped tabs
    t = t.strip()

    # Step 2: boolean synonym mapping (before any other processing)
    if t in BOOL_MAP:
        return BOOL_MAP[t]

    # Step 3: normalize percentage spacing
    t = re.sub(r"\s*%", "%", t)

    # Step 4: try canonical number conversion (handles $5M, 5 billion, etc.)
    converted = _to_canonical_number(t)
    if converted != t:
        # Successfully converted — normalize the result
        t = converted

    # Step 5: strip currency symbols, commas in numbers
    t = re.sub(r"\$\s*", "", t)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)

    # Step 6: strip remaining punctuation except % . -
    t = t.translate(str.maketrans("", "", string.punctuation.replace("%","").replace(".","").replace("-","")))
    t = re.sub(r"\s+", " ", t).strip()

    # Step 7: boolean check again after cleaning
    if t in BOOL_MAP:
        return BOOL_MAP[t]

    # Step 8: if still long text, extract first number/yes/no
    if len(t.split()) > 3:
        m = re.search(r"(-?\d[\d]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", t)
        if m:
            extracted = m.group(1).strip()
            return BOOL_MAP.get(extracted, extracted)

    # Step 9: normalize float representation — remove trailing zeros
    # e.g. "4.50" == "4.5", "93.40" == "93.4"
    m = re.match(r"^(-?\d+\.\d+)(%?)$", t)
    if m:
        try:
            val = float(m.group(1))
            pct_suffix = m.group(2)
            t = f"{val:.10f}".rstrip("0").rstrip(".") + pct_suffix
        except ValueError:
            pass

    return t

def exact_match(p, r): return float(normalize(p) == normalize(r))

def token_f1(p, r):
    pt, rt = normalize(p).split(), normalize(r).split()
    if not pt or not rt: return 0.0
    common = Counter(pt) & Counter(rt)
    num = sum(common.values())
    if num == 0: return 0.0
    prec, rec = num/len(pt), num/len(rt)
    return 2*prec*rec/(prec+rec)

def compute_metrics(preds, refs):
    sm = SmoothingFunction().method1
    tr = [[normalize(r).split()] for r in refs]
    tp = [normalize(p).split()   for p in preds]
    b1 = corpus_bleu(tr, tp, weights=(1,0,0,0), smoothing_function=sm)
    b4 = corpus_bleu(tr, tp, weights=(.25,.25,.25,.25), smoothing_function=sm)
    scorer = rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    r1,r2,rl,em_l,f1_l = [],[],[],[],[]
    for p,r in zip(preds,refs):
        s = scorer.score(normalize(r), normalize(p))
        r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        em_l.append(exact_match(p,r)); f1_l.append(token_f1(p,r))
    n = len(preds)
    return {
        "bleu1":       round(b1,4), "bleu4":       round(b4,4),
        "rouge1":      round(sum(r1)/n,4), "rouge2": round(sum(r2)/n,4),
        "rougeL":      round(sum(rl)/n,4),
        "exact_match": round(sum(em_l)/n,4), "f1": round(sum(f1_l)/n,4),
    }

def run_subprocess_model(model_key: str, out_path: str):
    """Run one model in a fresh subprocess — guarantees VRAM is fully freed."""
    script = Path(__file__).parent / "run_one_model.py"
    cmd = [sys.executable, str(script), "--model", model_key, "--out", out_path]
    logger.info(f"Subprocess: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Subprocess failed for model={model_key}")
        sys.exit(1)

def print_table(results):
    keys   = ["bleu1","bleu4","rouge1","rouge2","rougeL","exact_match","f1"]
    models = list(results.keys())
    w = 14 + 20*len(models)
    print("\n" + "="*w)
    print("  QFIS EVALUATION RESULTS")
    print("="*w)
    print(f"{'Metric':<14}" + "".join(f"{m:>20}" for m in models))
    print("-"*w)
    for k in keys:
        row = f"{k:<14}"
        for m in models:
            row += f"{results[m]['metrics'][k]:>20.4f}"
        print(row)
    print("="*w)
    base = results["Base Phi-2"]["metrics"]
    ft   = results["Fine-tuned (QLoRA)"]["metrics"]
    print("\n  IMPROVEMENT: Fine-tuned vs Base")
    print("-"*45)
    for k in keys:
        d = ft[k] - base[k]
        bar = ("▲" if d>0 else "▼") * max(1, int(abs(d)*40))
        print(f"  {k:<14} {d:+.4f}  {bar}")
    print()

def plot_comparison(results):
    models = list(results.keys())
    keys   = ["bleu1","bleu4","rouge1","rouge2","rougeL","exact_match","f1"]
    labels = ["BLEU-1","BLEU-4","ROUGE-1","ROUGE-2","ROUGE-L","Exact Match","F1"]
    colors = ["#6c757d","#0d6efd","#198754"]
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    fig.patch.set_facecolor("#0f1629")
    for ax in axes: ax.set_facecolor("#151e38")
    x, w = range(len(keys)), 0.25
    ax = axes[0]
    for i,(m,c) in enumerate(zip(models,colors)):
        vals = [results[m]["metrics"][k] for k in keys]
        ax.bar([xi+i*w for xi in x], vals, w, label=m, color=c, alpha=0.9)
    ax.set_xticks([xi+w for xi in x])
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Score", color="white")
    ax.set_title("QFIS: Base vs Prompt-Engineered vs Fine-tuned", color="white", fontweight="bold")
    ax.legend(facecolor="#1a2647", labelcolor="white")
    ax.set_ylim(0,1); ax.tick_params(colors="white")
    ax.spines[:].set_color("#2563eb"); ax.grid(axis="y", alpha=0.2, color="white")
    bv = [results["Base Phi-2"]["metrics"][k] for k in keys]
    fv = [results["Fine-tuned (QLoRA)"]["metrics"][k] for k in keys]
    impr = [round(f-b,4) for f,b in zip(fv,bv)]
    ax2 = axes[1]
    bars = ax2.barh(labels, impr, color=["#198754" if v>=0 else "#dc3545" for v in impr], alpha=0.9)
    ax2.axvline(0, color="white", lw=0.8)
    ax2.set_xlabel("Improvement over Base", color="white")
    ax2.set_title("Fine-tuned vs Base: Improvement", color="white", fontweight="bold")
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#2563eb")
    for bar,val in zip(bars,impr):
        ax2.text(val+(0.003 if val>=0 else -0.003), bar.get_y()+bar.get_height()/2,
                 f"{val:+.4f}", va="center", fontsize=9, color="white",
                 ha="left" if val>=0 else "right")
    ax2.grid(axis="x", alpha=0.2, color="white")
    plt.tight_layout()
    out = RESULTS_DIR / "evaluation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1629")
    logger.info(f"Graph saved: {out}")
    plt.close()

def analyze_hallucinations(results):
    out = []
    for ex in results.get("Fine-tuned (QLoRA)",{}).get("examples",[]):
        pn = set(re.findall(r"\d+\.?\d*", normalize(ex["prediction"])))
        rn = set(re.findall(r"\d+\.?\d*", normalize(ex["reference"])))
        hall = pn - rn
        if hall and ex["exact_match"] == 0:
            out.append({"question":ex["question"],"reference":ex["reference"],
                        "prediction":ex["prediction"],"hallucinated_values":list(hall),
                        "type":"numerical_hallucination"})
    return out[:8]

def load_preds(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    preds = [r["prediction"] for r in records]
    refs  = [r["reference"]  for r in records]
    metrics = compute_metrics(preds, refs)
    examples = [
        {**r, "exact_match": exact_match(r["prediction"], r["reference"])}
        for r in records[:8]
    ]
    return {"metrics": metrics, "examples": examples}

def run_evaluation():
    logger.info("="*60)
    logger.info("QFIS Evaluation — BLEU / ROUGE / EM / F1")
    logger.info("="*60)

    if not (MODEL_DIR / "adapter_config.json").exists():
        logger.error("Fine-tuned model not found in models/v1/")
        sys.exit(1)

    tmp = Path(tempfile.gettempdir())
    paths = {
        "Base Phi-2":         str(tmp / "qfis_base_preds.json"),
        "Prompt-Engineered":  str(tmp / "qfis_pe_preds.json"),
        "Fine-tuned (QLoRA)": str(tmp / "qfis_ft_preds.json"),
    }
    keys = {"Base Phi-2":"base", "Prompt-Engineered":"pe", "Fine-tuned (QLoRA)":"ft"}

    # Run each model in its own subprocess
    for label, out_path in paths.items():
        logger.info(f"Running: {label}")
        run_subprocess_model(keys[label], out_path)
        logger.info(f"Done: {label}")

    # Compute metrics from saved predictions
    results = {label: load_preds(path) for label, path in paths.items()}

    print_table(results)
    plot_comparison(results)
    hall = analyze_hallucinations(results)

    final = {
        "results": results,
        "hallucination_analysis": hall,
        "improvement_over_base": {
            k: round(results["Fine-tuned (QLoRA)"]["metrics"][k]
                     - results["Base Phi-2"]["metrics"][k], 4)
            for k in results["Base Phi-2"]["metrics"]
        },
        "eval_samples": 40,
    }
    out = RESULTS_DIR / "evaluation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {out}")
    logger.info("View at http://localhost:3000/evaluate")

if __name__ == "__main__":
    run_evaluation()
