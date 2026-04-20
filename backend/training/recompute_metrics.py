"""
Recompute metrics from saved prediction files using fixed normalization.
No GPU needed — runs instantly.
"""
import json, re, string, tempfile
from collections import Counter
from pathlib import Path
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer as rs

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_DIR   = ROOT.parent / "models" / "v1"
RESULTS_DIR = DATA_DIR

BOOL_MAP = {
    "true":"yes","false":"no","correct":"yes","incorrect":"no",
    "higher":"yes","lower":"no","greater":"yes","less":"no",
    "increased":"yes","decreased":"no","more":"yes","fewer":"no",
}

def normalize(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"\$\s*", "", t)
    t = re.sub(r"\s*%", "%", t)
    t = t.translate(str.maketrans("", "", string.punctuation.replace("%","").replace(".","").replace("-","")))
    t = re.sub(r"\s+", " ", t).strip()
    if t in BOOL_MAP:
        return BOOL_MAP[t]
    if len(t.split()) > 3:
        m = re.search(r"(-?\d[\d]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", t)
        if m:
            val = m.group(1).strip()
            return BOOL_MAP.get(val, val)
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

tmp = Path(tempfile.gettempdir())
model_files = {
    "Base Phi-2":         tmp / "qfis_base_preds.json",
    "Prompt-Engineered":  tmp / "qfis_pe_preds.json",
    "Fine-tuned (QLoRA)": tmp / "qfis_ft_preds.json",
}

results = {}
for label, path in model_files.items():
    if not path.exists():
        print(f"Missing: {path}"); continue
    records = json.load(open(path, encoding="utf-8"))
    preds = [r["prediction"] for r in records]
    refs  = [r["reference"]  for r in records]
    metrics = compute_metrics(preds, refs)
    examples = [{**r, "exact_match": exact_match(r["prediction"], r["reference"])} for r in records[:8]]
    results[label] = {"metrics": metrics, "examples": examples}
    print(f"{label}: {metrics}")

# Print table
keys   = ["bleu1","bleu4","rouge1","rouge2","rougeL","exact_match","f1"]
models = list(results.keys())
w = 14 + 20*len(models)
print("\n" + "="*w)
print("  QFIS EVALUATION RESULTS (Fixed Normalization)")
print("="*w)
print(f"{'Metric':<14}" + "".join(f"{m:>20}" for m in models))
print("-"*w)
for k in keys:
    row = f"{k:<14}"
    for m in models:
        row += f"{results[m]['metrics'][k]:>20.4f}"
    print(row)
print("="*w)

if "Base Phi-2" in results and "Fine-tuned (QLoRA)" in results:
    base = results["Base Phi-2"]["metrics"]
    ft   = results["Fine-tuned (QLoRA)"]["metrics"]
    print("\n  IMPROVEMENT: Fine-tuned vs Base")
    print("-"*45)
    for k in keys:
        d = ft[k] - base[k]
        bar = ("▲" if d>0 else "▼") * max(1, int(abs(d)*40))
        print(f"  {k:<14} {d:+.4f}  {bar}")

# Save updated results
final = {
    "results": results,
    "hallucination_analysis": [],
    "improvement_over_base": {
        k: round(results["Fine-tuned (QLoRA)"]["metrics"][k]
                 - results["Base Phi-2"]["metrics"][k], 4)
        for k in keys
    } if "Base Phi-2" in results and "Fine-tuned (QLoRA)" in results else {},
    "eval_samples": 40,
}
out = RESULTS_DIR / "evaluation_results.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out}")
