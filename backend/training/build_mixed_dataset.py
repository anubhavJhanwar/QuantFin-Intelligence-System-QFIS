"""
Builds a mixed training dataset from FinQA + Synthetic samples.

Strategy:
  - FinQA: filter to simple extractable questions only
    (answer appears directly in context, short answer, no complex expressions)
  - Synthetic: generate 500 controlled single-step arithmetic samples
  - Mix and split 80/20 train/test
  - Target: ~1000-1300 total, ~800-1040 train, ~200-260 test

Output:
  data/processed/mixed_train.json
  data/processed/mixed_test.json
  data/processed/mixed_stats.json
"""

import json, re, random, hashlib
from pathlib import Path
from collections import Counter

random.seed(42)

ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── FinQA filtering ────────────────────────────────────────────────────────────

def is_simple_finqa(record: dict) -> bool:
    """
    Keep only FinQA samples where:
    1. Answer is a short plain number or percentage (≤15 chars)
    2. Answer is NOT a complex expression (no colons, slashes)
    3. Numeric part of answer appears somewhere in the context
    4. Context is not empty
    """
    answer  = str(record.get("completion", record.get("answer", ""))).strip()
    context = str(record.get("context", "")).strip()

    if not answer or not context:
        return False

    # Skip very long answers
    if len(answer) > 15:
        return False

    # Skip answers with colons (ratio format like "1:2") or slashes
    if ":" in answer or "/" in answer:
        return False

    # Skip answers that are just years
    if re.match(r"^\d{4}$", answer):
        return False

    # Skip very large raw numbers (model can't handle these)
    try:
        num = float(re.sub(r"[%,]", "", answer))
        if abs(num) > 500000:
            return False
    except ValueError:
        pass

    # Extract numeric part of answer and check if it appears in context
    answer_clean = re.sub(r"[%$,\s]", "", answer).strip()
    m = re.search(r"(\d+\.?\d*)", answer_clean)
    if m:
        num_str = m.group(1)
        if num_str in context:
            return True
        # Also try without decimal trailing zeros
        try:
            val = float(num_str)
            if str(int(val)) in context or f"{val}" in context:
                return True
        except ValueError:
            pass

    # Accept if full answer appears in context
    if answer_clean.lower() in context.lower():
        return True

    return False


def load_finqa_simple(path: Path, max_samples: int = 800) -> list:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    simple = [r for r in records if is_simple_finqa(r)]
    random.shuffle(simple)
    simple = simple[:max_samples]

    # Normalize to standard format
    out = []
    for r in simple:
        out.append({
            "question":   r["question"],
            "answer":     r.get("completion", r.get("answer", "")),
            "context":    r["context"][:900],
            "prompt":     f"Context: {r['context'][:900]}\n\nQuestion: {r['question']}\n\nAnswer:",
            "completion": r.get("completion", r.get("answer", "")),
            "source":     "finqa",
        })
    return out


# ── Synthetic generation ───────────────────────────────────────────────────────

COMPANIES = [
    "Acme Corp", "GlobalTech Inc", "Summit Financial", "Apex Industries",
    "Crestview Holdings", "Meridian Group", "Pinnacle Corp", "Vantage Capital",
    "Horizon Enterprises", "Sterling Partners", "Atlas Capital", "Nexus Group",
    "Vertex Holdings", "Quantum Finance", "Orion Industries",
]
METRICS = [
    "revenue", "net income", "operating profit", "gross profit",
    "total assets", "operating expenses", "net sales", "EBITDA",
    "cash flow from operations", "total liabilities", "net revenue",
    "cost of goods sold", "research and development expenses", "capital expenditure",
]
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


def fmt(n):
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    return str(round(n, 2))


def pct_fmt(n):
    v = round(n, 2)
    return f"{int(v)}%" if v == int(v) else f"{v}%"


def generate_synthetic(n_each_type: int) -> list:
    samples = []

    # Type 1: Direct extraction
    for _ in range(n_each_type):
        company = random.choice(COMPANIES)
        metric  = random.choice(METRICS)
        year    = random.choice(YEARS)
        value   = round(random.uniform(10, 5000), 1)
        unit    = random.choice(["million", "billion"])
        context = (
            f"{company} reported {metric} of {fmt(value)} {unit} in {year}. "
            f"The company continued to expand its operations across multiple segments "
            f"during the fiscal year."
        )
        question = f"what was the {metric} of {company} in {year} in {unit}?"
        answer   = fmt(value)
        samples.append({"question": question, "answer": answer, "context": context,
                        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                        "completion": answer, "source": "synthetic", "type": "extraction"})

    # Type 2: Difference
    for _ in range(n_each_type):
        company = random.choice(COMPANIES)
        metric  = random.choice(METRICS)
        y1, y2  = sorted(random.sample(YEARS, 2))
        v1      = round(random.uniform(50, 3000), 1)
        v2      = round(v1 + random.uniform(-500, 500), 1)
        diff    = round(v2 - v1, 1)
        context = (
            f"{company} reported {metric} of {fmt(v1)} million in {y1} "
            f"and {fmt(v2)} million in {y2}."
        )
        question = f"what was the change in {metric} from {y1} to {y2} in million?"
        answer   = fmt(diff)
        samples.append({"question": question, "answer": answer, "context": context,
                        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                        "completion": answer, "source": "synthetic", "type": "difference"})

    # Type 3: Percentage change
    for _ in range(n_each_type):
        company = random.choice(COMPANIES)
        metric  = random.choice(METRICS)
        y1, y2  = sorted(random.sample(YEARS, 2))
        v1      = round(random.uniform(100, 2000), 1)
        change  = round(random.uniform(-30, 50), 1)
        v2      = round(v1 * (1 + change / 100), 1)
        pct_val = round((v2 - v1) / v1 * 100, 2)
        context = (
            f"{company} {metric} was {fmt(v1)} million in {y1} "
            f"and {fmt(v2)} million in {y2}."
        )
        question = f"what was the percentage change in {metric} from {y1} to {y2}?"
        answer   = pct_fmt(pct_val)
        samples.append({"question": question, "answer": answer, "context": context,
                        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                        "completion": answer, "source": "synthetic", "type": "pct_change"})

    # Type 4: Ratio
    for _ in range(n_each_type // 2):
        company = random.choice(COMPANIES)
        m1, m2  = random.sample(METRICS, 2)
        year    = random.choice(YEARS)
        v1      = round(random.uniform(100, 2000), 1)
        v2      = round(random.uniform(100, 2000), 1)
        ratio   = round(v1 / v2, 3)
        context = (
            f"In {year}, {company} reported {m1} of {fmt(v1)} million "
            f"and {m2} of {fmt(v2)} million."
        )
        question = f"what is the ratio of {m1} to {m2} in {year}?"
        answer   = fmt(ratio)
        samples.append({"question": question, "answer": answer, "context": context,
                        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                        "completion": answer, "source": "synthetic", "type": "ratio"})

    # Type 5: Sum
    for _ in range(n_each_type // 2):
        company = random.choice(COMPANIES)
        metric  = random.choice(METRICS)
        years   = sorted(random.sample(YEARS, 3))
        vals    = [round(random.uniform(20, 500), 1) for _ in range(3)]
        total   = round(sum(vals), 1)
        context = (
            f"{company} reported {metric} of {fmt(vals[0])} million in {years[0]}, "
            f"{fmt(vals[1])} million in {years[1]}, "
            f"and {fmt(vals[2])} million in {years[2]}."
        )
        question = f"what was the total {metric} from {years[0]} to {years[2]} in million?"
        answer   = fmt(total)
        samples.append({"question": question, "answer": answer, "context": context,
                        "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                        "completion": answer, "source": "synthetic", "type": "sum"})

    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Building Mixed Dataset (FinQA + Synthetic)")
    print("="*60)

    # Load and filter FinQA
    finqa_path = DATA_DIR / "train.json"
    if not finqa_path.exists():
        print(f"ERROR: {finqa_path} not found. Run dataset.py first.")
        return

    print("\n[1/4] Loading and filtering FinQA train split...")
    finqa_samples = load_finqa_simple(finqa_path, max_samples=900)
    print(f"  FinQA simple samples: {len(finqa_samples)}")

    # Generate synthetic
    print("\n[2/4] Generating synthetic samples...")
    # n_each_type=140 gives: 140+140+140+70+70 = 560 synthetic
    synthetic_samples = generate_synthetic(n_each_type=140)
    print(f"  Synthetic samples: {len(synthetic_samples)}")
    type_counts = Counter(s["type"] for s in synthetic_samples)
    for t, c in type_counts.items():
        print(f"    {t:<15} {c}")

    # Mix
    print("\n[3/4] Mixing datasets...")
    all_samples = finqa_samples + synthetic_samples
    random.shuffle(all_samples)
    print(f"  Total mixed: {len(all_samples)}")

    # Deduplicate by question hash
    seen = set()
    unique = []
    for s in all_samples:
        key = hashlib.md5(s["question"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"  After dedup: {len(unique)}")

    # 80/20 split
    split = int(len(unique) * 0.8)
    train = unique[:split]
    test  = unique[split:]

    # Save
    print("\n[4/4] Saving...")
    train_path = DATA_DIR / "mixed_train.json"
    test_path  = DATA_DIR / "mixed_test.json"
    stats_path = DATA_DIR / "mixed_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    # Stats
    train_sources = Counter(s["source"] for s in train)
    test_sources  = Counter(s["source"] for s in test)
    stats = {
        "total": len(unique),
        "train": len(train),
        "test":  len(test),
        "train_finqa":     train_sources.get("finqa", 0),
        "train_synthetic": train_sources.get("synthetic", 0),
        "test_finqa":      test_sources.get("finqa", 0),
        "test_synthetic":  test_sources.get("synthetic", 0),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  mixed_train.json : {len(train)} samples")
    print(f"    FinQA:     {train_sources.get('finqa', 0)}")
    print(f"    Synthetic: {train_sources.get('synthetic', 0)}")
    print(f"  mixed_test.json  : {len(test)} samples")
    print(f"    FinQA:     {test_sources.get('finqa', 0)}")
    print(f"    Synthetic: {test_sources.get('synthetic', 0)}")
    print(f"\nSample (train[0]):")
    print(f"  Q: {train[0]['question']}")
    print(f"  A: {train[0]['answer']}")
    print(f"  Source: {train[0]['source']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
