"""
Generates a synthetic financial QA dataset designed to test
single-step arithmetic reasoning — the exact capability that
QLoRA fine-tuning teaches Phi-2.

Question types:
  1. Direct extraction   — "what was X in year Y?"
  2. Difference          — "what was the change in X from Y1 to Y2?"
  3. Percentage change   — "what was the % change in X from Y1 to Y2?"
  4. Ratio               — "what is the ratio of X to Y?"
  5. Simple sum          — "what is the total of X + Y + Z?"

All answers are exact, single numbers or percentages.
Context always contains the numbers needed — no truncation issues.
"""

import json, random, math
from pathlib import Path

random.seed(42)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = [
    "Acme Corp", "GlobalTech Inc", "Summit Financial", "Apex Industries",
    "Crestview Holdings", "Meridian Group", "Pinnacle Corp", "Vantage Capital",
    "Horizon Enterprises", "Sterling Partners",
]
METRICS = ["revenue", "net income", "operating profit", "gross profit",
           "total assets", "operating expenses", "net sales", "EBITDA",
           "cash flow from operations", "total liabilities"]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def fmt(n):
    """Format a number cleanly — no trailing zeros."""
    if isinstance(n, float) and n == int(n):
        return str(int(n))
    return f"{round(n, 2)}"


def pct(n):
    """Format as percentage string."""
    v = round(n, 2)
    if v == int(v):
        return f"{int(v)}%"
    return f"{v}%"


samples = []


# ── Type 1: Direct extraction ──────────────────────────────────────────────────
for _ in range(50):
    company = random.choice(COMPANIES)
    metric  = random.choice(METRICS)
    year    = random.choice(YEARS)
    value   = round(random.uniform(10, 5000), 1)
    unit    = random.choice(["million", "billion"])
    unit_m  = 1 if unit == "million" else 1000

    context = (
        f"{company} reported {metric} of {fmt(value)} {unit} in {year}. "
        f"The company continued to expand its operations across multiple segments."
    )
    question = f"what was the {metric} of {company} in {year} in {unit}?"
    answer   = fmt(value)

    samples.append({
        "question":   question,
        "answer":     answer,
        "context":    context,
        "prompt":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "completion": answer,
        "type":       "extraction",
    })


# ── Type 2: Difference ─────────────────────────────────────────────────────────
for _ in range(50):
    company = random.choice(COMPANIES)
    metric  = random.choice(METRICS)
    y1, y2  = sorted(random.sample(YEARS, 2))
    v1      = round(random.uniform(50, 3000), 1)
    v2      = round(v1 + random.uniform(-500, 500), 1)
    diff    = round(v2 - v1, 1)
    unit    = "million"

    context = (
        f"{company} reported {metric} of {fmt(v1)} {unit} in {y1} "
        f"and {fmt(v2)} {unit} in {y2}."
    )
    question = f"what was the change in {metric} from {y1} to {y2} in {unit}?"
    answer   = fmt(diff)

    samples.append({
        "question":   question,
        "answer":     answer,
        "context":    context,
        "prompt":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "completion": answer,
        "type":       "difference",
    })


# ── Type 3: Percentage change ──────────────────────────────────────────────────
for _ in range(50):
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
    answer   = pct(pct_val)

    samples.append({
        "question":   question,
        "answer":     answer,
        "context":    context,
        "prompt":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "completion": answer,
        "type":       "pct_change",
    })


# ── Type 4: Ratio ──────────────────────────────────────────────────────────────
for _ in range(25):
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

    samples.append({
        "question":   question,
        "answer":     answer,
        "context":    context,
        "prompt":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "completion": answer,
        "type":       "ratio",
    })


# ── Type 5: Simple sum ─────────────────────────────────────────────────────────
for _ in range(25):
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

    samples.append({
        "question":   question,
        "answer":     answer,
        "context":    context,
        "prompt":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        "completion": answer,
        "type":       "sum",
    })


# ── Shuffle and split 80/20 train/test ────────────────────────────────────────
random.shuffle(samples)
split = int(len(samples) * 0.8)
train = samples[:split]
test  = samples[split:]

out_train = PROCESSED_DIR / "synthetic_train.json"
out_test  = PROCESSED_DIR / "synthetic_test.json"

with open(out_train, "w", encoding="utf-8") as f:
    json.dump(train, f, indent=2, ensure_ascii=False)

with open(out_test, "w", encoding="utf-8") as f:
    json.dump(test, f, indent=2, ensure_ascii=False)

print(f"Generated {len(samples)} synthetic samples")
print(f"  Train: {len(train)} → {out_train}")
print(f"  Test:  {len(test)}  → {out_test}")
print(f"\nType breakdown:")
from collections import Counter
counts = Counter(s["type"] for s in samples)
for t, c in counts.items():
    print(f"  {t:<15} {c}")
print(f"\nSample:")
print(f"  Q: {test[0]['question']}")
print(f"  A: {test[0]['answer']}")
print(f"  C: {test[0]['context']}")
