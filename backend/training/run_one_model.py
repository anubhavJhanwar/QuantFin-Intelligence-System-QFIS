"""
Runs inference for ONE model, saves predictions to a JSON file, then exits.
Called by evaluate.py as a subprocess — guarantees full VRAM release between models.

Usage:
  python run_one_model.py --model base    --out /tmp/base_preds.json
  python run_one_model.py --model pe      --out /tmp/pe_preds.json
  python run_one_model.py --model ft      --out /tmp/ft_preds.json
"""

import argparse, gc, json, re, string, sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "processed"
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_DIR = ROOT.parent / "models" / "v1"
BASE_MODEL = "microsoft/phi-2"
MAX_NEW_TOK = 50
MAX_EVAL    = 40

SYSTEM_PROMPT = (
    "You are a financial analyst. "
    "Answer using only the context. "
    "Give a short precise answer — a number, percentage, or yes/no.\n\n"
)

BOOL_MAP = {
    "true":"yes","false":"no","correct":"yes","incorrect":"no",
    "higher":"yes","lower":"no","greater":"yes","less":"no",
    "increased":"yes","decreased":"no","more":"yes","fewer":"no",
}

def extract_answer(raw: str) -> str:
    # Clean escape sequences
    raw = re.sub(r"\\+n", " ", raw).strip()
    raw = raw.strip().split("\n")[0].strip()
    raw = re.sub(r"^answer\s*[:\-]\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^the answer is\s*", "", raw, flags=re.IGNORECASE).strip()
    # Check for boolean synonyms first
    low = raw.lower().strip()
    if low in BOOL_MAP:
        return BOOL_MAP[low]
    # Prefer number/percentage/yes/no at start
    m = re.match(r"^(-?\$?\s*\d[\d,]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", raw, re.IGNORECASE)
    if m:
        val = re.sub(r"[\$,\s]", "", m.group(1)).strip().lower()
        return BOOL_MAP.get(val, val)
    # Search anywhere in the string for a number/percentage/yes/no
    m = re.search(r"(-?\$?\s*\d[\d,]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", raw, re.IGNORECASE)
    if m:
        val = re.sub(r"[\$,\s]", "", m.group(1)).strip().lower()
        return BOOL_MAP.get(val, val)
    # Take first sentence, max 10 words
    first = re.split(r"[.!?]", raw)[0].strip()
    words = first.split()
    return " ".join(words[:10]) if words else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base","pe","ft"], required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    with open(DATA_DIR / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)[:MAX_EVAL]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )

    if args.model == "ft":
        # Patch adapter_config for local PEFT compat
        cfg_path = MODEL_DIR / "adapter_config.json"
        cfg = json.loads(cfg_path.read_text())
        for k in ["eva_config","lora_bias","exclude_modules",
                  "layer_replication","use_dora","use_rslora"]:
            cfg.pop(k, None)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        model = PeftModel.from_pretrained(base, str(MODEL_DIR))
    else:
        model = base

    model.eval()
    label = {"base":"Base Phi-2","pe":"Prompt-Engineered","ft":"Fine-tuned (QLoRA)"}[args.model]

    results = []
    for r in tqdm(test_data, desc=label):
        ctx    = r["context"][:900]
        prefix = SYSTEM_PROMPT if args.model == "pe" else ""
        prompt = f"{prefix}Context: {ctx}\n\nQuestion: {r['question']}\n\nAnswer:"

        dev = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOK, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )
        new = out[0][enc["input_ids"].shape[1]:]
        pred = extract_answer(tokenizer.decode(new, skip_special_tokens=True))

        results.append({
            "question":   r["question"],
            "reference":  r["completion"],
            "prediction": pred,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions → {args.out}")


if __name__ == "__main__":
    main()
