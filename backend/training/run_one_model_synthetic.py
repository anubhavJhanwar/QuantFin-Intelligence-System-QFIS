"""
Runs inference for ONE model on the synthetic dataset.
Called by evaluate_synthetic.py as a subprocess.

Usage:
  python run_one_model_synthetic.py --model base --out /tmp/syn_base.json
  python run_one_model_synthetic.py --model ft   --out /tmp/syn_ft.json
"""

import argparse, json, re, string, sys
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
BASE_MODEL  = "microsoft/phi-2"
MAX_NEW_TOK = 50

BOOL_MAP = {
    "true":"yes","false":"no","correct":"yes","incorrect":"no",
    "higher":"yes","lower":"no","greater":"yes","less":"no",
    "increased":"yes","decreased":"no","more":"yes","fewer":"no",
}

def extract_answer(raw: str) -> str:
    raw = raw.strip().split("\n")[0].strip()
    raw = re.sub(r"^answer\s*[:\-]\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^the answer is\s*", "", raw, flags=re.IGNORECASE).strip()
    low = raw.lower().strip()
    if low in BOOL_MAP:
        return BOOL_MAP[low]
    m = re.match(r"^(-?\$?\s*\d[\d,]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", raw, re.IGNORECASE)
    if m:
        val = re.sub(r"[\$,\s]", "", m.group(1)).strip().lower()
        return BOOL_MAP.get(val, val)
    m = re.search(r"(-?\$?\s*\d[\d,]*\.?\d*\s*%?|\byes\b|\bno\b|\btrue\b|\bfalse\b)", raw, re.IGNORECASE)
    if m:
        val = re.sub(r"[\$,\s]", "", m.group(1)).strip().lower()
        return BOOL_MAP.get(val, val)
    first = re.split(r"[.!?]", raw)[0].strip()
    words = first.split()
    return " ".join(words[:10]) if words else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base","ft"], required=True)
    parser.add_argument("--out",   required=True)
    args = parser.parse_args()

    with open(DATA_DIR / "synthetic_test.json", encoding="utf-8") as f:
        test_data = json.load(f)

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
    label = {"base":"Base Phi-2","ft":"Fine-tuned (QLoRA)"}[args.model]

    results = []
    for r in tqdm(test_data, desc=label):
        # Use the stored prompt — already formatted correctly
        prompt = r["prompt"]

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
        new  = out[0][enc["input_ids"].shape[1]:]
        pred = extract_answer(tokenizer.decode(new, skip_special_tokens=True))

        results.append({
            "question":   r["question"],
            "reference":  r["completion"],
            "prediction": pred,
            "type":       r.get("type", "unknown"),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions → {args.out}")


if __name__ == "__main__":
    main()
