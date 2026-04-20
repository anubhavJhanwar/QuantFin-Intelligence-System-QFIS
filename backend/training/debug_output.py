"""Quick debug: see what base model actually outputs vs reference."""
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
    quantization_config=bnb, device_map="auto", trust_remote_code=True)
model.eval()

with open("data/processed/test.json") as f:
    samples = json.load(f)[:5]

for r in samples:
    prompt = f"Context: {r['context'][:400]}\n\nQuestion: {r['question']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new, skip_special_tokens=True).strip()
    print(f"Q:   {r['question']}")
    print(f"REF: {r['completion']}")
    print(f"OUT: {repr(raw)}")
    print("---")
