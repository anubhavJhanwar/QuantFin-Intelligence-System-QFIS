"""
QLoRA Fine-tuning pipeline for QFIS.
Model: microsoft/phi-2 (2.7B) with 4-bit quantization via bitsandbytes.
Saves adapter weights to /models/v1 — run ONCE, load forever.
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # backend/training/ → backend/
DATA_DIR = ROOT / "data" / "processed"
# Fallback: data might be at project root
if not DATA_DIR.exists():
    DATA_DIR = ROOT.parent / "data" / "processed"
MODEL_SAVE_DIR = ROOT.parent / "models" / "v1"
CHECKPOINT_DIR = ROOT.parent / "models" / "checkpoints"
LOG_DIR = ROOT.parent / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "finetune.log", rotation="10 MB")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/phi-2"
MAX_SEQ_LEN = 256        # halved → 4x faster per step
BATCH_SIZE = 4           # larger batch → fewer steps
GRAD_ACCUM = 2           # effective batch = 8
EPOCHS = 3
LR = 2e-4
LORA_R = 16              # higher rank → more capacity
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_TRAIN_SAMPLES = 2500  # more samples → better generalization


def load_split(name: str) -> list:
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(records: list, tokenizer) -> Dataset:
    """Build tokenized HuggingFace Dataset for causal LM.
    Only the completion (answer) tokens are supervised — prompt tokens get label=-100.
    """
    def tokenize(batch):
        prompts     = batch["prompt"]
        completions = batch["completion"]

        full_texts = [p + " " + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
        enc = tokenizer(full_texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

        labels = []
        for i, (prompt, ids) in enumerate(zip(prompts, enc["input_ids"])):
            # Find where the prompt ends so we only supervise the answer
            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
            prompt_len = len(prompt_ids)
            label = [-100] * prompt_len + ids[prompt_len:]
            # Pad labels to match full length
            label = label[:MAX_SEQ_LEN]
            label += [-100] * (MAX_SEQ_LEN - len(label))
            labels.append(label)

        enc["labels"] = labels
        return enc

    ds = Dataset.from_dict({"prompt": [r["prompt"] for r in records],
                             "completion": [r["completion"] for r in records]})
    return ds.map(tokenize, batched=True, remove_columns=["prompt", "completion"])


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """
    LoRA config for Phi-2.
    Targets q_proj and v_proj — most impactful attention layers.
    r=16 gives good capacity without OOM on 4GB VRAM.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"],
        bias="none",
        inference_mode=False,
    )


def train():
    logger.info("=" * 60)
    logger.info("QFIS — QLoRA Fine-tuning on Phi-2")
    logger.info("=" * 60)

    # ── Skip if already trained ────────────────────────────────────────────────
    if (MODEL_SAVE_DIR / "adapter_config.json").exists():
        logger.info(f"Fine-tuned model already exists at {MODEL_SAVE_DIR}. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ───────────────────────────────────────────────────────────────────
    logger.info("Loading dataset splits...")
    train_records = load_split("train")
    val_records = load_split("val")
    # Cap to MAX_TRAIN_SAMPLES for feasible training time on RTX 2050
    train_records = train_records[:MAX_TRAIN_SAMPLES]
    val_records = val_records[:300]
    logger.info(f"Train: {len(train_records)} | Val: {len(val_records)}")

    train_ds = build_dataset(train_records, tokenizer)
    val_ds = build_dataset(val_records, tokenizer)

    # ── Model with 4-bit quantization ─────────────────────────────────────────
    logger.info("Loading Phi-2 with 4-bit quantization (QLoRA)...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir=str(LOG_DIR),
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device == "cuda"),
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        report_to="none",
        dataloader_num_workers=0,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting QLoRA training...")
    trainer.train()

    # ── Save adapter weights ───────────────────────────────────────────────────
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))

    # Save training metadata
    meta = {
        "base_model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "quantization": "4-bit NF4",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
    }
    with open(MODEL_SAVE_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✓ Model saved to {MODEL_SAVE_DIR}")
    logger.info("Training complete. Run evaluate.py next.")


if __name__ == "__main__":
    train()
