# src/sft/train_sft.py
# Baseline SFT on CPU with TinyLlama (Windows + AMD-friendly)
# - No CUDA / bitsandbytes
# - No step-wise eval/save args (avoids version mismatches)
# - Uses LLaMA-style chat template for prompt formatting

from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

DATA_DIR = Path("data/processed/generative")
TRAIN_JS = DATA_DIR / "train.jsonl"
VAL_JS = DATA_DIR / "val.jsonl"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RUN_NAME = "tinyllama_sft_pubmedqa_cpu"
OUT_DIR = Path(f"outputs/sft/{RUN_NAME}")

MAX_LEN = 512
EPOCHS = 1
BSZ = 1
GR_ACC = 16
LR = 2e-4

SYSTEM_PROMPT = (
    "You are a careful medical QA assistant. Only answer using the provided context."
)


def to_chat(example: Dict) -> Dict:
    """Map {question, context, answer} → chat messages list."""
    q = (example.get("question") or "").strip()
    c = (example.get("context") or "").strip()
    a = (example.get("answer") or "").strip()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {q}\n\nContext:\n{c}\n\nAnswer succinctly based only on the context.",
        },
        {"role": "assistant", "content": a},
    ]
    return {"messages": messages}


def tokenize_with_chat_template(batch, tokenizer):
    """Use tokenizer.apply_chat_template → text, then tokenize."""
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch["messages"]
    ]
    return tokenizer(texts, truncation=True, max_length=MAX_LEN)


def main():
    assert TRAIN_JS.exists() and VAL_JS.exists(), "Processed train/val not found."

    # 1) Load data
    ds = load_dataset(
        "json", data_files={"train": str(TRAIN_JS), "validation": str(VAL_JS)}
    )
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)

    # 2) Tokenizer & model (CPU)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)  # CPU
    # LoRA (kept small for CPU training)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    # 3) Tokenize with chat template
    ds_tok = ds.map(
        lambda b: tokenize_with_chat_template(b, tok),
        batched=True,
        remove_columns=["messages"],
    )

    # 4) Trainer
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        run_name=RUN_NAME,
        per_device_train_batch_size=BSZ,
        per_device_eval_batch_size=BSZ,
        gradient_accumulation_steps=GR_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    # 5) Train → Evaluate → Save (manually)
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT_DIR)  # saves LoRA adapter if present
    tok.save_pretrained(OUT_DIR)
    print(f"Saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
