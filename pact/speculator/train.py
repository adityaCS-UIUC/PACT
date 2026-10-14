"""
Phase 3: Fine-tune the speculator using LoRA (PEFT).

Trains a sequence-classification head on top of a frozen base model.
Input  = [PROMPT] ... [THOUGHT] ... [TOOLS] ...
Output = single logit  ->  probability that the thought chunk is "ready
         to execute" (label 1) vs "keep thinking" (label 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from pact.speculator.dataset import build_dataset


def train_speculator(
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    data_path: str = "data/training/speculator_train.jsonl",
    output_dir: str = "checkpoints/speculator",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-4,
    max_length: int = 512,
) -> Path:
    """
    Fine-tune a base model with LoRA for binary classification
    (keep thinking vs ready to execute).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset ---
    ds: DatasetDict = build_dataset(data_path, tokenizer, max_length=max_length)
    print(f"Train: {len(ds['train'])}  Test: {len(ds['test'])}")

    # --- Base model (frozen) + classification head ---
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRA adapter ---
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )

    trainer.train()

    # --- Save adapter ---
    adapter_path = out / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")
    return adapter_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speculator (LoRA)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", default="data/training/speculator_train.jsonl")
    parser.add_argument("--output", default="checkpoints/speculator")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    train_speculator(
        base_model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
    )
