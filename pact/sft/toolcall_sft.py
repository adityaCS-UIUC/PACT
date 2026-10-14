#!/usr/bin/env python3
"""
Classification-style SFT for tool-call readiness (READY / NOT_READY).

Training objective — first-token binary classification:
  1. Forward-pass the INPUT only (thought + prompt prefix, no output appended).
  2. Read logits at the last real input position (= the model's prediction of the
     next token before it has seen any output).
  3. Extract logits for just the two target tokens: READY_tok and NOT_READY_tok.
  4. Apply cross-entropy over these two logits — this is a proper binary classifier.
  5. Auto-computed class weights upweight the minority class (READY, ~38%) to
     counteract the 1.6x NOT_READY majority.

Why this is better than generation loss:
  - Generation loss spreads signal across all tokens of the output string; the
    first token (which carries the full decision) is just one of many.
  - "NOT_READY" is 3 tokens ("NOT", "_", "READY") vs 1 for "READY" — more loss
    signal per NOT_READY example, creating implicit class imbalance.
  - Classification loss focuses 100% of the gradient on the decision token,
    gives identical loss magnitude per example regardless of label, and the
    class weights fix the remaining imbalance.

Inference (unchanged):
  Forward-pass the input, read P(READY) from first-token logits and threshold-sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


OUTPUT_PREFIX = "\n\n### Decision\n"


# ---------------------------------------------------------------------------
# Input / label formatting  (unchanged — used by eval scripts too)
# ---------------------------------------------------------------------------
def format_sft_example(row: dict[str, Any]) -> tuple[str, str]:
    thought = (row.get("cumulative_thought") or "").strip()

    judge_raw = (row.get("judge_raw") or "").strip().upper()
    if judge_raw in ("READY", "NOT_READY"):
        out = judge_raw
    else:
        ready = int(row.get("ready_label", row.get("label", 0)))
        out = "READY" if ready == 1 else "NOT_READY"

    inp = (
        "### Thought\n"
        f"{thought}\n\n"
        "Is the agent ready to call a tool? Answer READY or NOT_READY."
    )
    return inp, out


# ---------------------------------------------------------------------------
# Dataset — returns INPUT ONLY + integer class label (no output appended)
# ---------------------------------------------------------------------------
class JsonlToolcallDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
        *,
        max_rows: int | None = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        rng = random.Random(seed)

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if shuffle:
            rng.shuffle(rows)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inp, out = format_sft_example(self.rows[idx])
        # Only the input goes into the model — no output tokens appended.
        prefix = inp + OUTPUT_PREFIX
        enc = self.tokenizer(
            prefix,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        # class_label: 1 = READY, 0 = NOT_READY
        class_label = torch.tensor(1 if out == "READY" else 0, dtype=torch.long)
        return {"input_ids": input_ids, "class_label": class_label}


# ---------------------------------------------------------------------------
# Collator — pads input_ids to batch-max length, stacks class_labels
# ---------------------------------------------------------------------------
@dataclass
class _BinaryCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        class_labels = torch.stack([f["class_label"] for f in features])
        input_features = [{"input_ids": f["input_ids"]} for f in features]
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["class_label"] = class_labels
        return batch


# ---------------------------------------------------------------------------
# Custom Trainer — classification loss on first output token logits
# ---------------------------------------------------------------------------
class BinaryReadinessTrainer(Trainer):
    """Trains a causal LM as a binary classifier via first-token logits.

    For each example the model sees only the input.  The logit at the last
    real input position predicts the next (first output) token.  We extract
    the two decision-token logits and apply weighted cross-entropy.

    Class layout:
      index 0 → NOT_READY token
      index 1 → READY token
    """

    def __init__(
        self,
        *args,
        not_ready_tok_id: int,
        ready_tok_id: int,
        class_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.not_ready_tok_id = not_ready_tok_id
        self.ready_tok_id = ready_tok_id
        self.class_weight = class_weight  # shape [2]: [w_not_ready, w_ready]

    def _binary_logits(self, model_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract [logit_NOT_READY, logit_READY] at last real position."""
        last_pos = attention_mask.sum(dim=1) - 1              # [B]
        last_logits = model_logits[torch.arange(model_logits.shape[0]), last_pos]  # [B, vocab]
        return last_logits[:, [self.not_ready_tok_id, self.ready_tok_id]]          # [B, 2]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        class_labels = inputs.pop("class_label")              # [B]  0=NOT_READY, 1=READY
        outputs = model(**inputs)
        binary_logits = self._binary_logits(outputs.logits, inputs["attention_mask"])

        weight = (
            self.class_weight.to(binary_logits.device)
            if self.class_weight is not None
            else None
        )
        loss = F.cross_entropy(binary_logits, class_labels, weight=weight)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            class_labels = inputs.pop("class_label")
            outputs = model(**inputs)
            binary_logits = self._binary_logits(outputs.logits, inputs["attention_mask"])
            weight = (
                self.class_weight.to(binary_logits.device)
                if self.class_weight is not None
                else None
            )
            loss = F.cross_entropy(binary_logits, class_labels, weight=weight)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, binary_logits.float(), class_labels)


# ---------------------------------------------------------------------------
# Metrics logged during training
# ---------------------------------------------------------------------------
def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred          # logits: [N, 2]  labels: [N]
    preds = np.argmax(logits, axis=-1)  # 0=NOT_READY, 1=READY
    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SftConfig:
    data: Path
    output_dir: Path
    model: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 512
    seed: int = 42
    val_fraction: float = 0.05
    max_rows: int | None = None

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4    # effective batch = 16
    lr: float = 5e-5                        # conservative — classification loss converges fast
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 1

    use_class_weight: bool = True           # auto-upweight minority READY class
    use_lora: bool = True
    lora_r: int = 32                        # bumped from 16 for better classifier capacity
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    hf_token: str | None = None


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def run_sft(cfg: SftConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    token = cfg.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # decoder-only: pad on the left

    # ── Find decision token IDs ─────────────────────────────────────────────
    not_ready_tok = tokenizer.encode("NOT_READY", add_special_tokens=False)[0]
    ready_tok     = tokenizer.encode("READY",     add_special_tokens=False)[0]
    print(f"Decision token IDs — NOT_READY: {not_ready_tok}  READY: {ready_tok}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    ds_all = JsonlToolcallDataset(
        cfg.data, tokenizer, cfg.max_length,
        max_rows=cfg.max_rows, seed=cfg.seed, shuffle=True,
    )
    if len(ds_all) < 32:
        raise ValueError(f"Too few examples: {len(ds_all)}")

    n_val = max(1, int(len(ds_all) * cfg.val_fraction))
    ds_val   = torch.utils.data.Subset(ds_all, list(range(n_val)))
    ds_train = torch.utils.data.Subset(ds_all, list(range(n_val, len(ds_all))))

    # ── Auto class weights ──────────────────────────────────────────────────
    class_weight: torch.Tensor | None = None
    if cfg.use_class_weight:
        train_labels = [ds_all.rows[i].get("ready_label", ds_all.rows[i].get("label", 0))
                        for i in range(n_val, len(ds_all))]
        n_ready     = sum(int(l) == 1 for l in train_labels)
        n_not_ready = len(train_labels) - n_ready
        w_not_ready = 1.0
        w_ready     = n_not_ready / n_ready if n_ready else 1.0
        class_weight = torch.tensor([w_not_ready, w_ready], dtype=torch.float32)
        print(f"Class weights — NOT_READY: {w_not_ready:.3f}  READY: {w_ready:.3f}  "
              f"(train: {n_not_ready} NOT_READY, {n_ready} READY)")

    print(f"Train: {len(ds_train)}  Val: {len(ds_val)}")

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, trust_remote_code=True, token=token,
        torch_dtype=dtype, device_map=None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.use_lora:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but use_lora=True")
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["qkv_proj", "o_proj"],  # Phi-3-mini fused QKV
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",     # save best F1 checkpoint, not just loss
        greater_is_better=True,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,    # keep class_label in batch
    )

    collator = _BinaryCollator(tokenizer, pad_to_multiple_of=8)

    trainer = BinaryReadinessTrainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        not_ready_tok_id=not_ready_tok,
        ready_tok_id=ready_tok,
        class_weight=class_weight,
    )
    trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    return {
        "output_dir": str(cfg.output_dir.resolve()),
        "train_examples": len(ds_train),
        "val_examples": len(ds_val),
        "model": cfg.model,
        "use_lora": cfg.use_lora,
        "lora_r": cfg.lora_r,
        "max_length": cfg.max_length,
        "class_weight": class_weight.tolist() if class_weight is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-rows", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--hf-token", type=str, default=None)
    args = p.parse_args()

    cfg = SftConfig(
        data=args.data,
        output_dir=args.output_dir,
        model=args.model,
        max_length=args.max_length,
        epochs=args.epochs,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_rows=None if args.max_rows < 0 else args.max_rows,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=not args.no_class_weight,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=args.hf_token,
    )
    print(run_sft(cfg))


if __name__ == "__main__":
    main()
