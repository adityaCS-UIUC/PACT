#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) for tool-call prediction.

Given the task prompt + cumulative thought of an agent that is already READY
to call a tool, predict exactly which tool to call and with what arguments.

Data format: JSONL where each row contains:
  - prompt             : the original user task
  - cumulative_thought : the agent's reasoning so far
  - target_tool_call   : {"tool_name": str, "arguments": dict}
  - ready_label        : 1  (only READY rows are used)

Training objective:
  Input  = task prompt  +  cumulative_thought
  Output = "#TOOL[tool_name]{JSON}"

Including the task prompt is important because argument values (usernames, IDs,
keywords, etc.) are often stated explicitly in the original request but only
paraphrased — or omitted entirely — in the cumulative thought.

Loss is masked so only the output tokens contribute (standard SFT loss masking).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


OUTPUT_PREFIX = "\n\n### Tool Call\n"


def _format_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    args_json = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
    return f"#TOOL[{tool_name}]{args_json}"


def format_toolcall_example(row: dict[str, Any]) -> tuple[str, str] | None:
    """Return (input, output) or None if the row should be skipped."""
    if int(row.get("ready_label", row.get("label", 0))) != 1:
        return None

    tc = row.get("target_tool_call") or {}
    tool_name = (tc.get("tool_name") or "").strip()
    if not tool_name:
        return None

    arguments = tc.get("arguments") or {}
    task_prompt = (row.get("prompt") or "").strip()
    thought = (row.get("cumulative_thought") or "").strip()

    inp = (
        "### Task\n"
        f"{task_prompt}\n\n"
        "### Thought\n"
        f"{thought}\n\n"
        "The agent is READY. Output the tool call."
    )
    out = _format_tool_call(tool_name, arguments)
    return inp, out


class ToolCallPredDataset(Dataset):
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

        examples: list[tuple[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                result = format_toolcall_example(json.loads(line))
                if result is not None:
                    examples.append(result)

        if shuffle:
            rng.shuffle(examples)
        self.examples = examples
        print(f"ToolCallPredDataset: loaded {len(examples)} READY examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        inp, out = self.examples[idx]
        prefix = inp + OUTPUT_PREFIX

        # Tokenize without padding — DataCollatorForSeq2Seq handles per-batch padding.
        prefix_ids = self.tokenizer(
            prefix,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        full_ids = self.tokenizer(
            prefix + out,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = full_ids.clone()
        labels[: len(prefix_ids)] = -100   # mask loss on the input/prefix portion

        return {"input_ids": full_ids, "labels": labels}


@dataclass
class ToolCallPredConfig:
    data: Path
    output_dir: Path
    model: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 768          # prompt(~150t) + thought(~300t) + tool-call output
    seed: int = 42
    val_fraction: float = 0.05
    max_rows: int | None = None

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 1e-4
    warmup_ratio: float = 0.1    # more warmup for smaller dataset (~17k examples)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 2              # 2 epochs: more beneficial with ~17k examples

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    hf_token: str | None = None


def run_toolcall_pred_sft(cfg: ToolCallPredConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    token = cfg.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds_all = ToolCallPredDataset(
        cfg.data,
        tokenizer,
        cfg.max_length,
        max_rows=cfg.max_rows,
        seed=cfg.seed,
        shuffle=True,
    )
    if len(ds_all) < 32:
        raise ValueError(f"Too few READY examples: {len(ds_all)}")

    n_val = max(1, int(len(ds_all) * float(cfg.val_fraction)))
    ds_val = torch.utils.data.Subset(ds_all, list(range(n_val)))
    ds_train = torch.utils.data.Subset(ds_all, list(range(n_val, len(ds_all))))
    print(f"Train: {len(ds_train)}  Val: {len(ds_val)}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        trust_remote_code=True,
        token=token,
        torch_dtype=dtype,
        device_map=None,
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
            # Phi-3-mini fuses Q/K/V into qkv_proj (no separate q_proj/v_proj)
            target_modules=["qkv_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

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
        metric_for_best_model="eval_loss",
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
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
        "max_length": cfg.max_length,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-rows", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--hf-token", type=str, default=None)
    args = p.parse_args()

    cfg = ToolCallPredConfig(
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
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=args.hf_token,
    )
    print(run_toolcall_pred_sft(cfg))


if __name__ == "__main__":
    main()
