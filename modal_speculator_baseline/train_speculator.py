#!/usr/bin/env python3
"""
Train a frozen causal LM + linear classification head for tool-readiness.

Input format matches ``tasks_20k_gt_only.jsonl``:
  - prompt, available_tools, api_specs, ground_truth_path[{thought, action, action_input}]

The model sees the **full tool suite** (names + compact API docs) plus task, reasoning,
and the **proposed** tool call; it is trained to predict READY (1) vs NOT_READY (0).

Weak negatives: same proposed call with **truncated** reasoning (same idea as the Modal
baseline). Swap in oracle-labeled JSONL later without changing this script.

By default only **5000** (text, label) pairs are drawn with **reservoir sampling**
over the whole JSONL (uniform over all pairs, not just the first tasks). Pass
``--max-examples 0`` to use every pair in the file.

**GPU on Modal:** use ``modal_train_speculator.py`` (see that file's docstring).

Example (local):
  python train_speculator.py \\
    --data ../data/tasks_20k_gt_only.jsonl \\
    --output-dir checkpoints/phi3_speculator \\
    --model microsoft/Phi-3-mini-4k-instruct \\
    --epochs 2 --batch-size 4 --max-length 3072
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


# ---------------------------------------------------------------------------
# Formatting: tool suite + speculator prompt
# ---------------------------------------------------------------------------

TRAINING_TEMPLATE = """Speculator training example. Label semantics: 0 = NOT ready to execute the proposed tool; 1 = READY.

### User task
{user_task}

### Available tools (with API documentation)
{tools_section}

### Assistant reasoning
{reasoning}

### Proposed tool call
name: {tool_name}
arguments (JSON): {args_json}"""


def _truncate_words(text: str, ratio: float) -> str:
    words = text.split()
    if len(words) <= 3:
        return text
    n = max(1, int(len(words) * ratio))
    return " ".join(words[:n])


def format_tools_section(
    row: dict[str, Any],
    max_tools: int,
    desc_max_chars: int,
    schema_max_chars: int,
) -> str:
    """
    Build a text block from ``api_specs`` (preferred) and ``available_tools`` (fallback).
    """
    specs = row.get("api_specs") or []
    names = row.get("available_tools") or []

    if not specs:
        if not names:
            return "(No tools listed.)"
        lines = [f"- {n}" for n in names[:max_tools]]
        if len(names) > max_tools:
            lines.append(f"... and {len(names) - max_tools} more.")
        return "\n".join(lines)

    lines: list[str] = []
    for i, spec in enumerate(specs[:max_tools]):
        if not isinstance(spec, dict):
            continue
        api_name = spec.get("api_name", spec.get("tool_name", f"tool_{i}"))
        desc = (spec.get("api_description") or spec.get("tool_description") or "").strip()
        desc = desc[:desc_max_chars] + ("..." if len(desc) > desc_max_chars else "")
        params = spec.get("parameters") or spec.get("schema") or {}
        try:
            pstr = json.dumps(params, ensure_ascii=False)[:schema_max_chars]
        except (TypeError, ValueError):
            pstr = str(params)[:schema_max_chars]
        if len(json.dumps(params, default=str)) > schema_max_chars:
            pstr += "..."
        lines.append(f"- **{api_name}**\n  Description: {desc}\n  Parameters: {pstr}")

    if len(specs) > max_tools:
        lines.append(f"\n... and {len(specs) - max_tools} more tools omitted.")
    elif names and len(names) > len(specs):
        extra = [n for n in names if n not in {s.get("api_name") for s in specs if isinstance(s, dict)}]
        if extra:
            lines.append("\nAdditional tool names: " + ", ".join(extra[:20]))

    return "\n".join(lines) if lines else "(No tool definitions.)"


def build_examples_from_row(
    row: dict[str, Any],
    negative_truncation_ratio: float,
    max_tools: int,
    desc_max_chars: int,
    schema_max_chars: int,
) -> list[tuple[str, int]]:
    prompt = (row.get("prompt") or "").strip()
    tools_block = format_tools_section(
        row, max_tools=max_tools, desc_max_chars=desc_max_chars, schema_max_chars=schema_max_chars
    )
    user_task = prompt

    path = row.get("ground_truth_path") or []
    out: list[tuple[str, int]] = []

    for step in path:
        thought = (step.get("thought") or "").strip()
        action = (step.get("action") or "").strip()
        args = step.get("action_input") or {}
        if not thought or not action:
            continue
        args_json = json.dumps(args, ensure_ascii=False)[:800]

        pos = TRAINING_TEMPLATE.format(
            user_task=user_task,
            tools_section=tools_block,
            reasoning=thought,
            tool_name=action,
            args_json=args_json,
        )
        out.append((pos, 1))

        short = _truncate_words(thought, negative_truncation_ratio)
        if short != thought:
            neg = TRAINING_TEMPLATE.format(
                user_task=user_task,
                tools_section=tools_block,
                reasoning=short,
                tool_name=action,
                args_json=args_json,
            )
            out.append((neg, 0))

    return out


def load_all_examples(
    path: Path,
    max_rows: int | None,
    max_examples: int | None,
    negative_truncation_ratio: float,
    max_tools: int,
    desc_max_chars: int,
    schema_max_chars: int,
    seed: int,
) -> list[tuple[str, int]]:
    """
    Load (text, label) pairs from JSONL.

    If ``max_examples`` is set, use **reservoir sampling** over the stream so the
    subset is (approximately) uniform over all pairs in the file—not only the first tasks.
    """
    rng = random.Random(seed)

    if max_examples is None:
        texts: list[str] = []
        labels: list[int] = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for t, y in build_examples_from_row(
                    row,
                    negative_truncation_ratio=negative_truncation_ratio,
                    max_tools=max_tools,
                    desc_max_chars=desc_max_chars,
                    schema_max_chars=schema_max_chars,
                ):
                    texts.append(t)
                    labels.append(y)
        return list(zip(texts, labels))

    k = max_examples
    reservoir: list[tuple[str, int]] = []
    n_seen = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for t, y in build_examples_from_row(
                row,
                negative_truncation_ratio=negative_truncation_ratio,
                max_tools=max_tools,
                desc_max_chars=desc_max_chars,
                schema_max_chars=schema_max_chars,
            ):
                n_seen += 1
                if len(reservoir) < k:
                    reservoir.append((t, y))
                else:
                    j = rng.randint(0, n_seen - 1)
                    if j < k:
                        reservoir[j] = (t, y)

    if not reservoir:
        return []
    return reservoir


class SpeculatorTextDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, int]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text, label = self.pairs[idx]
        messages = [{"role": "user", "content": text}]
        try:
            enc_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            enc_text = text

        enc = self.tokenizer(
            enc_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


@dataclass
class Collator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }


class FrozenBackboneClassifier(nn.Module):
    """Last-token pooled hidden state -> 2-class logits (NOT_READY, READY)."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, 2, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        h = out.hidden_states[-1]
        idx = attention_mask.sum(dim=1) - 1
        b = h.size(0)
        pooled = h[torch.arange(b, device=h.device), idx].float()
        return self.classifier(pooled)


def train_loop(
    model: FrozenBackboneClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    *,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> None:
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    n_batches = len(train_loader)
    if n_batches == 0:
        raise ValueError("train_loader is empty")
    accum = max(1, int(gradient_accumulation_steps))
    opt_steps_per_epoch = max(1, (n_batches + accum - 1) // accum)
    total_optimizer_steps = max(1, epochs * opt_steps_per_epoch)
    warmup = max(1, int(total_optimizer_steps * warmup_ratio))
    sched = get_linear_schedule_with_warmup(opt, warmup, total_optimizer_steps)

    for epoch in range(epochs):
        model.classifier.train()
        total_loss = 0.0
        loader_it = iter(train_loader)
        pbar = tqdm(total=n_batches, desc=f"epoch {epoch + 1}/{epochs}")
        while True:
            micro_group: list[dict[str, torch.Tensor]] = []
            for _ in range(accum):
                try:
                    micro_group.append(next(loader_it))
                except StopIteration:
                    break
            if not micro_group:
                break
            g = len(micro_group)
            opt.zero_grad(set_to_none=True)
            last_ce = 0.0
            for batch in micro_group:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, mask)
                ce = F.cross_entropy(logits, labels)
                last_ce = float(ce.detach())
                total_loss += last_ce
                (ce / float(g)).backward()
                pbar.update(1)
                pbar.set_postfix(loss=last_ce)
            nn.utils.clip_grad_norm_(model.classifier.parameters(), max_grad_norm)
            opt.step()
            sched.step()
        pbar.close()

        print(f"  train CE: {total_loss / max(n_batches, 1):.4f}")

        if val_loader is not None:
            acc = evaluate_accuracy(model, val_loader, device)
            print(f"  val accuracy: {acc:.4f}")


@torch.no_grad()
def evaluate_accuracy(
    model: FrozenBackboneClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.classifier.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, mask)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += labels.size(0)
    return correct / max(total, 1)


CHECKPOINT_FORMAT_VERSION = 1


def save_speculator_checkpoint(
    output_dir: Path,
    classifier: nn.Module,
    tokenizer: Any,
    backbone_model_id: str,
    hidden_size: int,
    meta: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    """
    Persist artifacts needed to reload the speculator head + reproduce tokenization.

    - ``speculator_head.pt`` — versioned dict with ``state_dict`` (not raw state_dict alone)
    - ``classifier_head.pt`` — same weights as flat state_dict only (compat for old loaders)
    - ``tokenizer`` files — via ``save_pretrained``
    - ``speculator_meta.json`` / ``training_config.json``
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    state = classifier.state_dict()
    bundled = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "speculator_linear_head",
        "base_model": backbone_model_id,
        "hidden_size": hidden_size,
        "num_labels": 2,
        "label_names": ["NOT_READY", "READY"],
        "state_dict": {k: v.cpu() for k, v in state.items()},
    }
    head_bundled = output_dir / "speculator_head.pt"
    torch.save(bundled, head_bundled)

    head_flat = output_dir / "classifier_head.pt"
    torch.save(state, head_flat)

    tokenizer.save_pretrained(output_dir)

    meta_out = dict(meta)
    meta_out.update(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "head_checkpoint": "speculator_head.pt",
            "head_state_dict_only": "classifier_head.pt",
            "tokenizer_saved": True,
        }
    )
    (output_dir / "speculator_meta.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )

    # Verify bundled file reloads
    try:
        loaded = torch.load(head_bundled, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(head_bundled, map_location="cpu")
    if loaded.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(f"Unexpected checkpoint format: {loaded.get('format_version')}")
    probe = nn.Linear(hidden_size, 2, dtype=torch.float32)
    probe.load_state_dict(loaded["state_dict"])
    for k in state:
        d = (probe.state_dict()[k] - state[k].detach().cpu()).abs().max().item()
        if d > 1e-6:
            raise RuntimeError(f"Checkpoint verify failed for key {k!r} (max diff {d})")

    print(f"Saved bundled checkpoint: {head_bundled}")
    print(f"Saved flat state_dict:    {head_flat}")
    print(f"Saved tokenizer to:       {output_dir}")
    print(f"Saved speculator_meta.json & training_config.json")


@dataclass
class TrainingJobConfig:
    """Arguments for ``run_training`` (local or Modal)."""

    data: Path
    output_dir: Path
    model: str = "microsoft/Phi-3-mini-4k-instruct"
    epochs: int = 2
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    max_length: int = 3072
    val_fraction: float = 0.05
    seed: int = 42
    max_rows: int | None = None
    max_examples: int = 5000  # 0 = use all pairs
    negative_ratio: float = 0.22
    max_tools_in_prompt: int = 24
    tool_desc_chars: int = 280
    tool_schema_chars: int = 400
    hf_token: str | None = None


def run_training(cfg: TrainingJobConfig) -> dict[str, Any]:
    """
    Full training + checkpoint save. Safe to call from Modal (GPU) or locally.

    ``hf_token`` may be omitted if ``HF_TOKEN`` is set in the environment.
    Returns a small summary dict (paths, counts, final val accuracy).
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    token = cfg.hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: training on CPU will be very slow.")

    max_ex = None if cfg.max_examples == 0 else cfg.max_examples

    print("Loading examples...")
    pairs = load_all_examples(
        cfg.data,
        max_rows=cfg.max_rows,
        max_examples=max_ex,
        negative_truncation_ratio=cfg.negative_ratio,
        max_tools=cfg.max_tools_in_prompt,
        desc_max_chars=cfg.tool_desc_chars,
        schema_max_chars=cfg.tool_schema_chars,
        seed=cfg.seed,
    )
    if len(pairs) < 32:
        raise ValueError(f"Too few examples: {len(pairs)}")
    print(f"Using {len(pairs)} labeled pairs (cap={'all' if max_ex is None else max_ex}).")

    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        trust_remote_code=True,
        token=token,
        torch_dtype=dtype,
        device_map=None,
    )
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    hidden = int(backbone.config.hidden_size)
    model = FrozenBackboneClassifier(backbone, hidden).to(device)
    model.classifier.to(dtype=torch.float32)

    train_ds = SpeculatorTextDataset(train_pairs, tokenizer, cfg.max_length)
    val_ds = SpeculatorTextDataset(val_pairs, tokenizer, cfg.max_length)
    collator = Collator()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print(
        f"Training classifier head (hidden={hidden}) on {device}... "
        f"grad_accum={cfg.gradient_accumulation_steps} max_grad_norm={cfg.max_grad_norm}"
    )
    train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
    )

    final_val_acc = evaluate_accuracy(model, val_loader, device)
    print(f"Final val accuracy: {final_val_acc:.4f}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    training_config = {
        "data": str(cfg.data.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "model": cfg.model,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_grad_norm": cfg.max_grad_norm,
        "max_length": cfg.max_length,
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "max_rows": cfg.max_rows,
        "max_examples_requested": cfg.max_examples,
        "max_examples_effective": max_ex,
        "pair_sampling": "reservoir" if max_ex is not None else "all_pairs",
        "negative_ratio": cfg.negative_ratio,
        "max_tools_in_prompt": cfg.max_tools_in_prompt,
        "tool_desc_chars": cfg.tool_desc_chars,
        "tool_schema_chars": cfg.tool_schema_chars,
        "final_val_accuracy": final_val_acc,
    }

    meta = {
        "base_model": cfg.model,
        "hidden_size": hidden,
        "num_labels": 2,
        "label_map": {"0": "NOT_READY", "1": "READY"},
        "max_length": cfg.max_length,
        "training_template": "TRAINING_TEMPLATE_v1",
        "negative_truncation_ratio": cfg.negative_ratio,
        "max_tools_in_prompt": cfg.max_tools_in_prompt,
        "tool_desc_chars": cfg.tool_desc_chars,
        "tool_schema_chars": cfg.tool_schema_chars,
        "total_pairs_used": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "pooling": "last_non_pad_token_last_hidden_layer",
        "final_val_accuracy": final_val_acc,
    }

    save_speculator_checkpoint(
        cfg.output_dir,
        model.classifier,
        tokenizer,
        backbone_model_id=cfg.model,
        hidden_size=hidden,
        meta=meta,
        training_config=training_config,
    )

    return {
        "output_dir": str(cfg.output_dir.resolve()),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "final_val_accuracy": final_val_acc,
        "hidden_size": hidden,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="JSONL e.g. tasks_20k_gt_only.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Micro-batches per optimizer step (effective batch ≈ batch_size × this).",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip classifier gradients before each optimizer step.",
    )
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None, help="Cap JSONL lines (debug).")
    p.add_argument(
        "--max-examples",
        type=int,
        default=5000,
        help="Max (text, label) pairs to use (default 5000 from ~20K). Use 0 for no cap.",
    )
    p.add_argument("--negative-ratio", type=float, default=0.22, help="Word-fraction for weak negatives.")
    p.add_argument("--max-tools-in-prompt", type=int, default=24)
    p.add_argument("--tool-desc-chars", type=int, default=280)
    p.add_argument("--tool-schema-chars", type=int, default=400)
    p.add_argument("--hf-token", type=str, default=None, help="Or set HF_TOKEN env.")
    args = p.parse_args()

    cfg = TrainingJobConfig(
        data=args.data,
        output_dir=args.output_dir,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_rows=args.max_rows,
        max_examples=args.max_examples,
        negative_ratio=args.negative_ratio,
        max_tools_in_prompt=args.max_tools_in_prompt,
        tool_desc_chars=args.tool_desc_chars,
        tool_schema_chars=args.tool_schema_chars,
        hf_token=args.hf_token,
    )
    try:
        run_training(cfg)
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
