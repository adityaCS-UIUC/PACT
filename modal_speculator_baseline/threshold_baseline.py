#!/usr/bin/env python3
"""
Speculator baseline on Modal: score "readiness to call the next tool", sweep a
confidence threshold, plot accuracy vs threshold.

**Classifier mode (default):** the frozen LM encodes each example; a small
**linear classification head** (hidden → 2 logits) outputs a proper probability
for class READY via softmax. Optionally **fine-tune only the head** for a few
steps on the train split (recommended — random head weights are arbitrary).

**Generative mode:** legacy prompt asking for a decimal + regex parse (no extra
layer).

Prerequisites
-------------
1. `pip install -r requirements.txt`
2. `modal setup` and authenticate (Modal token from dashboard).
3. Optional gated models: `modal secret create huggingface HF_TOKEN=...`

Run (from anywhere)
-------------------
  modal run modal_speculator_baseline/threshold_baseline.py -- \\
      --data path/to/tasks_20k_gt_only.jsonl --model phi3 --max-samples 400 \\
      --finetune-head-steps 300

Outputs `accuracy_vs_threshold.png` in the current working directory by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# HuggingFace IDs (swap if you need a specific instruct snapshot)
# ---------------------------------------------------------------------------

MODEL_IDS: dict[str, str] = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


# ---------------------------------------------------------------------------
# Dataset: gold ToolBench-style rows -> (prompt, label) scoring examples
# ---------------------------------------------------------------------------

SCORING_INSTRUCTION = """You evaluate whether an assistant has reasoned enough to execute a specific tool call next.

Output ONLY one decimal number between 0.0 and 1.0 (inclusive), meaning your confidence that the reasoning is sufficient to run the tool with the given arguments. No words, no explanation.

### User task
{user_task}

### Assistant reasoning so far
{reasoning}

### Proposed next tool
name: {tool_name}
arguments (JSON): {args_json}

Confidence (0.0-1.0):"""

# Encoded by the LM + linear head (no numeric generation). Labels: 0 = NOT_READY, 1 = READY.
CLASSIFIER_INPUT_TEMPLATE = """Speculator binary input. Classes: 0 = reasoning not sufficient to run the tool; 1 = sufficient to run it.

### User task
{user_task}

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


def examples_from_jsonl_line(
    row: dict[str, Any],
    text_mode: str = "classifier",
) -> list[tuple[str, int]]:
    """
    Build weak supervision pairs from ground_truth_path (training export):
      label 1: full gold thought before that tool call (ready).
      label 0: shortened thought from same step (premature / incomplete reasoning).
    text_mode: "classifier" uses CLASSIFIER_INPUT_TEMPLATE; "generative" uses SCORING_INSTRUCTION.
    """
    tmpl = CLASSIFIER_INPUT_TEMPLATE if text_mode == "classifier" else SCORING_INSTRUCTION
    prompt = (row.get("prompt") or "").strip()
    path = row.get("ground_truth_path") or []
    tools_block = ""
    avail = row.get("available_tools") or []
    if avail:
        tools_block = "\nAvailable tools (names): " + ", ".join(str(t) for t in avail[:40])
    specs = row.get("api_specs") or []
    if specs:
        tools_block += f"\n({len(specs)} API specs attached in task; omitted here for length.)"

    out: list[tuple[str, int]] = []
    for step in path:
        thought = (step.get("thought") or "").strip()
        action = step.get("action") or ""
        args = step.get("action_input") or {}
        if not thought or not action:
            continue
        args_json = json.dumps(args, ensure_ascii=False)[:800]
        base = tmpl.format(
            user_task=prompt + tools_block,
            reasoning=thought,
            tool_name=action,
            args_json=args_json,
        )
        out.append((base, 1))
        short = _truncate_words(thought, 0.22)
        if short != thought:
            neg = tmpl.format(
                user_task=prompt + tools_block,
                reasoning=short,
                tool_name=action,
                args_json=args_json,
            )
            out.append((neg, 0))
    return out


def load_dataset(
    path: Path,
    max_samples: int | None,
    max_rows: int | None,
    text_mode: str = "classifier",
) -> tuple[list[str], list[int]]:
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
            for t, y in examples_from_jsonl_line(row, text_mode=text_mode):
                texts.append(t)
                labels.append(y)
                if max_samples is not None and len(texts) >= max_samples:
                    return texts, labels
    return texts, labels


def accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> float:
    pred = (scores >= t).astype(np.int32)
    return float((pred == labels).mean())


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, n: int = 51) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, 1.0, n)
    accs = np.array([accuracy_at_threshold(scores, labels, float(t)) for t in ts])
    return ts, accs


def save_plot(ts: np.ndarray, accs: np.ndarray, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(ts, accs, marker="o", markersize=3)
    plt.xlabel("Confidence threshold (predict READY if score ≥ threshold)")
    plt.ylabel("Accuracy on held-out pairs")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Modal GPU worker
# ---------------------------------------------------------------------------

import modal

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "sentencepiece",
        "safetensors",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("speculator-threshold-baseline")


@app.cls(
    gpu="L4",
    image=cuda_image,
    timeout=60 * 60,
    container_idle_timeout=600,
    # Gated models (e.g. Phi-3): `modal secret create huggingface HF_TOKEN=hf_...`
    # then uncomment:
    # secrets=[modal.Secret.from_name("huggingface")],
)
class SpeculatorScorer:
    """
    Frozen causal LM + optional linear head (hidden → 2 classes).
    Class 1 probability = confidence that the reasoning is READY for the tool call.
    """

    model_id: str = "microsoft/Phi-3-mini-4k-instruct"

    @modal.enter()
    def setup(self) -> None:
        import os
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        tok_kw: dict[str, Any] = {"trust_remote_code": True, "token": token}
        if torch.cuda.is_available():
            tok_kw["torch_dtype"] = torch.bfloat16
            tok_kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **tok_kw)
        self._device = next(self.model.parameters()).device
        self._dtype = next(self.model.parameters()).dtype

        hidden = int(self.model.config.hidden_size)
        # Head in float32 for stable CE training; LM stays bf16/fp16.
        self.classifier = nn.Linear(hidden, 2, dtype=torch.float32)
        self.classifier.to(device=self._device, dtype=torch.float32)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _encode_user_text(self, user_text: str, for_generation: bool) -> str:
        messages = [{"role": "user", "content": user_text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=for_generation,
            )
        except Exception:
            return user_text + ("\nConfidence (0.0-1.0):" if for_generation else "")

    def _pooled_hidden(self, prompts: list[str], for_generation: bool) -> Any:
        import torch

        texts = [self._encode_user_text(p, for_generation=for_generation) for p in prompts]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=3072,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
        h = out.hidden_states[-1]  # [B, T, H]
        mask = enc["attention_mask"]
        idx = mask.sum(dim=1) - 1
        b = h.shape[0]
        pooled = h[torch.arange(b, device=h.device), idx]
        return pooled.float()

    def _classify_batch(self, prompts: list[str]) -> Any:
        import torch
        import torch.nn.functional as F

        pooled = self._pooled_hidden(prompts, for_generation=False)
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)[:, 1]
        return probs

    @modal.method()
    def finetune_classifier_head(
        self,
        train_prompts: list[str],
        train_labels: list[int],
        steps: int,
        lr: float,
        batch_size: int,
    ) -> float:
        """Train only the linear head with cross-entropy. Returns mean loss over last epoch."""
        import random
        import torch
        import torch.nn.functional as F

        if steps <= 0 or not train_prompts:
            return 0.0

        self.classifier.train()
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        labels_t = torch.tensor(train_labels, dtype=torch.long, device=self._device)
        n = len(train_prompts)
        losses: list[float] = []

        for _ in range(steps):
            idxs = [random.randrange(n) for _ in range(min(batch_size, n))]
            batch_p = [train_prompts[i] for i in idxs]
            batch_y = labels_t[idxs]
            pooled = self._pooled_hidden(batch_p, for_generation=False)
            logits = self.classifier(pooled)
            loss = F.cross_entropy(logits, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        self.classifier.eval()
        return sum(losses) / max(len(losses), 1)

    @modal.method()
    def score_batch(
        self,
        prompts: list[str],
        batch_size: int = 4,
        score_mode: str = "classifier",
    ) -> list[float]:
        import torch

        if score_mode == "generative":
            return [self._score_one_generative(p) for p in prompts]

        scores: list[float] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            with torch.no_grad():
                probs = self._classify_batch(chunk)
            scores.extend(probs.cpu().tolist())
        return scores

    def _score_one_generative(self, prompt: str) -> float:
        import re
        import torch

        text = self._encode_user_text(prompt, for_generation=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        m = re.search(r"[-+]?\d*\.?\d+", gen.strip())
        if not m:
            return 0.5
        v = float(m.group(0))
        return max(0.0, min(1.0, v))


@app.local_entrypoint()
def main(
    data: str = str(
        Path(__file__).resolve().parents[1] / "data" / "tasks_20k_gt_only.jsonl"
    ),
    model: str = "phi3",
    max_samples: int = 500,
    max_rows: int | None = None,
    batch_size: int = 4,
    plot_path: str = "accuracy_vs_threshold.png",
    val_fraction: float = 0.2,
    dry_run: bool = False,
    score_mode: str = "classifier",
    finetune_head_steps: int = 300,
    finetune_lr: float = 3e-3,
    finetune_batch_size: int = 8,
) -> None:
    """
    Local entrypoint for `modal run ...`.

    score_mode: "classifier" (LM + linear softmax prob for READY) or "generative".
    finetune_head_steps: SGD steps on train split updating **only** the linear head (0 = skip).
    """
    path = Path(data)
    if not path.is_file():
        raise SystemExit(f"Data file not found: {path}")

    key = model.lower().strip()
    if key not in MODEL_IDS:
        raise SystemExit(f"Unknown model {model!r}. Choose from {list(MODEL_IDS)}.")
    model_id = MODEL_IDS[key]

    text_mode = "classifier" if score_mode == "classifier" else "generative"
    texts, labels = load_dataset(
        path, max_samples=max_samples, max_rows=max_rows, text_mode=text_mode
    )
    if len(texts) < 20:
        raise SystemExit(f"Too few examples ({len(texts)}). Lower filters or check JSONL.")

    rng = np.random.default_rng(42)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    val_texts = [texts[i] for i in val_idx]
    val_labels = np.array([labels[i] for i in val_idx], dtype=np.int32)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    print(
        f"Examples: total={len(texts)}  train={len(train_texts)}  val={len(val_texts)}  "
        f"model={model_id}  score_mode={score_mode}"
    )

    if dry_run:
        scores = rng.uniform(0.0, 1.0, size=len(val_texts))
        print("Dry run: random scores (no GPU).")
    else:
        scorer = SpeculatorScorer(model_id=model_id)
        if score_mode == "classifier" and finetune_head_steps > 0 and train_texts:
            loss = scorer.finetune_classifier_head.remote(
                train_texts,
                train_labels,
                finetune_head_steps,
                finetune_lr,
                finetune_batch_size,
            )
            print(f"Head fine-tune mean batch CE loss (last run): {loss:.4f}")
        elif score_mode == "classifier" and finetune_head_steps <= 0:
            print(
                "Warning: classifier head is randomly initialized; "
                "set --finetune-head-steps > 0 for a meaningful baseline."
            )

        scores_list: list[float] = []
        step = 32
        for j in range(0, len(val_texts), step):
            batch = val_texts[j : j + step]
            scores_list.extend(
                scorer.score_batch.remote(
                    batch,
                    batch_size=batch_size,
                    score_mode=score_mode,
                )
            )
        scores = np.array(scores_list, dtype=np.float64)

    ts, accs = sweep_thresholds(scores, val_labels)
    best_i = int(np.argmax(accs))
    print(f"Best threshold ≈ {ts[best_i]:.3f}  accuracy ≈ {accs[best_i]:.4f}")

    out = Path(plot_path)
    title = f"{key} ({score_mode}) — accuracy vs confidence threshold (val)"
    save_plot(ts, accs, out, title=title)
    print(f"Wrote plot: {out.resolve()}")


def cli() -> None:
    """Run without Modal (plot only) or document flags; Modal entrypoint uses kwargs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", choices=list(MODEL_IDS.keys()), default="phi3")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--plot-path", type=Path, default=Path("accuracy_vs_threshold.png"))
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--dry-run", action="store_true", help="Random scores; test plotting only.")
    p.add_argument(
        "--score-mode",
        choices=["classifier", "generative"],
        default="classifier",
        help="classifier: frozen LM + linear head softmax; generative: parse a number.",
    )
    p.add_argument(
        "--finetune-head-steps",
        type=int,
        default=300,
        help="SGD steps on train split for the linear head only (classifier mode). 0 = random head.",
    )
    p.add_argument("--finetune-lr", type=float, default=3e-3)
    p.add_argument("--finetune-batch-size", type=int, default=8)
    p.add_argument(
        "--note",
        action="store_true",
        help="Print how to run with Modal.",
    )
    args = p.parse_args()
    if args.note:
        print(
            "Run on GPU via Modal:\n"
            "  modal run modal_speculator_baseline/threshold_baseline.py -- \\\n"
            f"    --data {args.data} --model {args.model} --max-samples {args.max_samples}\n"
            "Create HF secret if needed:\n"
            "  modal secret create huggingface HF_TOKEN=your_token_here\n"
        )
        return

    # Re-invoke through Modal local entrypoint is awkward from pure CLI; call main logic:
    main(
        data=str(args.data),
        model=args.model,
        max_samples=args.max_samples,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        plot_path=str(args.plot_path),
        val_fraction=args.val_fraction,
        dry_run=args.dry_run,
        score_mode=args.score_mode,
        finetune_head_steps=args.finetune_head_steps,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.finetune_batch_size,
    )


if __name__ == "__main__":
    # Use `modal run ...` for GPU. Plain `python` works if Modal CLI is configured.
    cli()
