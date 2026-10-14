#!/usr/bin/env python3
"""
Evaluate base (non-finetuned) Phi-3-mini on the test set for READY/NOT_READY prediction.

For each row in data/test.jsonl:
  - Input : cumulative_thought  (same format as SFT training)
  - Output: P(READY) and P(NOT_READY) from the first output token's logits
  - Saves a results JSONL to Modal Volume `pact-eval-results`

This gives the zero-shot baseline to compare against the fine-tuned checkpoint.

Run (from repo root CS-598/):
    modal run modal_phi3_baseline_eval/modal_eval_phi3_base.py

Optional flags:
    --model           HF model id  (default: microsoft/Phi-3-mini-4k-instruct)
    --test-file       JSONL filename under data/  (default: test.jsonl)
    --batch-size      Inference batch size        (default: 16)
    --max-length      Max tokens per example      (default: 512)
    --run-name        Output name in the volume   (default: phi3_base_<timestamp>)

Results are written to volume `pact-eval-results` at /<run_name>/predictions.jsonl
Fetch locally with:
    modal volume get pact-eval-results <run_name> ./eval_results/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
_BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "safetensors",
        "tqdm>=4.66.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)


def _build_image() -> modal.Image:
    if not modal.is_local():
        return _BASE_IMAGE
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"data/ directory not found at {data_dir}")
    return _BASE_IMAGE.add_local_dir(data_dir, remote_path="/mnt/data")


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
_EVAL_VOL = modal.Volume.from_name("pact-eval-results", create_if_missing=True)

app = modal.App("phi3-base-eval")

# ---------------------------------------------------------------------------
# Input formatting  (must match SFT training exactly)
# ---------------------------------------------------------------------------
_OUTPUT_PREFIX = "\n\n### Decision\n"


def _format_input(row: dict) -> str:
    thought = (row.get("cumulative_thought") or "").strip()
    return (
        "### Thought\n"
        f"{thought}\n\n"
        "Is the agent ready to call a tool? Answer READY or NOT_READY."
        + _OUTPUT_PREFIX
    )


def _true_label(row: dict) -> str:
    judge_raw = (row.get("judge_raw") or "").strip().upper()
    if judge_raw in ("READY", "NOT_READY"):
        return judge_raw
    return "READY" if int(row.get("ready_label", row.get("label", 0))) == 1 else "NOT_READY"


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",           # A10G (24 GB) is sufficient for inference on 3.8B model
    image=_build_image(),
    timeout=4 * 3600,
    volumes={"/eval_results": _EVAL_VOL},
    # Uncomment if Phi-3 is gated on your HF account:
    # secrets=[modal.Secret.from_name("huggingface")],
)
def eval_on_modal(
    *,
    test_file: str,
    run_name: str,
    model_id: str,
    batch_size: int,
    max_length: int,
) -> dict:
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda")

    # ── Load model & tokenizer ──────────────────────────────────────────────
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # for decoder-only batch inference

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # ── Find first-token IDs for "READY" and "NOT_READY" ───────────────────
    # We compare only the very first generated token's logit as a confidence score.
    ready_tok = tokenizer.encode("READY", add_special_tokens=False)[0]
    not_ready_tok = tokenizer.encode("NOT_READY", add_special_tokens=False)[0]
    print(f"Token IDs  →  READY: {ready_tok}  NOT_READY: {not_ready_tok}")

    # ── Load test data ──────────────────────────────────────────────────────
    test_path = Path("/mnt/data") / test_file
    if not test_path.is_file():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    rows: list[dict] = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} test rows from {test_file}")

    # ── Batched inference ───────────────────────────────────────────────────
    results: list[dict] = []

    @torch.no_grad()
    def _predict_batch(batch_rows: list[dict]) -> list[dict]:
        prompts = [_format_input(r) for r in batch_rows]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc)
        logits = outputs.logits  # [B, seq_len, vocab]

        # Last real token position per example (predicts the next token)
        last_pos = enc["attention_mask"].sum(dim=1) - 1  # [B]
        last_logits = logits[torch.arange(len(batch_rows)), last_pos]  # [B, vocab]

        # Softmax over just the two target tokens → confidence
        target_logits = last_logits[:, [ready_tok, not_ready_tok]]  # [B, 2]
        probs = target_logits.softmax(dim=-1).cpu().tolist()

        batch_results = []
        for row, (p_ready, p_not_ready) in zip(batch_rows, probs):
            predicted = "READY" if p_ready >= 0.5 else "NOT_READY"
            batch_results.append(
                {
                    "query_id":       row.get("query_id", ""),
                    "step_idx":       row.get("step_idx", -1),
                    "sentence_idx":   row.get("sentence_idx", -1),
                    "true_label":     _true_label(row),
                    "predicted_label": predicted,
                    "p_ready":        round(p_ready, 6),
                    "p_not_ready":    round(p_not_ready, 6),
                }
            )
        return batch_results

    for i in tqdm(range(0, len(rows), batch_size), desc="Evaluating"):
        batch = rows[i : i + batch_size]
        results.extend(_predict_batch(batch))

    # ── Write results ───────────────────────────────────────────────────────
    out_dir = Path("/eval_results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _EVAL_VOL.commit()

    # ── Quick accuracy summary ──────────────────────────────────────────────
    correct = sum(1 for r in results if r["predicted_label"] == r["true_label"])
    total = len(results)
    n_ready_true = sum(1 for r in results if r["true_label"] == "READY")
    n_ready_pred = sum(1 for r in results if r["predicted_label"] == "READY")

    summary = {
        "model":           model_id,
        "test_file":       test_file,
        "total":           total,
        "correct":         correct,
        "accuracy":        round(correct / total, 4) if total else 0,
        "true_ready_frac": round(n_ready_true / total, 4) if total else 0,
        "pred_ready_frac": round(n_ready_pred / total, 4) if total else 0,
        "output_file":     str(out_file),
        "volume":          "pact-eval-results",
        "run_name":        run_name,
    }
    print("\n=== Baseline Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model: str = "microsoft/Phi-3-mini-4k-instruct",
    test_file: str = "test.jsonl",
    batch_size: int = 16,
    max_length: int = 512,
    run_name: str = "",
) -> None:
    name = run_name.strip() or time.strftime("phi3_base_%Y%m%d_%H%M%S")
    print(f"Run name : {name}")
    print(f"Volume   : pact-eval-results  ->  /{name}/predictions.jsonl")

    result = eval_on_modal.remote(
        test_file=test_file,
        run_name=name,
        model_id=model,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("\nDone:", result)
    print(
        f"\nFetch results:\n"
        f"  modal volume get pact-eval-results {name} ./eval_results/"
    )


if __name__ == "__main__":
    print("Use: modal run modal_phi3_baseline_eval/modal_eval_phi3_base.py [--test-file ...]")
