#!/usr/bin/env python3
"""
Compare base vs. fine-tuned Phi-3-mini on READY / NOT_READY prediction.

Evaluates both models on ALL rows of data/test.jsonl and reports:
  - Accuracy
  - Precision, Recall, F1  (for the READY class)
  - P(READY) confidence score per row  (useful for threshold sweeping later)

Confidence method:
  One forward pass per batch; read logits at the last input position.
  P(READY) = softmax( [logit_READY, logit_NOT_READY] )[0]
  This is fast (no generation loop) and gives a continuous score.

Run (from repo root CS-598/):
    modal run modal_phi3_baseline_eval/modal_eval_rnr.py \\
        --ckpt-run-name phi3_rnr_sft_20260422_120000

Flags:
    --ckpt-run-name   Name of the fine-tuned R/NR checkpoint in pact-sft-ckpts (required)
    --model           Base HF model id  (default: microsoft/Phi-3-mini-4k-instruct)
    --test-file       JSONL filename under data/  (default: test.jsonl)
    --batch-size      Inference batch size  (default: 16)
    --max-length      Max input tokens  (default: 512)
    --run-name        Output folder in pact-eval-results  (default: rnr_eval_<ts>)

Fetch results:
    modal volume get pact-eval-results <run-name> ./eval_results/
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
        "peft>=0.12.0",
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
        raise RuntimeError(f"data/ not found at {data_dir}")
    return _BASE_IMAGE.add_local_dir(data_dir, remote_path="/mnt/data")


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
_CKPT_VOL = modal.Volume.from_name("pact-sft-ckpts", create_if_missing=False)
_EVAL_VOL = modal.Volume.from_name("pact-eval-results", create_if_missing=True)

app = modal.App("phi3-rnr-eval")

# ---------------------------------------------------------------------------
# Input formatters
# ---------------------------------------------------------------------------
_OUTPUT_PREFIX = "\n\n### Decision\n"


def _format_input_base(row: dict) -> str:
    """Zero-shot prompt for the base model — explains the task and answer format."""
    thought = (row.get("cumulative_thought") or "").strip()
    return (
        "You are an agent evaluator. Read the agent's current thought and decide "
        "whether it has gathered enough information to call a tool right now.\n\n"
        "Answer with exactly one word — either READY or NOT_READY — and nothing else.\n\n"
        "### Thought\n"
        f"{thought}\n\n"
        "Is the agent ready to call a tool?"
        + _OUTPUT_PREFIX
    )


def _format_input_ft(row: dict) -> str:
    """Prompt for the fine-tuned model — must match training-time format exactly.

    Includes the task prompt because modal_sft_r_nr.py now uses
    toolcall_sft_with_prompt (prompt + thought as input).
    """
    task_prompt = (row.get("prompt") or "").strip()
    thought = (row.get("cumulative_thought") or "").strip()
    return (
        "### Task\n"
        f"{task_prompt}\n\n"
        "### Thought\n"
        f"{thought}\n\n"
        "Is the agent ready to call a tool? Answer READY or NOT_READY."
        + _OUTPUT_PREFIX
    )


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------
def _true_label(row: dict) -> str:
    judge_raw = (row.get("judge_raw") or "").strip().upper()
    if judge_raw in ("READY", "NOT_READY"):
        return judge_raw
    return "READY" if int(row.get("ready_label", row.get("label", 0))) == 1 else "NOT_READY"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _compute_metrics(results: list[dict], pred_key: str, true_key: str = "true_label") -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        pred = r[pred_key]
        true = r[true_key]
        if true == "READY" and pred == "READY":
            tp += 1
        elif true == "NOT_READY" and pred == "READY":
            fp += 1
        elif true == "READY" and pred == "NOT_READY":
            fn += 1
        else:
            tn += 1
    n = len(results)
    acc = round((tp + tn) / n * 100, 2) if n else 0
    prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) else 0
    rec = round(tp / (tp + fn) * 100, 2) if (tp + fn) else 0
    f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) else 0
    return {
        "accuracy_%": acc,
        "precision_%": prec,
        "recall_%": rec,
        "f1_%": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _threshold_sweep(
    results: list[dict],
    p_ready_key: str,
    true_key: str = "true_label",
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep P(READY) thresholds and return per-threshold metrics + best-by-F1 row."""
    if thresholds is None:
        thresholds = [round(t / 100, 2) for t in range(5, 100, 5)]  # 0.05 … 0.95

    rows = []
    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for r in results:
            pred = "READY" if r[p_ready_key] >= thresh else "NOT_READY"
            true = r[true_key]
            if true == "READY"     and pred == "READY":     tp += 1
            elif true == "NOT_READY" and pred == "READY":   fp += 1
            elif true == "READY"   and pred == "NOT_READY": fn += 1
            else:                                           tn += 1
        n = len(results)
        acc  = round((tp + tn) / n * 100, 2) if n else 0
        prec = round(tp / (tp + fp) * 100, 2) if (tp + fp) else 0
        rec  = round(tp / (tp + fn) * 100, 2) if (tp + fn) else 0
        f1   = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) else 0
        rows.append({
            "threshold": thresh,
            "accuracy_%": acc,
            "precision_%": prec,
            "recall_%": rec,
            "f1_%": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",           # A10G sufficient for inference-only (no generation loop)
    image=_build_image(),
    timeout=3 * 3600,
    volumes={
        "/ckpts": _CKPT_VOL,
        "/eval_results": _EVAL_VOL,
    },
    # secrets=[modal.Secret.from_name("huggingface")],
)
def eval_rnr_on_modal(
    *,
    test_file: str,
    ckpt_run_name: str,
    run_name: str,
    model_id: str,
    batch_size: int,
    max_length: int,
) -> dict:
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda")

    # ── Load test data ──────────────────────────────────────────────────────
    test_path = Path("/mnt/data") / test_file
    rows: list[dict] = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} test rows from {test_file}")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Find first-token IDs for READY and NOT_READY (done once, reused for both models)
    ready_tok = tokenizer.encode("READY", add_special_tokens=False)[0]
    not_ready_tok = tokenizer.encode("NOT_READY", add_special_tokens=False)[0]
    print(f"Token IDs — READY: {ready_tok}  NOT_READY: {not_ready_tok}")

    # ── Inference helper (single forward pass, no generation) ───────────────
    @torch.no_grad()
    def predict_batch(model, batch_rows: list[dict], fmt_fn) -> list[tuple[str, float, float]]:
        """Returns list of (predicted_label, p_ready, p_not_ready)."""
        prompts = [fmt_fn(r) for r in batch_rows]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc)
        logits = outputs.logits  # [B, seq_len, vocab]

        # Logit at the last real token position predicts the next (output) token
        last_pos = enc["attention_mask"].sum(dim=1) - 1  # [B]
        last_logits = logits[torch.arange(len(batch_rows)), last_pos]  # [B, vocab]

        target_logits = last_logits[:, [ready_tok, not_ready_tok]]  # [B, 2]
        probs = target_logits.softmax(dim=-1).cpu().tolist()

        results = []
        for p_ready, p_not_ready in probs:
            label = "READY" if p_ready >= 0.5 else "NOT_READY"
            results.append((label, round(p_ready, 6), round(p_not_ready, 6)))
        return results

    def run_eval(model, label: str, fmt_fn) -> list[dict]:
        model.eval()
        results = []
        for i in tqdm(range(0, len(rows), batch_size), desc=f"Eval {label}"):
            batch = rows[i : i + batch_size]
            preds = predict_batch(model, batch, fmt_fn)
            for row, (pred_label, p_ready, p_not_ready) in zip(batch, preds):
                results.append({
                    "query_id":              row.get("query_id", ""),
                    "step_idx":              row.get("step_idx", -1),
                    "sentence_idx":          row.get("sentence_idx", -1),
                    "true_label":            _true_label(row),
                    f"{label}_pred":         pred_label,
                    f"{label}_p_ready":      p_ready,
                    f"{label}_p_not_ready":  p_not_ready,
                    f"{label}_correct":      pred_label == _true_label(row),
                })
        return results

    # ── 1. Base model ───────────────────────────────────────────────────────
    print("\n=== Evaluating BASE model ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_results = run_eval(base_model, "base", _format_input_base)

    del base_model
    torch.cuda.empty_cache()

    # ── 2. Finetuned model ──────────────────────────────────────────────────
    ckpt_path = Path("/ckpts") / ckpt_run_name
    if not ckpt_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available: {[p.name for p in Path('/ckpts').iterdir()]}"
        )
    print(f"\n=== Evaluating FINETUNED model from {ckpt_path} ===")
    ft_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ft_model = PeftModel.from_pretrained(ft_base, str(ckpt_path))
    ft_results = run_eval(ft_model, "ft", _format_input_ft)

    del ft_model, ft_base
    torch.cuda.empty_cache()

    # ── Merge results by row index ──────────────────────────────────────────
    merged: list[dict] = []
    for b, f in zip(base_results, ft_results):
        row = {**b}
        for k, v in f.items():
            if k.startswith("ft_"):
                row[k] = v
        merged.append(row)

    # ── Fixed-threshold metrics (threshold = 0.5) ───────────────────────────
    base_metrics = _compute_metrics(merged, pred_key="base_pred")
    ft_metrics   = _compute_metrics(merged, pred_key="ft_pred")

    # ── Threshold sweep ─────────────────────────────────────────────────────
    base_sweep = _threshold_sweep(merged, p_ready_key="base_p_ready")
    ft_sweep   = _threshold_sweep(merged, p_ready_key="ft_p_ready")

    base_best = max(base_sweep, key=lambda r: r["f1_%"])
    ft_best   = max(ft_sweep,   key=lambda r: r["f1_%"])

    summary = {
        "model":          model_id,
        "ckpt_run_name":  ckpt_run_name,
        "test_file":      test_file,
        "total_rows":     len(merged),
        "at_threshold_0.5": {
            "base":      base_metrics,
            "finetuned": ft_metrics,
            "improvement": {
                "accuracy_%":  round(ft_metrics["accuracy_%"]  - base_metrics["accuracy_%"],  2),
                "precision_%": round(ft_metrics["precision_%"] - base_metrics["precision_%"], 2),
                "recall_%":    round(ft_metrics["recall_%"]    - base_metrics["recall_%"],    2),
                "f1_%":        round(ft_metrics["f1_%"]        - base_metrics["f1_%"],        2),
            },
        },
        "best_threshold": {
            "base":      base_best,
            "finetuned": ft_best,
        },
    }

    # ── Save results ────────────────────────────────────────────────────────
    out_dir = Path("/eval_results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save full sweep tables as CSV for easy plotting
    import csv
    for label, sweep in [("base", base_sweep), ("ft", ft_sweep)]:
        with (out_dir / f"threshold_sweep_{label}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep[0].keys())
            writer.writeheader()
            writer.writerows(sweep)

    _EVAL_VOL.commit()

    # ── Print fixed-threshold table ──────────────────────────────────────────
    bm = summary["at_threshold_0.5"]["base"]
    fm = summary["at_threshold_0.5"]["finetuned"]
    im = summary["at_threshold_0.5"]["improvement"]

    print("\n" + "=" * 60)
    print("=== R/NR @ threshold 0.5 ===")
    print("=" * 60)
    print(f"  {'Metric':<18}  {'Base':>8}  {'Finetuned':>10}  {'Δ':>7}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*7}")
    for metric in ("accuracy_%", "precision_%", "recall_%", "f1_%"):
        lbl = metric.replace("_%", "").capitalize()
        print(f"  {lbl:<18}  {bm[metric]:>7}%  {fm[metric]:>9}%  {im[metric]:>+6}%")
    print(f"  Confusion (base):  TP={bm['tp']} FP={bm['fp']} FN={bm['fn']} TN={bm['tn']}")
    print(f"  Confusion (ft)  :  TP={fm['tp']} FP={fm['fp']} FN={fm['fn']} TN={fm['tn']}")

    # ── Print threshold sweep highlights ────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== Threshold Sweep — Best F1 per model ===")
    print("=" * 60)
    print(f"  {'Thresh':<8}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for lbl, best in [("Base", base_best), ("FT", ft_best)]:
        print(f"  {lbl} best @ {best['threshold']:.2f}:")
        print(
            f"    Acc={best['accuracy_%']}%  "
            f"Prec={best['precision_%']}%  "
            f"Rec={best['recall_%']}%  "
            f"F1={best['f1_%']}%"
        )

    # ── Print full sweep table for FT model ─────────────────────────────────
    print("\n  FT full sweep:")
    print(f"  {'Thresh':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for row in ft_sweep:
        marker = " ◀ best F1" if row["threshold"] == ft_best["threshold"] else ""
        print(
            f"  {row['threshold']:>7.2f}  "
            f"{row['accuracy_%']:>6}%  "
            f"{row['precision_%']:>6}%  "
            f"{row['recall_%']:>6}%  "
            f"{row['f1_%']:>6}%"
            f"{marker}"
        )
    print("=" * 60)
    print(f"\nCSV sweep tables saved to: pact-eval-results / {run_name}/")
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    ckpt_run_name: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    test_file: str = "test.jsonl",
    batch_size: int = 16,
    max_length: int = 768,
    run_name: str = "",
) -> None:
    if not ckpt_run_name:
        raise SystemExit(
            "ERROR: --ckpt-run-name is required.\n"
            "Pass the run name of your fine-tuned R/NR checkpoint, e.g.\n"
            "  modal run modal_phi3_baseline_eval/modal_eval_rnr.py "
            "--ckpt-run-name phi3_rnr_sft_20260422_120000"
        )

    name = run_name.strip() or time.strftime("rnr_eval_%Y%m%d_%H%M%S")
    print(f"Run name      : {name}")
    print(f"Checkpoint    : pact-sft-ckpts / {ckpt_run_name}")
    print(f"Results volume: pact-eval-results / {name}")

    result = eval_rnr_on_modal.remote(
        test_file=test_file,
        ckpt_run_name=ckpt_run_name,
        run_name=name,
        model_id=model,
        batch_size=batch_size,
        max_length=max_length,
    )
    print("\nFinal result:", result)
    print(f"\nFetch results:\n  modal volume get pact-eval-results {name} ./eval_results/")


if __name__ == "__main__":
    print("Use: modal run modal_phi3_baseline_eval/modal_eval_rnr.py --ckpt-run-name <name>")
