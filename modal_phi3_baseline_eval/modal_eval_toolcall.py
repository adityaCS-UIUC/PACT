#!/usr/bin/env python3
"""
Compare base vs. fine-tuned Phi-3-mini on tool-call prediction.

Evaluates both models on READY rows of data/test.jsonl and reports:
  - Tool-name accuracy  : predicted tool name == target tool name
  - Exact match         : tool name + JSON args match (after normalisation)
  - Per-row predictions saved to Modal Volume `pact-eval-results`

Run (from repo root CS-598/):
    modal run modal_phi3_baseline_eval/modal_eval_toolcall.py \\
        --ckpt-run-name phi3_toolcall_pred_20260422_120000

Flags:
    --ckpt-run-name   Name of the fine-tuned checkpoint in volume pact-sft-ckpts (required)
    --model           Base HF model id  (default: microsoft/Phi-3-mini-4k-instruct)
    --test-file       JSONL filename under data/  (default: test.jsonl)
    --batch-size      Generation batch size  (default: 8)
    --max-length      Max input tokens  (default: 512)
    --max-new-tokens  Max tokens to generate  (default: 128)
    --run-name        Output folder in pact-eval-results  (default: toolcall_eval_<ts>)

Fetch results:
    modal volume get pact-eval-results <run-name> ./eval_results/
"""

from __future__ import annotations

import json
import re
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

app = modal.App("phi3-toolcall-eval")

# ---------------------------------------------------------------------------
# Shared helpers  (input formatting + output parsing)
# ---------------------------------------------------------------------------
_OUTPUT_PREFIX = "\n\n### Tool Call\n"


def _format_input_base(row: dict) -> str:
    """Prompt for the base (non-finetuned) model.

    Explains the expected output format explicitly so the model has a fair
    chance without any fine-tuning.  No task-specific examples are given —
    just format instructions.
    """
    thought = (row.get("cumulative_thought") or "").strip()
    return (
        "You are a tool-calling agent. When the agent is ready, output exactly "
        "one tool call using this format and nothing else:\n"
        "#TOOL[tool_name]{\"arg\": \"value\"}\n\n"
        "Example:\n"
        "#TOOL[search_web]{\"query\": \"capital of France\"}\n\n"
        "### Thought\n"
        f"{thought}\n\n"
        "The agent is READY. Output the tool call."
        + _OUTPUT_PREFIX
    )


def _format_input_ft(row: dict) -> str:
    """Prompt for the fine-tuned model — must match the training-time format exactly."""
    thought = (row.get("cumulative_thought") or "").strip()
    return (
        "### Thought\n"
        f"{thought}\n\n"
        "The agent is READY. Output the tool call."
        + _OUTPUT_PREFIX
    )


def _target_tool_call_str(row: dict) -> str:
    tc = row.get("target_tool_call") or {}
    tool_name = (tc.get("tool_name") or "").strip()
    arguments = tc.get("arguments") or {}
    args_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"),
                           sort_keys=True)
    return f"#TOOL[{tool_name}]{args_json}"


def _parse_prediction(raw: str) -> tuple[str, dict]:
    """Extract (tool_name, args_dict) from generated text; returns ("", {}) on failure."""
    raw = raw.strip()
    m = re.search(r"#TOOL\[([^\]]+)\](\{.*\})", raw, re.DOTALL)
    if not m:
        return "", {}
    tool_name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        args = {}
    return tool_name, args


def _normalise(tool_name: str, args: dict) -> str:
    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"#TOOL[{tool_name}]{args_json}"


def _is_ready_row(row: dict) -> bool:
    return (
        int(row.get("ready_label", row.get("label", 0))) == 1
        and bool((row.get("target_tool_call") or {}).get("tool_name", "").strip())
    )


# ---------------------------------------------------------------------------
# Modal function — runs both evals in one GPU container
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    image=_build_image(),
    timeout=4 * 3600,
    volumes={
        "/ckpts": _CKPT_VOL,
        "/eval_results": _EVAL_VOL,
    },
    # secrets=[modal.Secret.from_name("huggingface")],
)
def eval_toolcall_on_modal(
    *,
    test_file: str,
    ckpt_run_name: str,
    run_name: str,
    model_id: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> dict:
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda")

    # ── Load test data (READY rows only) ───────────────────────────────────
    test_path = Path("/mnt/data") / test_file
    rows: list[dict] = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if _is_ready_row(row):
                    rows.append(row)
    print(f"Loaded {len(rows)} READY test rows from {test_file}")

    # ── Load tokenizer (shared by both models) ──────────────────────────────
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Generation helper ───────────────────────────────────────────────────
    @torch.no_grad()
    def generate_batch(model, batch_rows: list[dict], fmt_fn) -> list[str]:
        prompts = [fmt_fn(r) for r in batch_rows]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy — deterministic, comparable
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    def run_eval(model, label: str, fmt_fn) -> list[dict]:
        model.eval()
        results = []
        for i in tqdm(range(0, len(rows), batch_size), desc=f"Eval {label}"):
            batch = rows[i : i + batch_size]
            preds = generate_batch(model, batch, fmt_fn)
            for row, raw_pred in zip(batch, preds):
                tc = row.get("target_tool_call") or {}
                target_tool_name = (tc.get("tool_name") or "").strip()
                target_args = tc.get("arguments") or {}
                target_str = _normalise(target_tool_name, target_args)

                pred_tool_name, pred_args = _parse_prediction(raw_pred)
                pred_str = _normalise(pred_tool_name, pred_args) if pred_tool_name else raw_pred.strip()

                results.append({
                    "query_id":          row.get("query_id", ""),
                    "step_idx":          row.get("step_idx", -1),
                    "sentence_idx":      row.get("sentence_idx", -1),
                    "target":            target_str,
                    "target_tool_name":  target_tool_name,
                    f"{label}_raw":      raw_pred.strip(),
                    f"{label}_pred":     pred_str,
                    f"{label}_tool_name_match": pred_tool_name == target_tool_name,
                    f"{label}_exact_match":     pred_str == target_str,
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

    # Free base model VRAM before loading finetuned
    del base_model
    torch.cuda.empty_cache()

    # ── 2. Finetuned model ──────────────────────────────────────────────────
    ckpt_path = Path("/ckpts") / ckpt_run_name
    if not ckpt_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available runs: {[p.name for p in Path('/ckpts').iterdir()]}"
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

    # ── Merge base + finetuned results by row index ─────────────────────────
    merged: list[dict] = []
    for b, f in zip(base_results, ft_results):
        row = {**b}
        for k, v in f.items():
            if k.startswith("ft_") or k in ("ft_raw", "ft_pred"):
                row[k] = v
        merged.append(row)

    # ── Compute summary ─────────────────────────────────────────────────────
    n = len(merged)
    def _pct(key): return round(sum(r[key] for r in merged) / n * 100, 2) if n else 0

    summary = {
        "model":                    model_id,
        "ckpt_run_name":            ckpt_run_name,
        "test_file":                test_file,
        "ready_rows_evaluated":     n,
        "base_tool_name_acc_%":     _pct("base_tool_name_match"),
        "base_exact_match_%":       _pct("base_exact_match"),
        "ft_tool_name_acc_%":       _pct("ft_tool_name_match"),
        "ft_exact_match_%":         _pct("ft_exact_match"),
        "tool_name_improvement_%":  round(_pct("ft_tool_name_match") - _pct("base_tool_name_match"), 2),
        "exact_match_improvement_%": round(_pct("ft_exact_match") - _pct("base_exact_match"), 2),
    }

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = Path("/eval_results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _EVAL_VOL.commit()

    print("\n" + "=" * 50)
    print("=== Tool Call Prediction — Comparison Summary ===")
    print("=" * 50)
    print(f"  Rows evaluated       : {n}")
    print(f"  Base  — tool name acc: {summary['base_tool_name_acc_%']}%")
    print(f"  Base  — exact match  : {summary['base_exact_match_%']}%")
    print(f"  FT    — tool name acc: {summary['ft_tool_name_acc_%']}%")
    print(f"  FT    — exact match  : {summary['ft_exact_match_%']}%")
    print(f"  Δ tool name          : +{summary['tool_name_improvement_%']}%")
    print(f"  Δ exact match        : +{summary['exact_match_improvement_%']}%")
    print("=" * 50)

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    ckpt_run_name: str,                          # required: name of the finetuned checkpoint
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    test_file: str = "test.jsonl",
    batch_size: int = 8,
    max_length: int = 512,
    max_new_tokens: int = 128,
    run_name: str = "",
) -> None:
    if not ckpt_run_name:
        raise SystemExit("ERROR: --ckpt-run-name is required. "
                         "Pass the run name of your finetuned checkpoint, e.g.\n"
                         "  modal run modal_phi3_baseline_eval/modal_eval_toolcall.py "
                         "--ckpt-run-name phi3_toolcall_pred_20260422_120000")

    name = run_name.strip() or time.strftime("toolcall_eval_%Y%m%d_%H%M%S")
    print(f"Run name      : {name}")
    print(f"Checkpoint    : pact-sft-ckpts / {ckpt_run_name}")
    print(f"Results volume: pact-eval-results / {name}")

    result = eval_toolcall_on_modal.remote(
        test_file=test_file,
        ckpt_run_name=ckpt_run_name,
        run_name=name,
        model_id=model,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    print("\nFinal result:", result)
    print(f"\nFetch results:\n  modal volume get pact-eval-results {name} ./eval_results/")


if __name__ == "__main__":
    print("Use: modal run modal_phi3_baseline_eval/modal_eval_toolcall.py "
          "--ckpt-run-name <name>")
