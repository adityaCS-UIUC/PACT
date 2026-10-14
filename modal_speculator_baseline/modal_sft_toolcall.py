#!/usr/bin/env python3
"""
Tool-call prediction SFT on a Modal GPU (A100).

Trains Phi-3-mini (LoRA) to predict which tool to call and with what arguments,
given the original task prompt + the agent's cumulative thought.

Only READY rows (ready_label=1) with a valid target_tool_call are used (~18k rows
out of 48k total in train.jsonl).

  Input  = task prompt  +  cumulative_thought
  Output = "#TOOL[tool_name]{JSON}"

Including the task prompt grounds argument prediction in the original user request
(usernames, IDs, keywords, etc. are often stated there but omitted from the thought).

Run (from repo root CS-598/):
    modal run modal_speculator_baseline/modal_sft_toolcall.py

Optional flags:
    --data-file   train.jsonl (default)
    --model       microsoft/Phi-3-mini-4k-instruct (default)
    --epochs      2 (default)
    --batch-size  4 (default)
    --lr          1e-4 (default)

Artifacts are written to Modal volume `pact-sft-ckpts` under /<run_name>/.
Fetch locally:
    modal volume get pact-sft-ckpts <run_name> ./modal_ckpt/
"""

from __future__ import annotations

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

_CKPT_VOL = modal.Volume.from_name("pact-sft-ckpts", create_if_missing=True)

app = modal.App("qwen2.5-toolcall-pred-sft")


def _build_image() -> modal.Image:
    if not modal.is_local():
        return _BASE_IMAGE
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"data/ not found at {data_dir}")
    return _BASE_IMAGE.add_local_dir(repo_root, remote_path="/repo").add_local_dir(
        data_dir, remote_path="/mnt/data"
    )


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100",
    image=_build_image(),
    timeout=12 * 3600,
    volumes={"/ckpts": _CKPT_VOL},
    # secrets=[modal.Secret.from_name("huggingface")],  # uncomment if Phi-3 is gated
)
def sft_toolcall_on_modal(
    *,
    data_file: str,
    run_name: str,
    model: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    max_length: int,
    val_fraction: float,
    seed: int,
    max_rows: int,
    no_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict:
    sys.path.insert(0, "/repo")
    from pact.sft.toolcall_pred_sft import ToolCallPredConfig, run_toolcall_pred_sft

    data_path = Path("/mnt/data") / data_file
    if not data_path.is_file():
        raise FileNotFoundError(f"Mounted data missing: {data_path}")

    out_dir = Path("/ckpts") / run_name
    cfg = ToolCallPredConfig(
        data=data_path,
        output_dir=out_dir,
        model=model,
        max_length=max_length,
        seed=seed,
        val_fraction=val_fraction,
        max_rows=None if max_rows < 0 else max_rows,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        use_lora=not no_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    summary = run_toolcall_pred_sft(cfg)
    summary["volume"] = "pact-sft-ckpts"
    summary["run_name"] = run_name
    summary["remote_checkpoint_path"] = f"/ckpts/{run_name}"
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    data_file: str = "train.jsonl",
    run_name: str = "",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 1,                     # 2 epochs: ~17k READY examples benefit from 2 passes
    batch_size: int = 4,                 # A100 + LoRA + 512 tokens fits batch=4 comfortably
    gradient_accumulation_steps: int = 4,  # effective batch = 16
    lr: float = 1e-4,
    warmup_ratio: float = 0.1,           # more warmup for smaller dataset
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    max_length: int = 768,               # prompt + thought + tool call; 768 covers >99%
    val_fraction: float = 0.05,
    seed: int = 42,
    max_rows: int = -1,
    no_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    name = run_name.strip() or time.strftime("qwen2.5_toolcall_prompt_%Y%m%d_%H%M%S")
    print(f"Run name : {name}")
    print(f"Volume   : pact-sft-ckpts  ->  /ckpts/{name}/")

    result = sft_toolcall_on_modal.remote(
        data_file=data_file,
        run_name=name,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        max_length=max_length,
        val_fraction=val_fraction,
        seed=seed,
        max_rows=max_rows,
        no_lora=no_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print("Done:", result)
    print(f"\nFetch checkpoint:\n  modal volume get pact-sft-ckpts {name} ./modal_ckpt/")


if __name__ == "__main__":
    print("Use: modal run modal_speculator_baseline/modal_sft_toolcall.py [--epochs 2 ...]")
