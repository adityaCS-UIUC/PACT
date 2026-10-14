#!/usr/bin/env python3
"""
Binary readiness SFT on a Modal GPU (A100) — classification objective + prompt.

Trains Phi-3-mini (LoRA) using first-token binary classification loss:
  Input  = task prompt + cumulative_thought
  Output = cross-entropy over [NOT_READY_tok, READY_tok] at last input position

Key improvements over generation-loss approach:
  - Classification loss: 100% of gradient on the decision token (no output appended)
  - Auto class weights: upweights minority READY class (~1.62x) to fix imbalance
  - Task prompt included: gives model full goal context, especially for early steps
  - Best checkpoint saved by validation F1, not loss
  - Logs accuracy / precision / recall / F1 at every eval epoch

At inference: one forward pass, read P(READY) from first-token logits, sweep threshold.

Run (from repo root `CS-598/`)
--------------------------------
  modal run modal_speculator_baseline/modal_sft_r_nr.py

Artifacts are written to Modal volume `pact-sft-ckpts` under `/<run_name>/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

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

app = modal.App("qwen2.5-rnr-sft")


def _local_pkg_and_data_dirs() -> tuple[Path, Path]:
    pkg = Path(__file__).resolve().parent.parent  # repo root
    data_dir = pkg / "data"
    return pkg, data_dir


def _image() -> modal.Image:
    if not modal.is_local():
        return _BASE_IMAGE
    repo_root, data_dir = _local_pkg_and_data_dirs()
    if not data_dir.is_dir():
        raise RuntimeError(
            f"Data directory not found: {data_dir}\n"
            "Place your JSONL under repo data/."
        )
    return _BASE_IMAGE.add_local_dir(repo_root, remote_path="/repo").add_local_dir(
        data_dir, remote_path="/mnt/data"
    )


@app.function(
    gpu="A100",
    image=_image(),
    timeout=12 * 3600,
    volumes={"/ckpts": _CKPT_VOL},
    # If Phi-3 is gated, create secret then uncomment:
    # secrets=[modal.Secret.from_name("huggingface")],
)
def sft_on_modal(
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
    no_class_weight: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict:
    sys.path.insert(0, "/repo")
    from pact.sft.toolcall_sft_with_prompt import SftConfig, run_sft

    data_path = Path("/mnt/data") / data_file
    if not data_path.is_file():
        raise FileNotFoundError(f"Mounted data missing: {data_path}")

    out_dir = Path("/ckpts") / run_name
    cfg = SftConfig(
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
        use_class_weight=not no_class_weight,
        use_lora=not no_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hf_token=None,
    )
    summary = run_sft(cfg)
    summary["volume"] = "pact-sft-ckpts"
    summary["run_name"] = run_name
    summary["remote_checkpoint_path"] = f"/ckpts/{run_name}"
    print(summary)
    return summary


@app.local_entrypoint()
def main(
    data_file: str = "train.jsonl",
    run_name: str = "",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 1,
    batch_size: int = 2,                   # prompt+thought ~500 tokens; batch=2 safe at 768
    gradient_accumulation_steps: int = 8,  # effective batch = 16
    lr: float = 1e-4,                      # conservative: classification loss converges fast
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    max_length: int = 768,                 # prompt(~150t) + thought(~300t) + overhead
    val_fraction: float = 0.05,
    seed: int = 42,
    max_rows: int = -1,
    no_lora: bool = False,
    no_class_weight: bool = False,         # pass --no-class-weight to disable
    lora_r: int = 16,                      # bumped for better classifier capacity
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    name = run_name.strip() or time.strftime("qwen2.5_rnr_cls_%Y%m%d_%H%M%S")
    print(f"Run name: {name}")
    print(f"Volume:   pact-sft-ckpts  ->  /ckpts/{name}/")
    result = sft_on_modal.remote(
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
        no_class_weight=no_class_weight,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    print("Done:", result)


if __name__ == "__main__":
    print("Use: modal run modal_speculator_baseline/modal_sft_toolcall.py [--data-file ...]")

