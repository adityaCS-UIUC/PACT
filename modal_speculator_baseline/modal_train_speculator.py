#!/usr/bin/env python3
"""
Run ``train_speculator.run_training`` on a Modal GPU; save checkpoints to a Modal Volume.

Setup
-----
1. ``pip install modal`` and ``modal setup``
2. Put your JSONL under ``<repo>/data/`` (e.g. ``data/tasks_20k_gt_only.jsonl``) — the whole
   ``data/`` directory is copied into the container at ``/mnt/data`` when the image is built
   (Modal 1.x ``Image.add_local_dir``; no ``Mount``).
3. Gated models (Phi-3): ``modal secret create huggingface HF_TOKEN=hf_...`` and uncomment
   ``secrets=`` on the function below.

Run (from repo root ``CS-598/``)
--------------------------------
  In Modal 1.x, ``local_entrypoint`` parameters become ``modal run`` options directly — do **not**
  use a bare ``--`` separator (it produces "unexpected extra arguments").

  modal run modal_speculator_baseline/modal_train_speculator.py \\
    --data-file tasks_20k_gt_only.jsonl \\
    --model microsoft/Phi-3-mini-4k-instruct \\
    --epochs 2 \\
    --batch-size 4 \\
    --lr 5e-4 \\
    --gradient-accumulation-steps 4 \\
    --max-examples 5000

Checkpoints are written under volume ``pact-speculator-ckpts`` at ``/<run_name>/``.
After the run, copy them locally, for example::

  modal volume get pact-speculator-ckpts speculator_20260403_120000 ./modal_ckpt

(Exact ``modal volume`` syntax can vary slightly by CLI version — use ``modal volume --help``.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local paths for ``Image.add_local_dir`` (see ``_train_image``)
# ---------------------------------------------------------------------------
# In the Modal worker, this module is often imported as a single file under
# ``/root/modal_train_speculator.py``, so ``Path(__file__).parent.parent`` is
# ``/`` and ``.../data`` becomes ``/data``. Only resolve repo paths when
# ``modal.is_local()`` is true (the machine running ``modal run``).

_BASE_TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "sentencepiece",
        "safetensors",
        "tqdm>=4.66.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

_CHECKPOINT_VOL = modal.Volume.from_name("pact-speculator-ckpts", create_if_missing=True)

app = modal.App("speculator-train")


def _local_pkg_and_data_dirs() -> tuple[Path, Path]:
    """Package dir + repo ``data/`` on the host that runs ``modal run``."""
    pkg = Path(__file__).resolve().parent
    data_dir = pkg.parent / "data"
    return pkg, data_dir


def _train_image() -> modal.Image:
    """Bundle package + repo ``data/`` into the image (host only; Modal 1.x replaces ``Mount``)."""
    if not modal.is_local():
        # Worker: image layers already include ``/pkg`` and ``/mnt/data``; do not use ``__file__`` here.
        return _BASE_TRAIN_IMAGE
    pkg, data_dir = _local_pkg_and_data_dirs()
    if not data_dir.is_dir():
        raise RuntimeError(
            f"Data directory not found: {data_dir}\n"
            "Place your JSONL under repo data/ (e.g. data/tasks_20k_gt_only.jsonl)."
        )
    return _BASE_TRAIN_IMAGE.add_local_dir(pkg, remote_path="/pkg").add_local_dir(
        data_dir, remote_path="/mnt/data"
    )


@app.function(
    gpu="A10G",
    image=_train_image(),
    timeout=6 * 3600,
    volumes={"/ckpts": _CHECKPOINT_VOL},
    # Phi-3 / gated HF: create secret then uncomment:
    # secrets=[modal.Secret.from_name("huggingface")],
)
def train_on_modal(
    data_file: str,
    run_name: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    val_fraction: float,
    seed: int,
    max_rows: int,
    max_examples: int,
    negative_ratio: float,
    max_tools_in_prompt: int,
    tool_desc_chars: int,
    tool_schema_chars: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> dict:
    sys.path.insert(0, "/pkg")

    from train_speculator import TrainingJobConfig, run_training

    max_rows_opt = None if max_rows < 0 else max_rows

    data_path = Path("/mnt/data") / data_file
    if not data_path.is_file():
        raise FileNotFoundError(f"Mounted data missing: {data_path}")

    out_dir = Path("/ckpts") / run_name
    cfg = TrainingJobConfig(
        data=data_path,
        output_dir=out_dir,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        val_fraction=val_fraction,
        seed=seed,
        max_rows=max_rows_opt,
        max_examples=max_examples,
        negative_ratio=negative_ratio,
        max_tools_in_prompt=max_tools_in_prompt,
        tool_desc_chars=tool_desc_chars,
        tool_schema_chars=tool_schema_chars,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        hf_token=None,
    )
    summary = run_training(cfg)
    summary["volume"] = "pact-speculator-ckpts"
    summary["run_name"] = run_name
    summary["remote_checkpoint_path"] = f"/ckpts/{run_name}"
    return summary


@app.local_entrypoint()
def main(
    data_file: str = "tasks_20k_gt_only.jsonl",
    run_name: str = "",
    model: str = "microsoft/Phi-3-mini-4k-instruct",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 3072,
    val_fraction: float = 0.05,
    seed: int = 42,
    max_rows: int = -1,
    max_examples: int = 5000,
    negative_ratio: float = 0.22,
    max_tools_in_prompt: int = 24,
    tool_desc_chars: int = 280,
    tool_schema_chars: int = 400,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> None:
    name = run_name.strip() or time.strftime("speculator_%Y%m%d_%H%M%S")
    print(f"Run name: {name}")
    print(f"Volume:   pact-speculator-ckpts  ->  /ckpts/{name}/")

    result = train_on_modal.remote(
        data_file=data_file,
        run_name=name,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        val_fraction=val_fraction,
        seed=seed,
        max_rows=max_rows,
        max_examples=max_examples,
        negative_ratio=negative_ratio,
        max_tools_in_prompt=max_tools_in_prompt,
        tool_desc_chars=tool_desc_chars,
        tool_schema_chars=tool_schema_chars,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )
    print("Done:", result)
    print(
        "\nFetch checkpoints from the volume (see `modal volume --help`), e.g. copy "
        f"run folder `{name}` to your machine."
    )


if __name__ == "__main__":
    print(
        "Use: modal run modal_speculator_baseline/modal_train_speculator.py "
        "[--data-file ... --model ...]  (no bare `--` before flags; see -h)"
    )
