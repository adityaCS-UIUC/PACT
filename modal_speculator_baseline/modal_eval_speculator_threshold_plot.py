#!/usr/bin/env python3
"""
Run ``threshold_plot_png_bytes`` on a Modal GPU: load checkpoint from volume
``pact-speculator-ckpts``, JSONL from baked-in ``data/``, return a PNG (accuracy vs
``P(READY)`` threshold). Avoids local RAM for Phi-3 + long contexts.

Prerequisites
-------------
- Same as training: ``modal setup``, repo ``data/`` present when you ``modal run``
  (image includes ``/mnt/data`` via ``add_local_dir``).
- Checkpoint directory must already exist on the volume (e.g. from
  ``modal_train_speculator``), e.g. ``/ckpts/speculator_20260403_205533/``.

Save the plot locally (bytes return value)
------------------------------------------
  From repo root ``CS-598/``::

    modal run -w accuracy_vs_threshold_val.png \\
      modal_speculator_baseline/modal_eval_speculator_threshold_plot.py \\
      --volume-run speculator_20260403_205533 \\
      --data-file tasks_20k_gt_only.jsonl

  Memory: eval uses ``max_length`` from ``training_config.json`` (often 3072) and **padded**
  batches. Default ``--batch-size`` is **2** (training often used 4); if you still OOM, pass
  ``--batch-size 1``. GPU is **A100** (was A10G) for more VRAM headroom.

  Phi-3 / gated HF: ``modal secret create huggingface HF_TOKEN=hf_...`` then
  uncomment ``secrets=`` on the function below.

Logs on the Modal dashboard will include the numeric summary dict (best threshold,
accuracies, pair counts, precision/recall/F1, confusion counts at 0.5 and best threshold).
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

_BASE_EVAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "accelerate>=1.0.0",
        "sentencepiece",
        "safetensors",
        "matplotlib>=3.8.0",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

_CHECKPOINT_VOL = modal.Volume.from_name("pact-speculator-ckpts", create_if_missing=True)

app = modal.App("speculator-eval-threshold-plot")


def _local_pkg_and_data_dirs() -> tuple[Path, Path]:
    pkg = Path(__file__).resolve().parent
    data_dir = pkg.parent / "data"
    return pkg, data_dir


def _eval_image() -> modal.Image:
    if not modal.is_local():
        return _BASE_EVAL_IMAGE
    pkg, data_dir = _local_pkg_and_data_dirs()
    if not data_dir.is_dir():
        raise RuntimeError(
            f"Data directory not found: {data_dir}\n"
            "Place your JSONL under repo data/ (e.g. data/tasks_20k_gt_only.jsonl)."
        )
    return _BASE_EVAL_IMAGE.add_local_dir(pkg, remote_path="/pkg").add_local_dir(
        data_dir, remote_path="/mnt/data"
    )


@app.function(
    gpu="A100",
    image=_eval_image(),
    timeout=90 * 60,
    volumes={"/ckpts": _CHECKPOINT_VOL},
    # Gated models: uncomment after `modal secret create huggingface HF_TOKEN=...`
    # secrets=[modal.Secret.from_name("huggingface")],
)
def eval_threshold_plot_on_modal(
    volume_run_name: str,
    data_file: str,
    split: str,
    batch_size: int,
    n_thresholds: int,
) -> bytes:
    sys.path.insert(0, "/pkg")
    from eval_speculator_threshold_plot import print_threshold_eval_summary, threshold_plot_png_bytes

    name = volume_run_name.strip().strip("/")
    ckpt = Path("/ckpts") / name
    data = Path("/mnt/data") / data_file
    png, summary = threshold_plot_png_bytes(
        ckpt,
        data,
        split=split,
        batch_size=batch_size,
        n_thresholds=n_thresholds,
        hf_token=None,
    )
    print_threshold_eval_summary(summary)
    print("Summary (JSON-serializable):", summary)
    return png


@app.local_entrypoint()
def main(
    volume_run: str = "speculator_20260403_205533",
    data_file: str = "tasks_20k_gt_only.jsonl",
    split: str = "val",
    batch_size: int = 2,
    n_thresholds: int = 51,
) -> bytes:
    """
    Return PNG bytes. Use ``modal run -w out.png ...`` to write the file locally
    without holding the model in RAM on your laptop.
    """
    return eval_threshold_plot_on_modal.remote(
        volume_run_name=volume_run,
        data_file=data_file,
        split=split,
        batch_size=batch_size,
        n_thresholds=n_thresholds,
    )
