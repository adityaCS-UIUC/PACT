#!/usr/bin/env python3
"""
Plot **accuracy vs confidence threshold** for a ``train_speculator`` checkpoint.

Also reports **precision / recall / F1** for the READY class and **truth tables**
(confusion matrices) at threshold 0.5 and at the threshold that maximizes accuracy
on the same scored split.

Uses the **same** example construction and train/val split as ``run_training``
(``training_config.json`` next to ``speculator_head.pt``), and scores
``P(READY) = softmax(logits)[1]`` from the frozen LM + trained linear head.

**Intended workflow:** run on Modal (checkpoint stays on volume; no local Phi-3 load).
From the repo root::

     modal run -w accuracy_vs_threshold_val.png \\
       modal_speculator_baseline/modal_eval_speculator_threshold_plot.py \\
       --volume-run speculator_YYYYMMDD_HHMMSS \\
       --data-file tasks_20k_gt_only.jsonl

Logs include the numeric summary (accuracy sweep, precision/recall/F1, confusion tables).

**Local run (opt-in only):** this file refuses to load the model unless you pass
``--allow-local`` (high RAM/VRAM). Use only for debugging; prefer Modal above.

Requires: torch, transformers, numpy, matplotlib (see ``requirements_train.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_speculator import (  # noqa: E402
    Collator,
    FrozenBackboneClassifier,
    SpeculatorTextDataset,
    load_all_examples,
)


def _load_training_config(checkpoint_dir: Path) -> dict[str, Any]:
    p = checkpoint_dir / "training_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p} (expected next to speculator_head.pt)")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_speculator_bundle(path: Path) -> tuple[dict[str, torch.Tensor], str, int]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if loaded.get("format_version") != 1:
        raise RuntimeError(f"Unsupported speculator_head format: {loaded.get('format_version')}")
    state = loaded.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("speculator_head.pt missing state_dict")
    base = loaded.get("base_model")
    if not base:
        raise RuntimeError("speculator_head.pt missing base_model")
    hidden = int(loaded.get("hidden_size", 0))
    if hidden <= 0:
        raise RuntimeError("speculator_head.pt missing hidden_size")
    return state, str(base), hidden


def accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> float:
    pred = (scores >= t).astype(np.int32)
    return float((pred == labels).mean())


def confusion_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> dict[str, int]:
    """
    Binary confusion for label 1 = READY, prediction READY iff score >= t.

    Returns counts: tp, fp, tn, fn (actual class × predicted class).
    """
    pred = (scores >= t).astype(np.int32)
    y = labels.astype(np.int32)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, t: float
) -> dict[str, int | float | None]:
    """
    Confusion counts plus precision, recall, F1 for the READY positive class.
    Undefined precision/recall/F1 use None (e.g. no predicted or no actual READY).
    """
    c = confusion_at_threshold(scores, labels, t)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = None
    acc = float((tp + tn) / n) if n > 0 else None
    out: dict[str, int | float | None] = {**c, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
    return out


def format_confusion_table_lines(metrics: dict[str, Any], title: str) -> list[str]:
    """ASCII truth table + precision/recall/F1 lines for printing."""
    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    tn = int(metrics["tn"])
    fn = int(metrics["fn"])
    lines = [
        title,
        "                 Predicted",
        "              NOT_READY    READY",
        f"Actual NOT_READY  {tn:7d}  {fp:7d}",
        f"       READY       {fn:7d}  {tp:7d}",
    ]
    p, r, f1, a = metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]
    lines.append(
        f"Precision (READY): {p:.4f}" if p is not None else "Precision (READY): n/a (no predicted READY)"
    )
    lines.append(
        f"Recall (READY):    {r:.4f}" if r is not None else "Recall (READY):    n/a (no actual READY)"
    )
    lines.append(
        f"F1 (READY):        {f1:.4f}" if f1 is not None else "F1 (READY):        n/a"
    )
    lines.append(f"Accuracy:          {a:.4f}" if a is not None else "Accuracy:          n/a")
    return lines


def print_threshold_eval_summary(summary: dict[str, Any]) -> None:
    """Stdout report (used by Modal logs and by the local CLI after eval)."""
    print(
        f"Pairs: total={summary['total_pairs']} train={summary['train_pairs']} "
        f"val={summary['val_pairs']} scoring={summary['split']}={summary['scored_pairs']}"
    )
    print(
        f"Class balance (actual): READY={summary['n_actual_ready']}  "
        f"NOT_READY={summary['n_actual_not_ready']}"
    )
    print(f"Accuracy @ threshold 0.5: {summary['accuracy_at_threshold_0.5']:.4f}")
    print(f"Best threshold ≈ {summary['best_threshold']:.3f}  accuracy ≈ {summary['best_accuracy']:.4f}")
    print()
    for line in format_confusion_table_lines(
        summary["metrics_at_threshold_0.5"],
        "Truth table @ threshold 0.5 (predict READY if P(READY) ≥ 0.5)",
    ):
        print(line)
    print()
    for line in format_confusion_table_lines(
        summary["metrics_at_best_threshold"],
        f"Truth table @ best-accuracy threshold ({summary['best_threshold']:.4f})",
    ):
        print(line)


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, 1.0, n)
    accs = np.array([accuracy_at_threshold(scores, labels, float(t)) for t in ts])
    return ts, accs


def _render_plot_png_bytes(ts: np.ndarray, accs: np.ndarray, title: str, dpi: int = 150) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(ts, accs, marker="o", markersize=3)
    plt.xlabel("Confidence threshold (predict READY if score ≥ threshold)")
    plt.ylabel("Accuracy on held-out pairs")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close()
    return buf.getvalue()


def save_plot(ts: np.ndarray, accs: np.ndarray, out: Path, title: str) -> None:
    out.write_bytes(_render_plot_png_bytes(ts, accs, title))


@torch.no_grad()
def collect_ready_probs(
    model: FrozenBackboneClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.classifier.eval()
    scores_list: list[float] = []
    labels_list: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, mask)
        probs = F.softmax(logits.float(), dim=-1)[:, 1]
        scores_list.extend(probs.cpu().tolist())
        labels_list.extend(labels.cpu().tolist())
    return np.array(scores_list, dtype=np.float64), np.array(labels_list, dtype=np.int32)


def threshold_plot_png_bytes(
    checkpoint_dir: Path,
    data_path: Path,
    *,
    split: str = "val",
    batch_size: int = 8,
    n_thresholds: int = 51,
    hf_token: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Build the same split as training, score ``P(READY)``, return PNG bytes + numeric summary.
    Used by local CLI and ``modal_eval_speculator_threshold_plot.py``.
    """
    ckpt_dir = checkpoint_dir.resolve()
    head_path = ckpt_dir / "speculator_head.pt"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing {head_path}")

    tc = _load_training_config(ckpt_dir)
    data_path = data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Data JSONL not found: {data_path}")

    max_rows = tc.get("max_rows")
    max_ex_req = tc.get("max_examples_requested", tc.get("max_examples", 5000))
    max_examples = None if max_ex_req == 0 else int(max_ex_req)

    random.seed(int(tc["seed"]))
    torch.manual_seed(int(tc["seed"]))

    pairs = load_all_examples(
        data_path,
        max_rows=max_rows,
        max_examples=max_examples,
        negative_truncation_ratio=float(tc["negative_ratio"]),
        max_tools=int(tc["max_tools_in_prompt"]),
        desc_max_chars=int(tc["tool_desc_chars"]),
        schema_max_chars=int(tc["tool_schema_chars"]),
        seed=int(tc["seed"]),
    )
    if len(pairs) < 32:
        raise ValueError(f"Too few examples: {len(pairs)}")

    rng = random.Random(int(tc["seed"]))
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * float(tc["val_fraction"])))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    eval_pairs = val_pairs if split == "val" else train_pairs

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    state_dict, base_model, hidden = _load_speculator_bundle(head_path)
    max_length = int(tc["max_length"])

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=token,
        torch_dtype=dtype,
        device_map=None,
    )
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    model = FrozenBackboneClassifier(backbone, int(backbone.config.hidden_size)).to(device)
    if model.classifier.weight.shape[1] != hidden:
        raise ValueError(
            f"Checkpoint hidden_size {hidden} vs backbone {model.classifier.weight.shape[1]} — wrong base model?"
        )
    model.classifier.load_state_dict(state_dict)

    ds = SpeculatorTextDataset(eval_pairs, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=Collator(), num_workers=0)

    scores, labels = collect_ready_probs(model, loader, device)
    ts, accs = sweep_thresholds(scores, labels, n=int(n_thresholds))
    best_i = int(np.argmax(accs))
    acc_half = accuracy_at_threshold(scores, labels, 0.5)
    t_best = float(ts[best_i])
    metrics_05 = classification_metrics_at_threshold(scores, labels, 0.5)
    metrics_best = classification_metrics_at_threshold(scores, labels, t_best)
    n_ready = int((labels == 1).sum())
    n_not_ready = int((labels == 0).sum())

    title = f"{split} split — accuracy vs P(READY) threshold\n{ckpt_dir.name}"
    png = _render_plot_png_bytes(ts, accs, title)

    summary: dict[str, Any] = {
        "checkpoint_dir": str(ckpt_dir),
        "data_path": str(data_path),
        "split": split,
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "scored_pairs": len(eval_pairs),
        "n_actual_ready": n_ready,
        "n_actual_not_ready": n_not_ready,
        "accuracy_at_threshold_0.5": acc_half,
        "best_threshold": t_best,
        "best_accuracy": float(accs[best_i]),
        "n_thresholds": int(n_thresholds),
        "metrics_at_threshold_0.5": {**metrics_05, "threshold": 0.5},
        "metrics_at_best_threshold": {**metrics_best, "threshold": t_best},
    }
    return png, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing speculator_head.pt, tokenizer, training_config.json",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="JSONL path (defaults to training_config['data'] if that file exists)",
    )
    p.add_argument("--plot-path", type=Path, default=Path("accuracy_vs_threshold_val.png"))
    p.add_argument("--split", choices=("val", "train"), default="val", help="Which split to score (same shuffle as training)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-thresholds", type=int, default=51, help="Number of thresholds in [0, 1]")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow loading the LM + eval on this machine. Without this flag, the CLI exits and tells you to use Modal.",
    )
    args = p.parse_args()

    if not args.allow_local:
        raise SystemExit(
            "This eval is meant to run on Modal (avoids loading Phi-3 locally).\n\n"
            "From the repo root:\n"
            "  modal run -w accuracy_vs_threshold_val.png \\\n"
            "    modal_speculator_baseline/modal_eval_speculator_threshold_plot.py \\\n"
            "    --volume-run YOUR_RUN_DIR_NAME \\\n"
            "    --data-file tasks_20k_gt_only.jsonl\n\n"
            "Check Modal logs for the full summary dict (precision/recall/F1, confusion counts).\n\n"
            "To force a local run anyway, pass --allow-local."
        )

    ckpt_dir = args.checkpoint_dir.resolve()
    tc = _load_training_config(ckpt_dir)
    data_path = args.data
    if data_path is None:
        cand = Path(tc["data"])
        data_path = cand if cand.is_file() else None
    if data_path is None or not data_path.is_file():
        raise SystemExit(
            "Pass --data /path/to/tasks_20k_gt_only.jsonl "
            "(training_config data path is missing or not on this machine)."
        )

    png, summary = threshold_plot_png_bytes(
        ckpt_dir,
        data_path,
        split=args.split,
        batch_size=args.batch_size,
        n_thresholds=args.n_thresholds,
        hf_token=args.hf_token,
    )
    print_threshold_eval_summary(summary)
    args.plot_path.write_bytes(png)
    print(f"Wrote {args.plot_path.resolve()}")


if __name__ == "__main__":
    main()
