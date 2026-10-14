#!/usr/bin/env python3
"""
R/NR SFT variant that includes the original task prompt in the model input.

Identical to toolcall_sft.py in every way EXCEPT format_sft_example now prepends
the task prompt so the model can use the full goal context when deciding READY/NOT_READY.

  Input  = task prompt  +  cumulative_thought
  Output = "READY" or "NOT_READY"

Why this helps:
  Early in a trajectory the cumulative_thought is only 1-2 sentences and does not
  fully reflect the task.  Including the original prompt lets the model reason about
  whether the current thought is sufficient relative to WHAT was asked.

Max-length guidance:
  Prompts are typically 50-150 tokens; thoughts 50-300 tokens.
  768 tokens covers >99% of examples (vs 512 for thought-only).
"""

from __future__ import annotations

from typing import Any

import pact.sft.toolcall_sft as _base

# Re-export so the Modal runner can `from pact.sft.toolcall_sft_with_prompt import SftConfig, run_sft`
SftConfig = _base.SftConfig


def _format_with_prompt(row: dict[str, Any]) -> tuple[str, str]:
    task_prompt = (row.get("prompt") or "").strip()
    thought = (row.get("cumulative_thought") or "").strip()

    judge_raw = (row.get("judge_raw") or "").strip().upper()
    if judge_raw in ("READY", "NOT_READY"):
        out = judge_raw
    else:
        ready = int(row.get("ready_label", row.get("label", 0)))
        out = "READY" if ready == 1 else "NOT_READY"

    inp = (
        "### Task\n"
        f"{task_prompt}\n\n"
        "### Thought\n"
        f"{thought}\n\n"
        "Is the agent ready to call a tool? Answer READY or NOT_READY."
    )
    return inp, out


def run_sft(cfg: SftConfig) -> dict[str, Any]:
    """Run SFT with prompt-aware formatting.

    Temporarily patches the format function in the base module so that
    JsonlToolcallDataset picks it up, then restores the original.
    """
    original = _base.format_sft_example
    _base.format_sft_example = _format_with_prompt
    try:
        return _base.run_sft(cfg)
    finally:
        _base.format_sft_example = original
