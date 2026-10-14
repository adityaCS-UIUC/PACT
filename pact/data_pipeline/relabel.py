"""
Hindsight relabeling pipeline (Phase 2).

Reads raw TrajectoryLogs from Agent A, filters for success, chunks the
<think> blocks into cumulative sentences, runs the oracle test to find
the tau-point, and exports labeled JSONL for speculator training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pact.config import TrajectoryLog, SpeculatorSample
from pact.data_pipeline.oracle import oracle_check, oracle_check_adaptive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_think_text(text: str) -> str:
    """Pull content between <think> and </think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (rough but good enough for chunking)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _cumulative_chunks(sentences: list[str]) -> list[str]:
    """Build cumulative sentence chunks: [s1], [s1 s2], [s1 s2 s3], ..."""
    chunks: list[str] = []
    for i in range(1, len(sentences) + 1):
        chunks.append(" ".join(sentences[:i]))
    return chunks


# ---------------------------------------------------------------------------
# Core relabeling
# ---------------------------------------------------------------------------

def relabel_trajectory(
    log: TrajectoryLog,
    use_llm_oracle: bool = False,
) -> list[SpeculatorSample]:
    """
    Given a *successful* trajectory, produce speculator training samples.

    For each step that contains a think block followed by a tool call:
      1. Split the think text into sentences.
      2. Build cumulative chunks.
      3. Run the oracle: first chunk where oracle says YES -> label 1;
         all prior chunks -> label 0.
      4. Discard remaining chunks after the tau-point.

    When use_llm_oracle is True, uses the adaptive oracle (keyword for
    known PACT tools, LLM for unknown ToolBench APIs).
    """
    samples: list[SpeculatorSample] = []

    for step in log.steps:
        if not step.tool_calls:
            continue
        if not step.think_block or not step.think_block.full_text.strip():
            continue

        tc = step.tool_calls[0]
        think_text = step.think_block.full_text

        think_clean = _extract_think_text(think_text)
        sentences = _split_sentences(think_clean)
        if not sentences:
            continue

        chunks = _cumulative_chunks(sentences)

        for chunk in chunks:
            if use_llm_oracle:
                is_ready = oracle_check_adaptive(
                    chunk=chunk,
                    tool_name=tc.tool_name,
                    tool_arguments=tc.arguments,
                    use_llm_for_unknown_tools=True,
                )
            else:
                is_ready = oracle_check(
                    chunk=chunk,
                    tool_name=tc.tool_name,
                    tool_arguments=tc.arguments,
                )

            label = 1 if is_ready else 0

            samples.append(SpeculatorSample(
                query_id=log.query_id,
                prompt=log.prompt,
                cumulative_thought=chunk,
                target_tool_call={
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                },
                label=label,
            ))

            if is_ready:
                break

    return samples


# ---------------------------------------------------------------------------
# Batch relabeling
# ---------------------------------------------------------------------------

def relabel_all(
    log_dir: str | Path,
    output_path: str | Path,
    only_successes: bool = True,
    use_llm_oracle: bool = False,
) -> int:
    """
    Process all trajectory logs in log_dir and write labeled JSONL.

    Args:
        use_llm_oracle: If True, use the adaptive oracle (LLM fallback
            for unknown ToolBench APIs).

    Returns the number of samples written.
    """
    log_dir = Path(log_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as f:
        for log_file in sorted(log_dir.glob("*.json")):
            log = TrajectoryLog.load(log_file)
            if only_successes and not log.task_success:
                continue
            samples = relabel_trajectory(log, use_llm_oracle=use_llm_oracle)
            for s in samples:
                line = json.dumps({
                    "query_id": s.query_id,
                    "prompt": s.prompt,
                    "cumulative_thought": s.cumulative_thought,
                    "target_tool_call": s.target_tool_call,
                    "label": s.label,
                }, default=str)
                f.write(line + "\n")
                total += 1

    print(f"Wrote {total} samples to {output_path}")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Relabel Agent A logs for speculator training")
    parser.add_argument("--log-dir", default="logs/agent_a")
    parser.add_argument("--output", default="data/training/speculator_train.jsonl")
    args = parser.parse_args()
    relabel_all(args.log_dir, args.output)
