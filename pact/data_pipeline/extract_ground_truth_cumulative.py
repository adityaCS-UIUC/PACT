"""
Extract ground-truth style paths into cumulative sentence-level logs.

Input A — ToolBench (``ground_truth_path`` non-empty):
  - query_id, prompt, api_specs, available_tools, …
  - ground_truth_path: [{"qwen_reasoning" or "thought", "action", "action_input"}, ...]

Input B — Qwen traces (no ``ground_truth_path``; example row shape):
  - Top-level ``qwen_reasoning`` is sentence-split and cumulated.
  - ``target_tool_call`` ({tool_name, arguments}) selects the step-0 tool target;
    if missing, the first ``gt_calls`` entry is used.
  - ``gt_calls`` (optional) is still written to ``*_actions.jsonl`` for every step.

Outputs (all JSONL):
  1) api_specs file      : one row per task with api_specs / available_tools
  2) actions file        : one row per ground-truth step (action + action_input)
  3) cumulative file     : one row per cumulative thought chunk. Each row is a
                           shallow copy of the input task (api_specs,
                           available_tools, ground_truth_path, etc.) plus
                           step_idx, sentence_idx, num_sentences_total,
                           cumulative_thought, target_tool_call, label.

For each step, reasoning text is taken from ``qwen_reasoning`` when non-empty,
otherwise ``thought``. Cumulative rows are created as:
  s1
  s1 + s2
  s1 + s2 + s3
  ...
where label=0 for intermediate chunks and label=1 for the final chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _split_sentences(text: str) -> list[str]:
    """
    Sentence splitting via Python library when available.
    - Preferred: nltk.sent_tokenize
    - Fallback: regex split on punctuation.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    try:
        from nltk.tokenize import sent_tokenize  # type: ignore
        parts = sent_tokenize(cleaned)
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)

    return [p.strip() for p in parts if p and p.strip()]


def _cumulative_chunks(sentences: list[str]) -> list[str]:
    chunks: list[str] = []
    for i in range(1, len(sentences) + 1):
        chunks.append(" ".join(sentences[:i]))
    return chunks


def _step_reasoning_text(step: dict[str, Any]) -> str:
    """Prefer ``qwen_reasoning``, fall back to ``thought`` for ToolBench rows."""
    qr = step.get("qwen_reasoning")
    if qr is not None and str(qr).strip():
        return str(qr).strip()
    return str(step.get("thought", "")).strip()


def _target_tool_call_from_row(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve (tool_name, arguments) for Qwen-style rows without ground_truth_path."""
    tc = row.get("target_tool_call") or {}
    if isinstance(tc, dict):
        name = str(tc.get("tool_name", "")).strip()
        args = tc.get("arguments", {}) or {}
        if name:
            return name, args if isinstance(args, dict) else {}
    gcalls = row.get("gt_calls") or []
    if gcalls and isinstance(gcalls, list) and isinstance(gcalls[0], dict):
        first = gcalls[0]
        name = str(first.get("action", "")).strip()
        args = first.get("action_input", {}) or {}
        if name:
            return name, args if isinstance(args, dict) else {}
    return "", {}


def _append_cumulative_rows(
    row: dict[str, Any],
    *,
    step_idx: int,
    reasoning_text: str,
    tool_name: str,
    action_input: dict[str, Any],
    cumulative_out: list[dict[str, Any]],
) -> None:
    if not reasoning_text or not tool_name:
        return
    sentences = _split_sentences(reasoning_text)
    if not sentences:
        return
    chunks = _cumulative_chunks(sentences)
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        cum_row = dict(row)
        cum_row.update(
            {
                "step_idx": step_idx,
                "sentence_idx": i,
                "num_sentences_total": total,
                "cumulative_thought": chunk,
                "target_tool_call": {
                    "tool_name": tool_name,
                    "arguments": action_input,
                },
                "label": 1 if i == total else 0,
            }
        )
        cumulative_out.append(cum_row)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def extract_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs_out: list[dict[str, Any]] = []
    actions_out: list[dict[str, Any]] = []
    cumulative_out: list[dict[str, Any]] = []

    for row in rows:
        qid = str(row.get("query_id", ""))
        prompt = str(row.get("prompt", ""))
        category = row.get("category")
        available_tools = row.get("available_tools", [])
        api_specs = row.get("api_specs", [])
        raw_path = row.get("ground_truth_path")
        gt_path = list(raw_path) if isinstance(raw_path, list) else []

        if gt_path:
            num_steps = len(gt_path)
        else:
            gcalls = row.get("gt_calls") or []
            if isinstance(gcalls, list) and gcalls:
                num_steps = len(gcalls)
            elif str(row.get("qwen_reasoning", "")).strip():
                num_steps = 1
            else:
                num_steps = 0

        specs_out.append({
            "query_id": qid,
            "category": category,
            "available_tools": available_tools,
            "api_specs": api_specs,
            "num_ground_truth_steps": num_steps,
        })

        if gt_path:
            for step_idx, step in enumerate(gt_path):
                if not isinstance(step, dict):
                    continue
                reasoning_text = _step_reasoning_text(step)
                action = str(step.get("action", "")).strip()
                action_input = step.get("action_input", {}) or {}
                if not isinstance(action_input, dict):
                    action_input = {}

                actions_out.append({
                    "query_id": qid,
                    "category": category,
                    "step_idx": step_idx,
                    "qwen_reasoning": reasoning_text,
                    "action": action,
                    "action_input": action_input,
                })

                _append_cumulative_rows(
                    row,
                    step_idx=step_idx,
                    reasoning_text=reasoning_text,
                    tool_name=action,
                    action_input=action_input,
                    cumulative_out=cumulative_out,
                )
        else:
            # Qwen-style: sentence-split top-level qwen_reasoning; target step 0 only.
            reasoning_text = str(row.get("qwen_reasoning", "")).strip()
            tool_name, action_input = _target_tool_call_from_row(row)
            gcalls = row.get("gt_calls") or []

            if isinstance(gcalls, list) and gcalls:
                for step_idx, call in enumerate(gcalls):
                    if not isinstance(call, dict):
                        continue
                    action = str(call.get("action", "")).strip()
                    a_in = call.get("action_input", {}) or {}
                    if not isinstance(a_in, dict):
                        a_in = {}
                    step_qr = reasoning_text if step_idx == 0 else ""
                    actions_out.append({
                        "query_id": qid,
                        "category": category,
                        "step_idx": step_idx,
                        "qwen_reasoning": step_qr,
                        "action": action,
                        "action_input": a_in,
                    })
                _append_cumulative_rows(
                    row,
                    step_idx=0,
                    reasoning_text=reasoning_text,
                    tool_name=tool_name,
                    action_input=action_input,
                    cumulative_out=cumulative_out,
                )
            elif tool_name:
                actions_out.append({
                    "query_id": qid,
                    "category": category,
                    "step_idx": 0,
                    "qwen_reasoning": reasoning_text,
                    "action": tool_name,
                    "action_input": action_input,
                })
                _append_cumulative_rows(
                    row,
                    step_idx=0,
                    reasoning_text=reasoning_text,
                    tool_name=tool_name,
                    action_input=action_input,
                    cumulative_out=cumulative_out,
                )
            elif reasoning_text:
                actions_out.append({
                    "query_id": qid,
                    "category": category,
                    "step_idx": 0,
                    "qwen_reasoning": reasoning_text,
                    "action": "",
                    "action_input": {},
                })

    return specs_out, actions_out, cumulative_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build cumulative sentence logs from ground_truth_path or Qwen "
            "(top-level qwen_reasoning + target_tool_call / gt_calls)"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL (ToolBench with ground_truth_path, or Qwen traces)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/ground_truth_extract"),
        help="Output directory for extracted JSONL files",
    )
    parser.add_argument(
        "--prefix",
        default="shard",
        help="Prefix for output file names",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    specs_out, actions_out, cumulative_out = extract_rows(rows)

    specs_path = args.output_dir / f"{args.prefix}_api_specs.jsonl"
    actions_path = args.output_dir / f"{args.prefix}_actions.jsonl"
    cumulative_path = args.output_dir / f"{args.prefix}_cumulative.jsonl"

    _write_jsonl(specs_path, specs_out)
    _write_jsonl(actions_path, actions_out)
    _write_jsonl(cumulative_path, cumulative_out)

    print(f"Input rows: {len(rows)}")
    print(f"API specs rows: {len(specs_out)} -> {specs_path}")
    print(f"Action rows: {len(actions_out)} -> {actions_path}")
    print(f"Cumulative rows: {len(cumulative_out)} -> {cumulative_path}")


if __name__ == "__main__":
    main()

