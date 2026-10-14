"""
LLM judge for READY vs NOT_READY on cumulative-thought rows.

Input row format (from extract_ground_truth_cumulative.py or relabel-like data):
  {
    "query_id": "...",
    "prompt": "...",
    "cumulative_thought": "...",
    "target_tool_call": {"tool_name": "...", "arguments": {...}},
    ...
  }

Output:
  Copies each input row and adds:
    - "ready_label": 1 (READY) or 0 (NOT_READY)
    - "judge_raw": raw model text (e.g., READY / NOT_READY / YES / NO)
    - "judge_model": model used

Judging intention:
  Determine whether this much reasoning is enough for a model to infer the
  exact target tool call. The reasoning need not literally mention the tool
  name or arguments verbatim.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

_JUDGE_SYSTEM = """\
You are a labeling judge.

Given:
1) a user task,
2) a cumulative reasoning chunk (partial thought),
3) a reference tool call (tool name + arguments),

decide whether THIS MUCH reasoning is enough for a strong model to infer and
execute the exact reference tool call.

Important:
- Do NOT require the reasoning to explicitly mention the tool name.
- Do NOT require argument values to appear verbatim.
- Paraphrases and implied entities are acceptable if they make the correct call
  inferable.
- If key details are missing or ambiguous, return NOT_READY.

Respond with exactly one token: READY or NOT_READY.
"""

_JUDGE_USER = """\
## User task
{prompt}

## Cumulative reasoning
{chunk}

## Reference tool call (for judgment only)
Tool: {tool_name}
Arguments: {arguments}

Is the reasoning sufficient to infer this exact call now?"""


def _parse_ready(raw: str) -> int:
    text = (raw or "").strip().upper()
    if text.startswith("READY") or text.startswith("YES"):
        return 1
    if text.startswith("NOT_READY") or text.startswith("NO"):
        return 0
    return 0


def _row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("query_id", "")),
        int(row.get("step_idx", -1)),
        int(row.get("sentence_idx", -1)),
    )


def _retry_sleep_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to sleep before retrying after rate limit / transient errors."""
    # Retry-After header (seconds)
    resp = getattr(exc, "response", None)
    if resp is not None:
        h = resp.headers.get("retry-after")
        if h:
            try:
                return float(h)
            except ValueError:
                pass

    # Message like "Please try again in 8.64s"
    msg = str(exc)
    m = re.search(r"try again in\s+([0-9.]+)\s*s", msg, re.I)
    if m:
        return float(m.group(1))

    # Exponential backoff with cap (RPD exhaustion may need hours; user can stop)
    base = min(300.0, 2.0 ** min(attempt, 8))
    return base + random.uniform(0, 1.0)


def _judge_row_with_retry(
    client: Any,
    row: dict[str, Any],
    model: str,
    *,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> tuple[int, str]:
    tc = row.get("target_tool_call") or {}
    tool_name = str(tc.get("tool_name", "")).strip()
    arguments = tc.get("arguments", {}) or {}
    prompt = str(row.get("prompt", "")).strip()
    chunk = str(row.get("cumulative_thought", "")).strip()

    user = _JUDGE_USER.format(
        prompt=prompt[:2500],
        chunk=chunk[:3000],
        tool_name=tool_name,
        arguments=json.dumps(arguments, default=str)[:1200],
    )

    from openai import RateLimitError, APIConnectionError, APITimeoutError

    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4,
                "temperature": 0.0,
            }
            if reasoning_effort is not None:
                create_kwargs["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**create_kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            return _parse_ready(raw), raw
        except Exception as e:  # noqa: BLE001 — narrow below where possible
            last_err = e
            retryable = isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError))
            if not retryable and "rate" in str(e).lower() and "limit" in str(e).lower():
                retryable = True

            if not retryable or attempt >= max_retries - 1:
                raise

            wait = _retry_sleep_seconds(e, attempt)
            if isinstance(e, RateLimitError) and "requests per day" in str(e).lower():
                print(
                    "\nNOTE: Daily request cap (RPD) hit for this model/key. "
                    "Sleeping before retry may not help until quota resets; "
                    "use --resume tomorrow, another API key, or a different model.\n",
                    flush=True,
                )
            time.sleep(wait)

    raise RuntimeError(f"judge failed after retries: {last_err!r}")


def _load_resume_state(path: Path) -> tuple[set[tuple[str, int, int]], set[str]]:
    """Return (written_row_keys, query_ids_that_already_emitted READY)."""
    written: set[tuple[str, int, int]] = set()
    ready_tasks: set[str] = set()
    if not path.exists():
        return written, ready_tasks
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            written.add(_row_key(row))
            if int(row.get("ready_label", 0)) == 1:
                ready_tasks.add(str(row.get("query_id", "")))
    return written, ready_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge for cumulative READY/NOT_READY labels")
    parser.add_argument("--input", type=Path, required=True, help="Input cumulative JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output judged JSONL")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        metavar="LEVEL",
        help=(
            "Optional OpenAI reasoning_effort for reasoning-capable models "
            '(e.g. "none" for fast classification / no extended reasoning). '
            "Omit for models that do not accept this parameter."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max input rows to scan")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing output (skip already-written rows and tasks already READY)",
    )
    parser.add_argument(
        "--max-retries-per-call",
        type=int,
        default=25,
        help="Retries for transient rate limits / network errors per row",
    )
    parser.add_argument(
        "--stop-after-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default: stop after first READY per query_id (later rows omitted). "
            "Use --no-stop-after-ready to LLM-judge every cumulative row (e.g. two-task debug)."
        ),
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite and not args.resume:
        raise SystemExit(
            f"Output exists: {args.output}. Use --overwrite, or --resume to continue."
        )

    if args.overwrite:
        written_keys: set[tuple[str, int, int]] = set()
        ready_tasks: set[str] = set()
    elif args.resume and args.output.exists():
        written_keys, ready_tasks = _load_resume_state(args.output)
    else:
        written_keys, ready_tasks = set(), set()

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore
            return iterable

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: openai. Activate project venv and install requirements."
        ) from e

    client = OpenAI()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite or not args.output.exists():
        mode = "w"
    else:
        mode = "a"

    n_in = 0
    n_written = 0
    n_skipped_resume = 0
    n_skipped_after_ready = 0

    ready_seen_by_task: dict[str, bool] = {q: True for q in ready_tasks}

    with args.output.open(mode, encoding="utf-8") as out:
        with args.input.open(encoding="utf-8") as inp:
            for line in tqdm(inp, desc="Input lines", unit="line"):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_in += 1
                if args.limit is not None and n_in > args.limit:
                    break

                qid = str(row.get("query_id", ""))
                key = _row_key(row)

                if key in written_keys:
                    n_skipped_resume += 1
                    continue

                if args.stop_after_ready and ready_seen_by_task.get(qid, False):
                    n_skipped_after_ready += 1
                    continue

                label, raw = _judge_row_with_retry(
                    client,
                    row,
                    args.model,
                    max_retries=args.max_retries_per_call,
                    reasoning_effort=args.reasoning_effort,
                )
                new_row = dict(row)
                new_row["ready_label"] = label
                new_row["judge_raw"] = raw
                new_row["judge_model"] = args.model
                out.write(json.dumps(new_row, ensure_ascii=False, default=str) + "\n")
                out.flush()
                n_written += 1
                written_keys.add(key)

                if args.stop_after_ready and label == 1:
                    ready_seen_by_task[qid] = True

    print(f"Input lines scanned: {n_in}")
    print(f"Rows written this run: {n_written}")
    print(f"Skipped (already in output, resume): {n_skipped_resume}")
    print(f"Skipped (after READY for query_id): {n_skipped_after_ready}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
