"""
Filter judged cumulative rows: READY before the full thought (last prefix).

Keeps rows where:
  - sentence_idx < num_sentences_total  (not the final cumulative chunk)
  - judge_raw indicates READY (case-insensitive, accepts leading READY)

Typical use: find cases where the model would commit early relative to the
full ground-truth thought span.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _is_ready(raw: Any) -> bool:
    t = str(raw or "").strip().upper()
    return t == "READY" or t.startswith("READY")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract rows with READY before final sentence_idx (early readiness)",
    )
    parser.add_argument("--input", type=Path, required=True, help="Judged JSONL (e.g. shard_02_cumulative_judged.jsonl)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output} (pass --overwrite)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    scanned = 0

    with args.input.open(encoding="utf-8") as inp, args.output.open("w", encoding="utf-8") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            row = json.loads(line)

            try:
                s_idx = int(row.get("sentence_idx", -1))
                n_total = int(row.get("num_sentences_total", -1))
            except (TypeError, ValueError):
                continue

            if s_idx >= n_total:
                continue
            if not _is_ready(row.get("judge_raw")):
                continue

            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            kept += 1

    print(f"Scanned rows: {scanned}")
    print(f"Matched (sentence_idx < num_sentences_total AND READY): {kept}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
