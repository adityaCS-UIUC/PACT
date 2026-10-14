"""
Remove rows whose ground_truth_path is empty.

This script reads a JSONL file, drops rows where `ground_truth_path` is an empty
list (`[]`), and writes the remaining rows to a new JSONL file.

Default paths:
  input : data/toolbench_shards_50k/main_50k.jsonl
  output: data/toolbench_shards_50k/main_no_empty_ground_truth_path.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _is_empty_ground_truth_path(row: dict[str, Any]) -> bool:
    value = row.get("ground_truth_path", None)
    return isinstance(value, list) and len(value) == 0


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Remove rows where ground_truth_path is exactly []",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "data" / "toolbench_shards_50k" / "main_50k.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "data"
        / "toolbench_shards_50k"
        / "main_no_empty_ground_truth_path.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    removed = 0
    written = 0

    with args.input.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            scanned += 1
            row = json.loads(raw)
            if _is_empty_ground_truth_path(row):
                removed += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            written += 1

    print(f"Scanned rows: {scanned}")
    print(f"Removed rows (ground_truth_path == []): {removed}")
    print(f"Written rows: {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
