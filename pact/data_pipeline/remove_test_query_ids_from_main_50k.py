"""
Remove test-set query_ids from main_50k JSONL.

This script compares rows by `query_id` only:
  - Reads query_id values from `data/test.jsonl`
  - Reads rows from `data/toolbench_shards_50k/main_50k.jsonl`
  - Drops rows whose query_id appears in test
  - Writes the filtered output to `data/toolbench_shards_50k/main_ds_create.jsonl`

Defaults are resolved from the repository root (directory containing `data/`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_query_ids(test_path: Path) -> set[str]:
    query_ids: set[str] = set()
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            qid = str(row.get("query_id", "")).strip()
            if qid:
                query_ids.add(qid)
    return query_ids


def main() -> None:
    repo = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Remove rows from main_50k.jsonl when their query_id appears in test.jsonl."
        )
    )
    parser.add_argument(
        "--main",
        type=Path,
        default=repo / "data" / "toolbench_shards_50k" / "main_50k.jsonl",
        help="Input main JSONL path",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=repo / "data" / "test.jsonl",
        help="Input test JSONL path (source of query_ids to exclude)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "toolbench_shards_50k" / "main_ds_create.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    args = parser.parse_args()

    if not args.main.is_file():
        raise SystemExit(f"Main file not found: {args.main}")
    if not args.test.is_file():
        raise SystemExit(f"Test file not found: {args.test}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    test_query_ids = _load_query_ids(args.test)

    input_rows = 0
    removed_rows = 0
    written_rows = 0

    with args.main.open(encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            input_rows += 1
            row = json.loads(raw)
            qid = str(row.get("query_id", "")).strip()
            if qid in test_query_ids:
                removed_rows += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"test_query_ids: {len(test_query_ids)}")
    print(f"input_rows: {input_rows}")
    print(f"removed_rows: {removed_rows}")
    print(f"written_rows: {written_rows}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
