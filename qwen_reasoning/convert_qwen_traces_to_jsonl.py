"""
Convert Qwen trace JSON files into JSONL.

Supports both trace roots used in this repo:
  - qwen_traces_v1
  - qwen_traces_v2

Each input .json file is expected to contain one JSON object (single task trace).
The script writes one compact JSON object per line in the output .jsonl file.

Default output:
  data/qwen_traces_combined.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_trace_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def _load_json_obj(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {path} (invalid JSON: {exc})")
        return None

    if not isinstance(obj, dict):
        print(f"[skip] {path} (top-level JSON is not an object)")
        return None
    return obj


def main() -> None:
    repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Convert Qwen trace .json files (v1/v2) into one JSONL file.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[repo / "qwen_traces_v1", repo / "qwen_traces_v2"],
        help="Input root directories to scan recursively for .json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "qwen_traces_combined.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--dedupe-by-query-id",
        action="store_true",
        help="Keep only first record per query_id across all inputs",
    )
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Only include traces where predicted tool name matches ground truth (tool_name_correct=True)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output if it exists",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output exists: {args.output}. Use --overwrite to replace it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for root in args.inputs:
        files.extend(_iter_trace_json_files(root))
    files = sorted(set(files))

    scanned_files = 0
    written_rows = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    skipped_incorrect = 0
    seen_qids: set[str] = set()

    with args.output.open("w", encoding="utf-8") as out:
        for path in files:
            scanned_files += 1
            row = _load_json_obj(path)
            if row is None:
                skipped_invalid += 1
                continue

            if args.only_correct and not row.get("tool_name_correct", False):
                skipped_incorrect += 1
                continue

            if args.dedupe_by_query_id:
                qid = str(row.get("query_id", "")).strip()
                if qid and qid in seen_qids:
                    skipped_duplicate += 1
                    continue
                if qid:
                    seen_qids.add(qid)

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"Input roots: {[str(p) for p in args.inputs]}")
    print(f"Scanned files: {scanned_files}")
    print(f"Written rows: {written_rows}")
    print(f"Skipped invalid/non-object JSON: {skipped_invalid}")
    print(f"Skipped incorrect tool calls: {skipped_incorrect}")
    print(f"Skipped duplicates by query_id: {skipped_duplicate}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
