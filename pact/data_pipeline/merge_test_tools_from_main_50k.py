"""
Copy ``available_tools`` and ``api_specs`` from ``main_50k.jsonl`` into test rows.

For each line in ``--test``, if ``query_id`` matches a row in ``--main`` and the
test row is missing either field (key absent or JSON ``null``), fill only the
missing fields from the main row. Writes one JSON object per line to
``--output``.

Default paths (relative to the ``CS-598`` repo root, i.e. the parent of
``pact/``) match the on-disk layout:

  ``data/test.jsonl``                          — input
  ``data/toolbench_shards_50k/main_50k.jsonl`` — tool metadata source
  ``data/test_updated.jsonl``                  — output

Run from the repo root (folder that contains ``data/`` and ``pact/``)::

    python3 pact/data_pipeline/merge_test_tools_from_main_50k.py

Defaults are resolved from the script path, so you can also invoke the script by
absolute path from any working directory. Input is ``test.jsonl`` (JSONL), not
``test.json``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def _needs_merge(row: dict[str, Any]) -> bool:
    return "available_tools" not in row or row["available_tools"] is None or "api_specs" not in row or row["api_specs"] is None


def _enrich_from_main(row: dict[str, Any], main_tools: dict[str, Any] | None) -> bool:
    if main_tools is None:
        return False
    changed = False
    if ("available_tools" not in row or row["available_tools"] is None) and "available_tools" in main_tools:
        row["available_tools"] = main_tools["available_tools"]
        changed = True
    if ("api_specs" not in row or row["api_specs"] is None) and "api_specs" in main_tools:
        row["api_specs"] = main_tools["api_specs"]
        changed = True
    return changed


def _index_main(conn: sqlite3.Connection, main_path: Path) -> int:
    conn.execute(
        "CREATE TABLE main_row (query_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    n = 0
    with main_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            qid = obj.get("query_id")
            if not qid:
                continue
            payload = json.dumps(
                {
                    "available_tools": obj.get("available_tools"),
                    "api_specs": obj.get("api_specs"),
                },
                ensure_ascii=False,
            )
            conn.execute(
                "INSERT OR REPLACE INTO main_row (query_id, payload) VALUES (?, ?)",
                (str(qid), payload),
            )
            n += 1
    conn.commit()
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge available_tools & api_specs from main_50k into test JSONL by query_id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example (defaults match data/test.jsonl and data/toolbench_shards_50k/main_50k.jsonl):

  python3 pact/data_pipeline/merge_test_tools_from_main_50k.py

Explicit paths (repo root = directory containing data/):

  python3 pact/data_pipeline/merge_test_tools_from_main_50k.py \\
    --test data/test.jsonl \\
    --main data/toolbench_shards_50k/main_50k.jsonl \\
    --output data/test_updated.jsonl
""",
    )
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--test",
        type=Path,
        default=repo / "data" / "test.jsonl",
        help="Input test JSONL (rows keyed by query_id)",
    )
    parser.add_argument(
        "--main",
        type=Path,
        default=repo / "data" / "toolbench_shards_50k" / "main_50k.jsonl",
        help="ToolBench main JSONL (source of available_tools & api_specs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "test_updated.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.test.is_file():
        raise SystemExit(f"Test file not found: {args.test}")
    if not args.main.is_file():
        raise SystemExit(f"Main file not found: {args.main}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "main_tools.sqlite"
        conn = sqlite3.connect(str(db_path))
        try:
            n_main = _index_main(conn, args.main)
            print(f"Indexed {n_main} rows from {args.main}", flush=True)

            n_out = 0
            n_need = 0
            n_filled = 0
            n_missing = 0
            missing_ids: set[str] = set()

            cur = conn.cursor()
            with args.test.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
                for line in fin:
                    raw = line.strip()
                    if not raw:
                        continue
                    row: dict[str, Any] = json.loads(raw)
                    qid = str(row.get("query_id", ""))

                    if _needs_merge(row) and qid:
                        n_need += 1
                        cur.execute("SELECT payload FROM main_row WHERE query_id = ?", (qid,))
                        hit = cur.fetchone()
                        if hit is None:
                            n_missing += 1
                            if len(missing_ids) < 50:
                                missing_ids.add(qid)
                        else:
                            main_part = json.loads(hit[0])
                            if _enrich_from_main(row, main_part):
                                n_filled += 1

                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_out += 1

            print(f"Wrote {n_out} lines to {args.output}", flush=True)
            print(f"Rows that needed tools/specs (missing key or null): {n_need}", flush=True)
            print(f"Rows enriched from main: {n_filled}", flush=True)
            print(f"Rows still missing lookup (no query_id in main): {n_missing}", flush=True)
            if missing_ids:
                sample = ", ".join(sorted(missing_ids)[:20])
                print(f"Sample missing query_id (up to 20): {sample}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
