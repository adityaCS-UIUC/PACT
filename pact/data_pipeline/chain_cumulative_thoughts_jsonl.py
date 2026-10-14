"""
Cross-step chaining for cumulative_thought in judged JSONL.

Within each query_id, rows are ordered by (step_idx, sentence_idx). Step 0
keeps its existing cumulative_thought. For step_idx > 0, each row's
cumulative_thought becomes:

    <final cumulative_thought from step_idx - 1> + separator +
    <this row's original cumulative_thought (intra-step local chain)>

so reasoning reads as one continuing trace across tool steps.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any


def _join_prefix_local(prefix: str, local: str, separator: str) -> str:
    p, l = prefix.strip(), local.strip()
    if not p:
        return l
    if not l:
        return p
    return f"{p}{separator}{l}"


def chain_rows_for_query(rows: list[dict[str, Any]], separator: str) -> None:
    """Mutate rows in place: same keys, updated cumulative_thought."""
    rows.sort(key=lambda r: (int(r["step_idx"]), int(r["sentence_idx"])))
    prefix = ""
    for _step_idx, group in groupby(rows, key=lambda r: int(r["step_idx"])):
        step_rows = list(group)
        for r in step_rows:
            local = str(r.get("cumulative_thought", ""))
            if prefix:
                r["cumulative_thought"] = _join_prefix_local(prefix, local, separator)
        prefix = str(step_rows[-1].get("cumulative_thought", ""))


def chain_file(
    input_path: Path,
    output_path: Path,
    separator: str = "\n\n",
    output_field: str | None = None,
) -> None:
    """
    Read JSONL, chain cumulative_thought per query_id, write JSONL.

    If output_field is set, write the chained text to that key and leave
    cumulative_thought unchanged; otherwise overwrite cumulative_thought.
    """
    records: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for order, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_order"] = order
            records.append(obj)

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_query[str(r["query_id"])].append(r)

    for rows in by_query.values():
        if output_field:
            for r in rows:
                r["_local_cumulative"] = str(r.get("cumulative_thought", ""))
            chain_rows_for_query(rows, separator)
            for r in rows:
                r[output_field] = r["cumulative_thought"]
                r["cumulative_thought"] = r.pop("_local_cumulative")
        else:
            chain_rows_for_query(rows, separator)

    records.sort(key=lambda r: int(r["_order"]))
    for r in records:
        r.pop("_order", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain cumulative_thought across step_idx within each query_id."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSONL (e.g. whole_set_cumulative_judged.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--separator",
        default="\n\n",
        help="Between prior-step tail and current-step local cumulative (default: blank line)",
    )
    parser.add_argument(
        "--output-field",
        default=None,
        metavar="NAME",
        help="If set, write chained text to this field and keep cumulative_thought as in input",
    )
    args = parser.parse_args()
    chain_file(
        args.input,
        args.output,
        separator=args.separator,
        output_field=args.output_field,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()