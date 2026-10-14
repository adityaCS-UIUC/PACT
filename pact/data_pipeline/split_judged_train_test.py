"""
Split judged cumulative JSONL into train / test by whole task (query_id).

Each task is all lines sharing the same ``query_id``. The script:
  1) discovers unique ``query_id`` values in the input,
  2) randomly selects ``--test-tasks`` ids for the test split (default 500),
  3) streams the input again and writes every row to either train or test.

Use ``--seed`` for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _collect_query_ids(path: Path) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid = str(json.loads(line).get("query_id", ""))
            if not qid:
                continue
            if qid not in seen:
                seen.add(qid)
                order.append(qid)
    return order


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split judged JSONL into train/test by whole query_id (task)",
    )
    parser.add_argument("--input", type=Path, required=True, help="Judged JSONL (e.g. *_cumulative_judged.jsonl)")
    parser.add_argument("--output-train", type=Path, required=True, help="Training JSONL output")
    parser.add_argument("--output-test", type=Path, required=True, help="Test JSONL output")
    parser.add_argument("--test-tasks", type=int, default=500, help="Number of whole tasks (query_ids) for test")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for task sampling")
    args = parser.parse_args()

    query_ids = _collect_query_ids(args.input)
    n_tasks = len(query_ids)
    if n_tasks == 0:
        raise SystemExit(f"No query_id values found in {args.input}")

    k = min(args.test_tasks, n_tasks)
    if args.test_tasks > n_tasks:
        print(
            f"WARNING: only {n_tasks} tasks in file; using test_tasks={k} "
            f"(requested {args.test_tasks}).",
            flush=True,
        )

    rng = random.Random(args.seed)
    test_ids = set(rng.sample(query_ids, k=k))

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    args.output_test.parent.mkdir(parents=True, exist_ok=True)

    n_train = n_test = 0
    with (
        args.input.open(encoding="utf-8") as inp,
        args.output_train.open("w", encoding="utf-8") as f_train,
        args.output_test.open("w", encoding="utf-8") as f_test,
    ):
        for line in inp:
            raw = line.strip()
            if not raw:
                continue
            row: dict[str, Any] = json.loads(raw)
            qid = str(row.get("query_id", ""))
            if qid in test_ids:
                f_test.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_test += 1
            else:
                f_train.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_train += 1

    train_task_count = n_tasks - k
    print(f"Unique tasks: {n_tasks}")
    print(f"Test tasks (sampled): {k}  |  Train tasks: {train_task_count}")
    print(f"Test rows written: {n_test} -> {args.output_test}")
    print(f"Train rows written: {n_train} -> {args.output_train}")


if __name__ == "__main__":
    main()
