from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThoughtStats:
    rows: int = 0
    non_empty_rows: int = 0
    total_chars: int = 0
    total_words: int = 0

    @property
    def avg_chars_per_row(self) -> float:
        return self.total_chars / self.rows if self.rows else 0.0

    @property
    def avg_words_per_row(self) -> float:
        return self.total_words / self.rows if self.rows else 0.0

    @property
    def avg_chars_non_empty(self) -> float:
        return self.total_chars / self.non_empty_rows if self.non_empty_rows else 0.0

    @property
    def avg_words_non_empty(self) -> float:
        return self.total_words / self.non_empty_rows if self.non_empty_rows else 0.0


def compute_stats(path: Path, field: str) -> ThoughtStats:
    stats = ThoughtStats()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(field, "")).strip()

            stats.rows += 1
            stats.total_chars += len(text)
            stats.total_words += len(text.split())
            if text:
                stats.non_empty_rows += 1
    return stats


def print_stats(label: str, path: Path, stats: ThoughtStats, field: str) -> None:
    print(f"{label}: {path}")
    print(f"  Rows: {stats.rows}")
    print(f"  Non-empty {field} rows: {stats.non_empty_rows}")
    print(f"  Average chars/row: {stats.avg_chars_per_row:.2f}")
    print(f"  Average words/row: {stats.avg_words_per_row:.2f}")
    print(f"  Average chars/non-empty row: {stats.avg_chars_non_empty:.2f}")
    print(f"  Average words/non-empty row: {stats.avg_words_non_empty:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare average cumulative thought length between two JSONL files."
        )
    )
    parser.add_argument("--file-a", type=Path, required=True, help="First JSONL file")
    parser.add_argument("--file-b", type=Path, required=True, help="Second JSONL file")
    parser.add_argument(
        "--field",
        default="cumulative_thought",
        help="Text field to average (default: cumulative_thought)",
    )
    args = parser.parse_args()

    stats_a = compute_stats(args.file_a, args.field)
    stats_b = compute_stats(args.file_b, args.field)

    print_stats("File A", args.file_a, stats_a, args.field)
    print()
    print_stats("File B", args.file_b, stats_b, args.field)
    print()
    print("Difference (A - B):")
    print(f"  Avg chars/row: {stats_a.avg_chars_per_row - stats_b.avg_chars_per_row:.2f}")
    print(f"  Avg words/row: {stats_a.avg_words_per_row - stats_b.avg_words_per_row:.2f}")
    print(
        "  Avg chars/non-empty row: "
        f"{stats_a.avg_chars_non_empty - stats_b.avg_chars_non_empty:.2f}"
    )
    print(
        "  Avg words/non-empty row: "
        f"{stats_a.avg_words_non_empty - stats_b.avg_words_non_empty:.2f}"
    )


if __name__ == "__main__":
    main()
