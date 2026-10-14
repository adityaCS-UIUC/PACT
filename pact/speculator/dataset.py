"""
Dataset loader for speculator training.

Reads the JSONL produced by the relabeling pipeline and wraps it
into a HuggingFace Dataset for use with the Trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


def load_speculator_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_input_text(record: dict[str, Any]) -> str:
    """
    Format a single training example as the text the model will see.

    Layout:
        [PROMPT] <user prompt>
        [THOUGHT] <cumulative reasoning so far>
        [TOOLS] <available tool name(s) from the target call>
    """
    prompt = record["prompt"]
    thought = record["cumulative_thought"]
    tool = record["target_tool_call"]["tool_name"]
    return f"[PROMPT] {prompt}\n[THOUGHT] {thought}\n[TOOLS] {tool}"


def build_dataset(
    jsonl_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    test_fraction: float = 0.15,
) -> DatasetDict:
    """
    Load JSONL, tokenize, and return a train/test DatasetDict.
    """
    raw = load_speculator_jsonl(jsonl_path)

    texts = [build_input_text(r) for r in raw]
    labels = [r["label"] for r in raw]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": labels,
    })

    split = ds.train_test_split(test_size=test_fraction, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})
