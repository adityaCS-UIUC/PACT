"""
ToolBench / StableToolBench adapter.

Downloads the ToolBench dataset from HuggingFace and converts it into
the PACT task JSONL format that generate.py consumes.

Handles both:
  - Benchmark splits (G1/G2/G3) for evaluation/data-gen
  - Training split conversations for extracting ground-truth solution paths
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset, DatasetDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOLBENCH_HF_REPO = "tuandunghcmut/toolbench-v1"
BENCHMARK_CONFIG = "benchmark"
DEFAULT_CONFIG = "default"

SPLIT_CATEGORY_MAP = {
    "g1_instruction": "G1_single",
    "g1_category": "G1_category",
    "g1_tool": "G1_tool",
    "g2_instruction": "G2_intra_category",
    "g2_category": "G2_category",
    "g3_instruction": "G3_inter_category",
}


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_json_field(raw: str | list | dict) -> Any:
    """Safely parse a JSON string field (api_list / relevant_apis)."""
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_solution_path_from_conversations(
    conversations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Extract the ground-truth solution path from a ToolBench training
    conversation.  Each assistant turn may contain:
        Thought: ...
        Action: <tool_name>
        Action Input: <json args>
    Followed by a function turn with the observation.
    """
    path: list[dict[str, Any]] = []
    for msg in conversations:
        if msg.get("from") != "assistant":
            continue
        text = msg.get("value", "")

        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(\S+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*\})", text, re.DOTALL)

        if action_match:
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip()
            try:
                action_input = json.loads(input_match.group(1)) if input_match else {}
            except json.JSONDecodeError:
                action_input = {}
            path.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
            })

    return path


def _extract_instruction_from_conversations(
    conversations: list[dict[str, str]],
) -> str:
    """Pull the user instruction from a conversation list."""
    for msg in conversations:
        if msg.get("from") == "user":
            return msg.get("value", "")
    return ""


def _extract_tools_from_system(
    conversations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Extract tool/API definitions from the system message.
    ToolBench system messages list available tools in a structured way.
    """
    for msg in conversations:
        if msg.get("from") == "system":
            text = msg.get("value", "")
            tools: list[dict[str, Any]] = []
            func_blocks = re.findall(
                r"Function Name:\s*(\S+).*?Description:\s*(.*?)(?=Function Name:|$)",
                text, re.DOTALL,
            )
            for name, desc in func_blocks:
                tools.append({
                    "api_name": name.strip(),
                    "api_description": desc.strip()[:200],
                })
            if tools:
                return tools
    return []


# ---------------------------------------------------------------------------
# Public: load benchmark splits
# ---------------------------------------------------------------------------

def load_benchmark_tasks(
    splits: list[str] | None = None,
    max_per_split: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load ToolBench benchmark tasks (G1/G2/G3) from HuggingFace.

    Returns a list of PACT-format task dicts:
        {query_id, prompt, available_tools, api_specs, category, ground_truth_path}
    """
    target_splits = splits or ["g1_instruction", "g2_instruction", "g3_instruction"]

    ds = load_dataset(TOOLBENCH_HF_REPO, BENCHMARK_CONFIG)

    tasks: list[dict[str, Any]] = []
    for split_name in target_splits:
        if split_name not in ds:
            print(f"  Warning: split '{split_name}' not found, skipping.")
            continue

        split_data = ds[split_name]
        category = SPLIT_CATEGORY_MAP.get(split_name, split_name)
        count = 0

        for row in split_data:
            qid = str(row.get("query_id", f"{split_name}_{count}"))
            query = row.get("query", "")
            api_list = _parse_json_field(row.get("api_list", "[]"))
            relevant_apis = _parse_json_field(row.get("relevant_apis", "[]"))

            tool_names = []
            api_specs: list[dict[str, Any]] = []
            for api in api_list:
                if isinstance(api, dict):
                    name = api.get("api_name", api.get("tool_name", "unknown"))
                    tool_names.append(name)
                    api_specs.append(api)
                elif isinstance(api, str):
                    tool_names.append(api)

            tasks.append({
                "query_id": f"tb_{qid}",
                "prompt": query,
                "available_tools": tool_names,
                "api_specs": api_specs,
                "relevant_apis": relevant_apis,
                "category": category,
                "ground_truth_path": [],
            })
            count += 1
            if max_per_split and count >= max_per_split:
                break

        print(f"  Loaded {count} tasks from {split_name} ({category})")

    return tasks


# ---------------------------------------------------------------------------
# Public: load training conversations (for ground-truth solution paths)
# ---------------------------------------------------------------------------

def load_training_tasks(
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load ToolBench training data and extract instruction + solution paths.
    These provide ground-truth (thought, action, action_input) sequences.
    """
    ds = load_dataset(TOOLBENCH_HF_REPO, DEFAULT_CONFIG, split="train")

    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        convs = row.get("conversations", [])
        if not convs:
            continue

        instruction = _extract_instruction_from_conversations(convs)
        if not instruction:
            continue

        solution_path = _extract_solution_path_from_conversations(convs)
        tools = _extract_tools_from_system(convs)
        tool_names = [t["api_name"] for t in tools]

        tasks.append({
            "query_id": f"tb_train_{row.get('id', i)}",
            "prompt": instruction,
            "available_tools": tool_names,
            "api_specs": tools,
            "category": "training",
            "ground_truth_path": solution_path,
        })

        if max_tasks and len(tasks) >= max_tasks:
            break

    print(f"  Loaded {len(tasks)} training tasks with solution paths")
    return tasks


# ---------------------------------------------------------------------------
# Public: export to PACT JSONL
# ---------------------------------------------------------------------------

def export_tasks_jsonl(tasks: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write PACT-format tasks to a JSONL file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"Exported {len(tasks)} tasks to {p}")


# ---------------------------------------------------------------------------
# CLI: download & export
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download ToolBench and export PACT tasks")
    parser.add_argument("--splits", nargs="+",
                        default=["g1_instruction", "g2_instruction", "g3_instruction"])
    parser.add_argument("--max-per-split", type=int, default=7000)
    parser.add_argument("--include-training", action="store_true",
                        help="Also load training conversations for ground-truth paths")
    parser.add_argument("--max-training", type=int, default=2000)
    parser.add_argument("--output", default="data/toolbench/pact_tasks.jsonl")
    args = parser.parse_args()

    print("Loading ToolBench benchmark splits...")
    tasks = load_benchmark_tasks(args.splits, max_per_split=args.max_per_split)

    if args.include_training:
        print("\nLoading ToolBench training conversations...")
        train_tasks = load_training_tasks(max_tasks=args.max_training)
        tasks.extend(train_tasks)

    export_tasks_jsonl(tasks, args.output)
    print(f"\nDone. Total tasks: {len(tasks)}")


if __name__ == "__main__":
    main()
