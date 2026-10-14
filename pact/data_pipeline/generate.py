"""
Data generation driver.

Supports two modes:
  1. Built-in tasks  — uses the PACT tool suite (RAG + extended).
  2. ToolBench tasks — downloads ToolBench from HuggingFace, builds
     dynamic per-task tools, generates trajectories, auto-grades them,
     then relabels into speculator training data.

End-to-end pipeline:
  Download tasks -> Run Agent A -> Grade -> Relabel -> JSONL
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_root / ".env", override=True)
    load_dotenv(_root / "pact" / ".env", override=True)
except ImportError:
    pass

from pact.config import PACTConfig
from pact.agent_a.react_agent import run_batch, SAMPLE_TASKS
from pact.data_pipeline.relabel import relabel_all


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    """
    Load tasks from a JSONL file.  Each line must have at least {"prompt": "..."}.
    Optional keys: query_id, ground_truth, api_specs, category.
    """
    tasks: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def save_sample_tasks(path: str | Path) -> None:
    """Write the built-in sample tasks to a JSONL file for reference."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for t in SAMPLE_TASKS:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(SAMPLE_TASKS)} sample tasks to {p}")


# ---------------------------------------------------------------------------
# Full pipeline (built-in tasks)
# ---------------------------------------------------------------------------

async def generate_training_data(
    tasks: list[dict[str, Any]] | None = None,
    config: PACTConfig | None = None,
    log_dir: str = "logs/agent_a",
    output_path: str = "data/training/speculator_train.jsonl",
) -> None:
    """
    End-to-end data generation with built-in tasks:
      1. Run Agent A on all tasks -> raw logs.
      2. Relabel successful logs -> training JSONL.
    """
    cfg = config or PACTConfig()
    task_list = tasks or SAMPLE_TASKS

    print("=" * 60)
    print("Phase 1: Running Agent A (baseline) over tasks")
    print("=" * 60)
    logs = await run_batch(task_list, config=cfg, save_dir=log_dir)
    n_success = sum(l.task_success for l in logs)
    print(f"\n{n_success}/{len(logs)} tasks succeeded.\n")

    print("=" * 60)
    print("Phase 2: Relabeling successful trajectories")
    print("=" * 60)
    n_samples = relabel_all(log_dir, output_path, only_successes=True)
    print(f"\nPipeline complete. {n_samples} training samples generated.")


# ---------------------------------------------------------------------------
# ToolBench pipeline
# ---------------------------------------------------------------------------

async def generate_toolbench_data(
    config: PACTConfig | None = None,
    splits: list[str] | None = None,
    max_per_split: int = 7000,
    max_total: int | None = None,
    include_training: bool = False,
    max_training: int = 2000,
    log_dir: str = "logs/toolbench_agent_a",
    output_path: str = "data/training/speculator_toolbench.jsonl",
    grading_strategy: str = "hybrid",
    cache_dir: str | None = None,
) -> None:
    """
    End-to-end ToolBench data generation pipeline:
      1. Download & adapt ToolBench tasks from HuggingFace.
      2. Run Agent A with dynamic per-task tools.
      3. Auto-grade trajectories (hybrid: path match + LLM judge).
      4. Relabel successful trajectories -> training JSONL.
    """
    cfg = config or PACTConfig()

    if cache_dir:
        from pact.tools.toolbench_tools import set_cache_dir
        set_cache_dir(cache_dir)

    # --- Step 1: Download & adapt ---
    print("=" * 60)
    print("Step 1: Downloading ToolBench and adapting to PACT format")
    print("=" * 60)

    from pact.data_pipeline.toolbench_adapter import (
        load_benchmark_tasks,
        load_training_tasks,
        export_tasks_jsonl,
    )

    target_splits = splits or ["g1_instruction", "g2_instruction", "g3_instruction"]
    per_split = max_per_split
    if max_total is not None:
        per_split = max(1, (max_total + len(target_splits) - 1) // len(target_splits))
    tasks = load_benchmark_tasks(target_splits, max_per_split=per_split)
    if max_total is not None:
        tasks = tasks[:max_total]

    if include_training:
        train_tasks = load_training_tasks(max_tasks=max_training)
        tasks.extend(train_tasks)

    tasks_path = Path(log_dir).parent / "toolbench_tasks.jsonl"
    export_tasks_jsonl(tasks, tasks_path)

    # --- Step 2: Run Agent A ---
    print("\n" + "=" * 60)
    print(f"Step 2: Running Agent A on {len(tasks)} ToolBench tasks")
    print("=" * 60)

    logs = await run_batch(tasks, config=cfg, save_dir=log_dir)

    from pact.tools.toolbench_tools import get_simulator
    get_simulator().print_stats()

    # --- Step 3: Auto-grade ---
    print("\n" + "=" * 60)
    print("Step 3: Auto-grading trajectories")
    print("=" * 60)

    ref_paths: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        gt = task.get("ground_truth_path", [])
        if gt:
            ref_paths[task["query_id"]] = gt

    from pact.data_pipeline.toolbench_grader import grade_logs
    grade_logs(
        log_dir=log_dir,
        reference_paths=ref_paths if ref_paths else None,
        strategy=grading_strategy,
        model=cfg.model_name,
    )

    # --- Step 4: Relabel ---
    print("\n" + "=" * 60)
    print("Step 4: Relabeling successful trajectories")
    print("=" * 60)

    n_samples = relabel_all(
        log_dir,
        output_path,
        only_successes=True,
        use_llm_oracle=True,
    )
    print(f"\nToolBench pipeline complete. {n_samples} training samples generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate speculator training data")
    parser.add_argument("--mode", choices=["builtin", "toolbench"], default="builtin",
                        help="'builtin' uses PACT sample tasks; 'toolbench' downloads from HF")
    parser.add_argument("--tasks", default=None,
                        help="Path to tasks JSONL (overrides mode)")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")

    parser.add_argument("--splits", nargs="+",
                        default=["g1_instruction", "g2_instruction", "g3_instruction"])
    parser.add_argument("--max-per-split", type=int, default=7000)
    parser.add_argument("--max-total", type=int, default=None,
                        help="Cap total benchmark tasks across splits (e.g. 12 for quick data-quality check)")
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument("--max-training", type=int, default=2000)
    parser.add_argument("--grading-strategy", choices=["path_match", "llm_judge", "hybrid"],
                        default="hybrid")
    parser.add_argument("--cache-dir", default=None,
                        help="MirrorAPI cache directory")

    args = parser.parse_args()
    cfg = PACTConfig(model_name=args.model)

    if args.tasks:
        tasks = load_tasks(args.tasks)
        log_dir = args.log_dir or "logs/agent_a"
        output = args.output or "data/training/speculator_train.jsonl"
        await generate_training_data(tasks, cfg, log_dir, output)

    elif args.mode == "toolbench":
        log_dir = args.log_dir or "logs/toolbench_agent_a"
        output = args.output or "data/training/speculator_toolbench.jsonl"
        await generate_toolbench_data(
            config=cfg,
            splits=args.splits,
            max_per_split=args.max_per_split,
            max_total=args.max_total,
            include_training=args.include_training,
            max_training=args.max_training,
            log_dir=log_dir,
            output_path=output,
            grading_strategy=args.grading_strategy,
            cache_dir=args.cache_dir,
        )

    else:
        log_dir = args.log_dir or "logs/agent_a"
        output = args.output or "data/training/speculator_train.jsonl"
        await generate_training_data(None, cfg, log_dir, output)


if __name__ == "__main__":
    asyncio.run(main())
