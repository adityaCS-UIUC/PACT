"""
Smoke test: run the full Agent A + ToolBench pipeline on 2-3 examples.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_pipeline.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── 0. Check API key ──────────────────────────────────────────────────────

def check_api_key() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key == "sk-...":
        print("ERROR: OPENAI_API_KEY not set.\n")
        print("Set it one of two ways:\n")
        print("  Option A (terminal):  export OPENAI_API_KEY='sk-...'")
        print("  Option B (.env file): create .env with OPENAI_API_KEY=sk-...")
        print("                        then: pip install python-dotenv")
        print("                        The script will auto-load it.\n")
        sys.exit(1)
    print(f"API key found: {key[:8]}...{key[-4:]}\n")

# Try loading from .env if python-dotenv is installed.
# override=True ensures the .env key takes priority over any existing env var.
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env", override=True)
    load_dotenv(_root / "pact" / ".env", override=True)
except ImportError:
    pass

check_api_key()


# ── 1. Test with built-in PACT tools (no ToolBench download) ─────────────

async def test_builtin() -> None:
    from pact.config import PACTConfig
    from pact.agent_a.react_agent import run_agent_a

    print("=" * 60)
    print("TEST 1: Agent A with built-in tools (2 tasks)")
    print("=" * 60)

    cfg = PACTConfig(model_name="gpt-4o-mini")

    tasks = [
        {
            "prompt": "What is the speed of light in km/s?",
            "ground_truth": "299,792",
            "query_id": "test_s01",
        },
        {
            "prompt": (
                "Find the speed of light from the document corpus, "
                "then convert it to miles per hour."
            ),
            "ground_truth": "670616",
            "query_id": "test_m01",
        },
    ]

    out_dir = Path("logs/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"\n--- Running {task['query_id']}: {task['prompt'][:50]}...")
        log = await run_agent_a(
            prompt=task["prompt"],
            config=cfg,
            query_id=task["query_id"],
            ground_truth=task.get("ground_truth"),
        )
        log.save(out_dir / f"{task['query_id']}.json")

        print(f"  Success:    {log.task_success}")
        print(f"  Steps:      {len(log.steps)}")
        print(f"  Latency:    {log.wall_clock_ms:.0f} ms")
        print(f"  Answer:     {log.final_answer[:120]}...")

        for i, step in enumerate(log.steps):
            if step.think_block:
                print(f"    Step {i} THINK: \"{step.think_block.full_text[:80]}...\"")
            for tc in step.tool_calls:
                print(f"    Step {i} ACTION: {tc.tool_name}({tc.arguments})")
                if tc.result:
                    print(f"             -> {tc.result[:80]}...")

    print("\nBuilt-in tools test done.\n")


# ── 2. Test ToolBench adapter (download 3 tasks) ─────────────────────────

async def test_toolbench_download() -> None:
    print("=" * 60)
    print("TEST 2: ToolBench adapter (download 3 tasks from G1)")
    print("=" * 60)

    from pact.data_pipeline.toolbench_adapter import load_benchmark_tasks

    tasks = load_benchmark_tasks(
        splits=["g1_instruction"],
        max_per_split=3,
    )

    print(f"\nDownloaded {len(tasks)} tasks:")
    for t in tasks:
        print(f"  [{t['query_id']}] {t['prompt'][:70]}...")
        print(f"    Tools: {t['available_tools'][:3]}...")
        print(f"    Category: {t['category']}")

    print("\nToolBench download test done.\n")
    return tasks


# ── 3. Test Agent A on ToolBench tasks (dynamic tools) ────────────────────

async def test_toolbench_agent_a(tasks: list) -> None:
    if not tasks:
        print("Skipping Agent A on ToolBench (no tasks).\n")
        return

    print("=" * 60)
    print(f"TEST 3: Agent A on {len(tasks)} ToolBench task(s)")
    print("=" * 60)

    from pact.config import PACTConfig
    from pact.agent_a.react_agent import run_agent_a
    from pact.tools.toolbench_tools import build_toolbench_registry

    cfg = PACTConfig(model_name="gpt-4o-mini")
    out_dir = Path("logs/smoke_test_toolbench")
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks[:3]:
        qid = task["query_id"]
        print(f"\n--- Running {qid}: {task['prompt'][:60]}...")

        try:
            task_registry = None
            if task.get("api_specs"):
                task_registry = build_toolbench_registry(task["api_specs"])
                print(f"    Dynamic tools: {task_registry.list_names()}")

            log = await run_agent_a(
                prompt=task["prompt"],
                config=cfg,
                query_id=qid,
                registry=task_registry,
            )
            log.save(out_dir / f"{qid}.json")

            print(f"  Steps:    {len(log.steps)}")
            print(f"  Latency:  {log.wall_clock_ms:.0f} ms")
            print(f"  Answer:   {(log.final_answer or '')[:120]}...")
            for i, step in enumerate(log.steps):
                if step.think_block:
                    print(f"    Step {i} THINK: \"{step.think_block.full_text[:80]}...\"")
                for tc in step.tool_calls:
                    print(f"    Step {i} ACTION: {tc.tool_name}({json.dumps(tc.arguments)[:80]})")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print("\nToolBench Agent A test done.\n")


# ── 4. Test relabeling on the built-in logs ───────────────────────────────

def test_relabel() -> None:
    print("=" * 60)
    print("TEST 4: Relabeling (on smoke test logs)")
    print("=" * 60)

    from pact.data_pipeline.relabel import relabel_all

    n = relabel_all(
        log_dir="logs/smoke_test",
        output_path="data/smoke_test/speculator_samples.jsonl",
        only_successes=False,
        use_llm_oracle=False,
    )
    print(f"  Generated {n} speculator training samples")

    out = Path("data/smoke_test/speculator_samples.jsonl")
    if out.exists():
        print("  Preview:")
        for line in out.read_text().strip().split("\n")[:3]:
            d = json.loads(line)
            lbl = d["label"]
            chunk = d["cumulative_thought"][:60]
            print(f"    label={lbl}  chunk=\"{chunk}...\"")

    print("\nRelabeling test done.\n")


# ── Main ──────────────────────────────────────────────────────────────────

async def main() -> None:
    print("PACT Pipeline Smoke Test")
    print("=" * 60)
    print()

    # Test 1: built-in tools
    await test_builtin()

    # Test 2: ToolBench download
    tb_tasks = await test_toolbench_download()

    # Test 3: Agent A on ToolBench
    await test_toolbench_agent_a(tb_tasks)

    # Test 4: Relabeling
    test_relabel()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print()
    print("Logs saved to:    logs/smoke_test/")
    print("Samples saved to: data/smoke_test/speculator_samples.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
