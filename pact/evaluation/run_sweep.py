"""
Phase 5: Parameter sweep over tau values.

Runs Agent B at multiple tau thresholds on a held-out test set,
collects metrics, and saves results for plotting.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pact.config import PACTConfig
from pact.agent_b.pact_agent import PACTAgent
from pact.speculator.model import Speculator
from pact.evaluation.metrics import compute_metrics, RunMetrics


DEFAULT_TAUS = [0.5, 0.7, 0.85, 0.9, 0.95, 0.99]


async def run_sweep(
    tasks: list[dict[str, Any]],
    taus: list[float] | None = None,
    base_config: PACTConfig | None = None,
    speculator: Speculator | None = None,
    output_dir: str = "results/sweep",
) -> list[RunMetrics]:
    """
    For each tau value, run Agent B on all tasks and collect metrics.
    Also runs Agent A (baseline, tau=None) for comparison.
    """
    tau_list = taus or DEFAULT_TAUS
    cfg = base_config or PACTConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_metrics: list[RunMetrics] = []

    # --- Baseline (Agent A, no speculation) ---
    print("=" * 60)
    print("Running BASELINE (Agent A, no tau)")
    print("=" * 60)
    baseline_cfg = PACTConfig(model_name=cfg.model_name)
    baseline_agent = PACTAgent(config=baseline_cfg, speculator=None)
    baseline_logs = await baseline_agent.run_batch(
        tasks, save_dir=str(out / "baseline")
    )
    baseline_m = compute_metrics(baseline_logs, tau=None)
    baseline_m.mode = "baseline"
    all_metrics.append(baseline_m)
    print(f"Baseline: accuracy={baseline_m.accuracy:.2%}  "
          f"latency={baseline_m.mean_latency_ms:.0f}ms\n")

    # --- Sweep over tau values ---
    for tau in tau_list:
        print("=" * 60)
        print(f"Running PACT with tau={tau}")
        print("=" * 60)

        sweep_cfg = PACTConfig(model_name=cfg.model_name, tau=tau)
        agent_b = PACTAgent(
            config=sweep_cfg,
            speculator=speculator,
        )
        logs = await agent_b.run_batch(
            tasks, save_dir=str(out / f"tau_{tau}")
        )
        m = compute_metrics(logs, tau=tau)
        all_metrics.append(m)
        print(f"tau={tau}: accuracy={m.accuracy:.2%}  "
              f"latency={m.mean_latency_ms:.0f}ms\n")

    # --- Save summary ---
    summary = []
    for m in all_metrics:
        summary.append({
            "tau": m.tau,
            "mode": m.mode,
            "accuracy": m.accuracy,
            "mean_latency_ms": m.mean_latency_ms,
            "median_latency_ms": m.median_latency_ms,
            "p95_latency_ms": m.p95_latency_ms,
            "mean_tokens": m.mean_tokens,
            "mean_tool_calls": m.mean_tool_calls,
            "num_tasks": m.num_tasks,
            "num_success": m.num_success,
        })

    summary_path = out / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSweep summary saved to {summary_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    import argparse
    from pact.agent_a.react_agent import SAMPLE_TASKS

    parser = argparse.ArgumentParser(description="PACT tau sweep")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="results/sweep")
    parser.add_argument(
        "--taus", nargs="+", type=float,
        default=DEFAULT_TAUS,
    )
    args = parser.parse_args()

    cfg = PACTConfig(model_name=args.model)
    await run_sweep(
        tasks=SAMPLE_TASKS,
        taus=args.taus,
        base_config=cfg,
        speculator=None,  # pass a trained Speculator here
        output_dir=args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())
