"""
Metrics collection for PACT evaluation.

Computes latency, accuracy (task success rate), token cost, and
tool-call statistics from a set of TrajectoryLogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pact.config import TrajectoryLog


@dataclass
class RunMetrics:
    """Aggregate metrics for a single evaluation run (fixed tau)."""
    tau: float | None
    mode: str
    num_tasks: int
    num_success: int
    accuracy: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    mean_tokens: float
    total_tool_calls: int
    mean_tool_calls: float


def compute_metrics(
    logs: list[TrajectoryLog],
    tau: float | None = None,
) -> RunMetrics:
    """Compute aggregate metrics from a list of trajectory logs."""
    n = len(logs)
    if n == 0:
        return RunMetrics(
            tau=tau, mode="unknown", num_tasks=0, num_success=0,
            accuracy=0.0, mean_latency_ms=0.0, median_latency_ms=0.0,
            p95_latency_ms=0.0, total_tokens=0, mean_tokens=0.0,
            total_tool_calls=0, mean_tool_calls=0.0,
        )

    successes = sum(1 for l in logs if l.task_success)
    latencies = sorted(l.wall_clock_ms for l in logs)
    tokens = [l.total_tokens for l in logs]
    tool_counts = [
        sum(len(s.tool_calls) for s in l.steps)
        for l in logs
    ]

    return RunMetrics(
        tau=tau or (logs[0].tau if logs else None),
        mode=logs[0].mode,
        num_tasks=n,
        num_success=successes,
        accuracy=successes / n,
        mean_latency_ms=sum(latencies) / n,
        median_latency_ms=latencies[n // 2],
        p95_latency_ms=latencies[int(n * 0.95)],
        total_tokens=sum(tokens),
        mean_tokens=sum(tokens) / n,
        total_tool_calls=sum(tool_counts),
        mean_tool_calls=sum(tool_counts) / n,
    )


def load_and_compute(log_dir: str | Path, tau: float | None = None) -> RunMetrics:
    """Load all trajectory JSONs from a directory and compute metrics."""
    d = Path(log_dir)
    logs = [TrajectoryLog.load(p) for p in sorted(d.glob("*.json"))]
    return compute_metrics(logs, tau=tau)
