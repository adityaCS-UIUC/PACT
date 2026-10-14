"""
Phase 5: Generate paper-ready plots from sweep results.

Produces:
  1. Accuracy vs Tau curve
  2. Latency vs Tau curve
  3. Accuracy-Latency tradeoff (Pareto front)
  4. Token cost vs Tau
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def plot_sweep(summary_path: str | Path, output_dir: str | Path = "results/plots") -> None:
    data = load_summary(summary_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline = [d for d in data if d["mode"] == "baseline"]
    pact = [d for d in data if d["mode"] == "pact"]
    pact.sort(key=lambda d: d["tau"])

    taus = [d["tau"] for d in pact]
    accs = [d["accuracy"] for d in pact]
    lats = [d["mean_latency_ms"] for d in pact]
    tokens = [d["mean_tokens"] for d in pact]
    tools = [d["mean_tool_calls"] for d in pact]

    baseline_acc = baseline[0]["accuracy"] if baseline else None
    baseline_lat = baseline[0]["mean_latency_ms"] if baseline else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("PACT: Speed-Accuracy Tradeoff (τ sweep)", fontsize=14)

    # --- 1. Accuracy vs Tau ---
    ax = axes[0, 0]
    ax.plot(taus, accs, "o-", color="tab:blue", label="PACT")
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="tab:red", linestyle="--", label="Baseline (Agent A)")
    ax.set_xlabel("τ (speculation threshold)")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Accuracy vs τ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. Latency vs Tau ---
    ax = axes[0, 1]
    ax.plot(taus, lats, "s-", color="tab:green", label="PACT")
    if baseline_lat is not None:
        ax.axhline(baseline_lat, color="tab:red", linestyle="--", label="Baseline")
    ax.set_xlabel("τ")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Latency vs τ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Accuracy vs Latency (Pareto) ---
    ax = axes[1, 0]
    ax.scatter(lats, accs, c=taus, cmap="viridis", s=80, edgecolors="k", zorder=3)
    for i, tau in enumerate(taus):
        ax.annotate(f"τ={tau}", (lats[i], accs[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    if baseline_acc is not None and baseline_lat is not None:
        ax.scatter([baseline_lat], [baseline_acc], color="red", marker="*",
                   s=200, zorder=4, label="Baseline")
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy-Latency Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Token cost vs Tau ---
    ax = axes[1, 1]
    ax.plot(taus, tokens, "D-", color="tab:orange", label="Mean tokens")
    ax2 = ax.twinx()
    ax2.plot(taus, tools, "^--", color="tab:purple", label="Mean tool calls")
    ax.set_xlabel("τ")
    ax.set_ylabel("Mean Tokens", color="tab:orange")
    ax2.set_ylabel("Mean Tool Calls", color="tab:purple")
    ax.set_title("Cost vs τ")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out / "pact_sweep.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot PACT sweep results")
    parser.add_argument("--summary", default="results/sweep/sweep_summary.json")
    parser.add_argument("--output", default="results/plots")
    args = parser.parse_args()
    plot_sweep(args.summary, args.output)
