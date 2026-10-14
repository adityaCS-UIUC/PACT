"""Shared configuration and data structures for the PACT framework."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class AgentMode(Enum):
    BASELINE = "baseline"  # Agent A: full sequential ReAct
    PACT = "pact"          # Agent B: speculative execution with tau


@dataclass
class PACTConfig:
    # --- Model ---
    model_name: str = "gpt-4o-mini"
    speculator_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    speculator_adapter_path: str | None = None

    # --- PACT knobs ---
    tau: float = 0.85
    max_steps: int = 10
    max_think_tokens: int = 1024

    # --- Agent mode ---
    mode: AgentMode = AgentMode.BASELINE

    # --- Paths ---
    log_dir: str = "logs/raw"
    training_data_dir: str = "data/training"


# ---------------------------------------------------------------------------
# Trajectory data structures (used for logging and data generation)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class ThinkBlock:
    full_text: str
    sentences: list[str] = field(default_factory=list)
    truncated_at_sentence: int | None = None


@dataclass
class AgentStep:
    step_idx: int
    think_block: ThinkBlock | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observation: str = ""


@dataclass
class TrajectoryLog:
    query_id: str
    prompt: str
    available_tools: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    task_success: bool = False
    total_tokens: int = 0
    wall_clock_ms: float = 0.0
    mode: str = "baseline"
    tau: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryLog":
        raw = json.loads(Path(path).read_text())
        steps = []
        for s in raw.pop("steps", []):
            tb = s.pop("think_block", None)
            if tb:
                tb = ThinkBlock(**tb)
            tcs = [ToolCallRecord(**tc) for tc in s.pop("tool_calls", [])]
            steps.append(AgentStep(think_block=tb, tool_calls=tcs, **s))
        return cls(steps=steps, **raw)


# ---------------------------------------------------------------------------
# Training data record (output of relabeling pipeline)
# ---------------------------------------------------------------------------

@dataclass
class SpeculatorSample:
    """One training sample for the speculator classifier."""
    query_id: str
    prompt: str
    cumulative_thought: str
    target_tool_call: dict[str, Any]
    label: int  # 0 = keep thinking, 1 = ready to execute


class Timer:
    """Simple context-manager timer returning elapsed ms."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
