"""
Auto-grading for ToolBench trajectories.

Implements three strategies (A, B, C) to determine task_success:
  A. Solution Path Match — compare tool-call sequences against reference.
  B. LLM Judge — ask gpt-4o-mini whether the task was completed.
  C. Hybrid — auto-pass obvious matches, LLM judge for ambiguous cases.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from pact.config import TrajectoryLog


# ---------------------------------------------------------------------------
# Strategy A: Solution Path Match (free, deterministic)
# ---------------------------------------------------------------------------

def _normalize_action(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def action_exact_match(
    agent_steps: list[dict[str, Any]],
    reference_path: list[dict[str, Any]],
) -> float:
    """
    Fraction of reference actions that appear in the agent's tool calls
    (order-insensitive bag match).  Returns 0.0–1.0.
    """
    if not reference_path:
        return 1.0

    ref_actions = [_normalize_action(s.get("action", "")) for s in reference_path]
    agent_actions = []
    for step in agent_steps:
        for tc in step.get("tool_calls", []):
            agent_actions.append(_normalize_action(tc.get("tool_name", "")))

    ref_counts = Counter(ref_actions)
    agent_counts = Counter(agent_actions)
    matched = sum(min(ref_counts[a], agent_counts.get(a, 0)) for a in ref_counts)
    return matched / len(ref_actions) if ref_actions else 1.0


def argument_f1(
    agent_steps: list[dict[str, Any]],
    reference_path: list[dict[str, Any]],
) -> float:
    """
    Token-level F1 between agent arguments and reference arguments
    across all tool calls.
    """
    def _arg_tokens(args: dict) -> set[str]:
        tokens: set[str] = set()
        for v in args.values():
            tokens.update(str(v).lower().split())
        return tokens

    ref_tokens: set[str] = set()
    for step in reference_path:
        ref_tokens.update(_arg_tokens(step.get("action_input", {})))

    agent_tokens: set[str] = set()
    for step in agent_steps:
        for tc in step.get("tool_calls", []):
            agent_tokens.update(_arg_tokens(tc.get("arguments", {})))

    if not ref_tokens:
        return 1.0
    if not agent_tokens:
        return 0.0

    precision = len(ref_tokens & agent_tokens) / len(agent_tokens) if agent_tokens else 0
    recall = len(ref_tokens & agent_tokens) / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grade_path_match(
    log: TrajectoryLog,
    reference_path: list[dict[str, Any]],
    action_threshold: float = 0.5,
    f1_threshold: float = 0.3,
) -> tuple[bool, float, float]:
    """
    Strategy A grading.
    Returns (success, action_em_score, arg_f1_score).
    """
    steps_as_dicts = [
        {
            "tool_calls": [
                {"tool_name": tc.tool_name, "arguments": tc.arguments}
                for tc in step.tool_calls
            ]
        }
        for step in log.steps
    ]

    em = action_exact_match(steps_as_dicts, reference_path)
    f1 = argument_f1(steps_as_dicts, reference_path)
    success = em >= action_threshold and f1 >= f1_threshold
    return success, em, f1


# ---------------------------------------------------------------------------
# Strategy B: LLM Judge (outcome-based, costs ~$0.002/task)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a grading assistant. Given a user instruction and an agent's final \
output, determine whether the agent successfully completed the task.

IMPORTANT CONTEXT: The agent used a simulator that generates realistic-but-synthetic API \
responses (e.g., placeholder names, example URLs, fictional values). Treat these simulated \
responses as valid data — do NOT fail a task solely because the data looks synthetic. \
Judge instead on whether the agent took the correct actions, called the right tools, \
and produced a structured final answer that addresses the task.

Respond with ONLY one word: PASS or FAIL."""

_JUDGE_USER_TEMPLATE = """\
## Instruction
{instruction}

## Agent's Final Output
{final_output}

## Available Tools
{tools}

Did the agent successfully complete the task? Respond PASS or FAIL."""


# Models that use the Responses API with reasoning= support.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def grade_llm_judge(
    log: TrajectoryLog,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    client: OpenAI | None = None,
) -> tuple[bool, str]:
    """
    Strategy B grading via LLM judge.

    Uses the Responses API (with reasoning= support) for gpt-5 / o-series models,
    and Chat Completions for everything else.
    Returns (success, raw_judgment).
    """
    c = client or OpenAI()

    tools_str = ", ".join(log.available_tools[:20])
    user_msg = _JUDGE_USER_TEMPLATE.format(
        instruction=log.prompt,
        final_output=log.final_answer[:2000],
        tools=tools_str,
    )

    if _is_reasoning_model(model):
        resp = c.responses.create(
            model=model,
            reasoning={"effort": reasoning_effort},
            input=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_output_tokens=2048,
        )
        judgment = (resp.output_text or "").strip().upper()
    else:
        resp = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        judgment = resp.choices[0].message.content.strip().upper()

    return judgment.startswith("PASS"), judgment


# ---------------------------------------------------------------------------
# Strategy C: Hybrid (recommended)
# ---------------------------------------------------------------------------

def grade_hybrid(
    log: TrajectoryLog,
    reference_path: list[dict[str, Any]] | None = None,
    auto_pass_action_em: float = 0.8,
    auto_pass_f1: float = 0.5,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    client: OpenAI | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Strategy C: auto-pass if path match is strong, else fall back to LLM judge.
    Returns (success, details_dict).
    """
    details: dict[str, Any] = {"strategy": "hybrid"}

    if reference_path:
        success_a, em, f1 = grade_path_match(log, reference_path)
        details["action_em"] = em
        details["argument_f1"] = f1

        if em >= auto_pass_action_em and f1 >= auto_pass_f1:
            details["decided_by"] = "path_match_auto_pass"
            return True, details

        if em == 0.0:
            details["decided_by"] = "path_match_auto_fail"
            return False, details

    success_b, judgment = grade_llm_judge(
        log, model=model, reasoning_effort=reasoning_effort, client=client
    )
    details["llm_judgment"] = judgment
    details["decided_by"] = "llm_judge"
    return success_b, details


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------

def grade_logs(
    log_dir: str | Path,
    reference_paths: dict[str, list[dict[str, Any]]] | None = None,
    strategy: str = "hybrid",
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> dict[str, bool]:
    """
    Grade all trajectory logs in a directory.

    Args:
        log_dir: directory of TrajectoryLog JSON files.
        reference_paths: {query_id: solution_path} for path matching.
        strategy: 'path_match', 'llm_judge', or 'hybrid'.
        model: model for LLM judge (gpt-5 uses Responses API with reasoning).
        reasoning_effort: effort level for reasoning models ('low', 'medium', 'high').

    Returns:
        {query_id: task_success} mapping.
    """
    d = Path(log_dir)
    refs = reference_paths or {}
    results: dict[str, bool] = {}
    client = None

    if strategy in ("llm_judge", "hybrid"):
        client = OpenAI()

    for log_file in sorted(d.glob("*.json")):
        log = TrajectoryLog.load(log_file)
        ref = refs.get(log.query_id)

        if strategy == "path_match":
            success, _, _ = grade_path_match(log, ref or [])
        elif strategy == "llm_judge":
            success, _ = grade_llm_judge(
                log, model=model, reasoning_effort=reasoning_effort, client=client
            )
        else:
            success, _ = grade_hybrid(
                log, ref, model=model, reasoning_effort=reasoning_effort, client=client
            )

        log.task_success = success
        log.save(log_file)
        results[log.query_id] = success

    n_pass = sum(results.values())
    print(f"Graded {len(results)} logs: {n_pass}/{len(results)} PASS "
          f"({n_pass / len(results):.1%})" if results else "No logs found.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Grade Agent A trajectory logs")
    parser.add_argument("--log-dir", default="logs/toolbench_agent_a")
    parser.add_argument("--strategy", choices=["path_match", "llm_judge", "hybrid"],
                        default="hybrid")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for gpt-5/o-series models")
    parser.add_argument("--ref-paths", default=None,
                        help="JSON file mapping query_id -> solution_path")
    args = parser.parse_args()

    refs = None
    if args.ref_paths:
        refs = json.loads(Path(args.ref_paths).read_text())

    grade_logs(args.log_dir, reference_paths=refs,
               strategy=args.strategy, model=args.model,
               reasoning_effort=args.reasoning_effort)
