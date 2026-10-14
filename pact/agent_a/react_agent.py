"""
Agent A: Sequential ReAct baseline using a manual loop.

Uses the OpenAI Chat Completions API directly (not the SDK's auto tool-calling)
so that we **always** capture the model's reasoning text before each action.
This is critical: the speculator trains on (cumulative_thought -> label) pairs,
so every step MUST produce thinking text.

Loop:
  1. Model generates text with <think>...</think> then Action/Action Input
  2. We parse the action
  3. Execute the tool via the ToolRegistry
  4. Append observation to conversation
  5. Repeat until "Final Answer:" or max steps
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import openai

from pact.config import (
    PACTConfig,
    TrajectoryLog,
    AgentStep,
    ThinkBlock,
    ToolCallRecord,
    Timer,
)
from pact.tools.rag_tools import build_rag_registry
from pact.tools.extended_tools import build_full_registry
from pact.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# System prompt — forces explicit <think> + Action text format
# ---------------------------------------------------------------------------

def _build_system_prompt(registry: ToolRegistry) -> str:
    tool_desc_lines = []
    for spec in registry.list_specs():
        params = spec.parameters.get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'string')}" for k, v in params.items()
        )
        tool_desc_lines.append(f"  - {spec.name}({param_str}): {spec.description[:120]}")
    tool_block = "\n".join(tool_desc_lines)

    return f"""\
You are a research assistant that answers questions using the available tools.

YOU MUST follow this EXACT output format for EVERY turn. Do not deviate.

FORMAT WHEN CALLING A TOOL:

<think>
Your step-by-step reasoning here. What do you know? What do you still need?
Which tool should you call and why? What arguments should you use?
</think>
Action: tool_name
Action Input: {{"key": "value"}}

FORMAT WHEN GIVING THE FINAL ANSWER:

<think>
Your final reasoning here.
</think>
Final Answer: your answer here

CRITICAL RULES:
1. ALWAYS write <think>...</think> BEFORE every Action or Final Answer. NEVER skip it.
2. "Action:" must contain ONLY the tool name, nothing else. No parentheses, no arguments.
3. "Action Input:" must be valid JSON on the SAME line or immediately after.
4. After each tool call you will see "Observation: <result>". Use it in your next step.
5. Many tasks require MULTIPLE tools in sequence. Plan the full chain.
6. Do NOT call tools you don't need.
7. ONE action per turn ONLY. Stop immediately after "Action Input: {{...}}". Do NOT write a second Action: in the same response.
8. NEVER write "Final Answer" without first calling at least one relevant tool. If unsure of arguments, use your best guess.

EXAMPLE:

<think>
The user wants the speed of light in mph. I know it's ~299,792 km/s. First I should
look it up to confirm, then convert the units.
</think>
Action: search
Action Input: {{"query": "speed of light", "max_results": 3}}

Available tools:
{tool_block}
"""


# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)

# Matches "Action: tool_name" and also "Action: tool_name(args)" inline style
_ACTION_RE = re.compile(r"Action:\s*(\S+?)(?:\(|$|\s)", re.IGNORECASE)
# Fallback: extract inline args from "Action: tool_name({...})" or "Action: tool_name(arg1, arg2)"
_INLINE_ARGS_RE = re.compile(r"Action:\s*\S+?\((.+)\)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from text, ignoring trailing content."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
                    depth = 0
    return None


def _parse_response(text: str) -> dict[str, Any]:
    """Parse the model's text output into think, action, action_input, final_answer."""
    result: dict[str, Any] = {
        "think": "",
        "action": None,
        "action_input": {},
        "final_answer": None,
        "raw": text,
    }

    think_match = _THINK_RE.search(text)
    if think_match:
        result["think"] = think_match.group(1).strip()

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # Try explicit "Action Input:" first — extract first balanced JSON only,
    # so multi-action blobs (where model wrote a second <think>/Action after the JSON)
    # don't end up as _raw strings that confuse the simulator.
    input_match = _ACTION_INPUT_RE.search(text)
    if input_match:
        raw_input = input_match.group(1).strip()
        extracted = _extract_json(raw_input)
        if extracted is not None:
            result["action_input"] = extracted
        else:
            try:
                result["action_input"] = json.loads(raw_input)
            except json.JSONDecodeError:
                result["action_input"] = {"_raw": raw_input}

    action_match = _ACTION_RE.search(text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Fallback: if no "Action Input:" but args are inline like tool_name({"key": "val"})
    if not input_match and result["action"]:
        inline_match = _INLINE_ARGS_RE.search(text)
        if inline_match:
            raw_inline = inline_match.group(1).strip()
            # Try parsing as JSON first (model wrote tool_name({"key": "val"}))
            try:
                result["action_input"] = json.loads(raw_inline)
            except json.JSONDecodeError:
                # Try wrapping as JSON object from keyword args: key="val", key2=val2
                pairs = re.findall(r'(\w+)\s*=\s*["\']?([^"\',$]+)["\']?', raw_inline)
                if pairs:
                    result["action_input"] = {k: v.strip() for k, v in pairs}
                else:
                    result["action_input"] = {"_raw": raw_inline}

    return result


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Default registry (built-in PACT tools)
# ---------------------------------------------------------------------------

_default_registry: ToolRegistry | None = None


def _get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = build_full_registry()
    return _default_registry


# ---------------------------------------------------------------------------
# Core: run Agent A on a single prompt
# ---------------------------------------------------------------------------

async def run_agent_a(
    prompt: str,
    config: PACTConfig | None = None,
    query_id: str | None = None,
    ground_truth: str | None = None,
    registry: ToolRegistry | None = None,
    tools: list | None = None,
) -> TrajectoryLog:
    """
    Execute Agent A on a single prompt using a manual ReAct loop.

    Args:
        registry: ToolRegistry to use. If None, uses the default built-in tools.
        tools: Ignored (kept for backward compat with test_pipeline).
              Use `registry` for custom tool sets.
    """
    cfg = config or PACTConfig()
    qid = query_id or str(uuid.uuid4())[:8]

    reg = registry or _get_default_registry()
    system_prompt = _build_system_prompt(reg)

    client = openai.AsyncOpenAI()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    log = TrajectoryLog(
        query_id=qid,
        prompt=prompt,
        available_tools=reg.list_names(),
        mode="baseline",
    )

    with Timer() as timer:
        total_tokens = 0

        for step_idx in range(cfg.max_steps):
            resp = await client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=cfg.max_think_tokens,
            )

            choice = resp.choices[0]
            text = choice.message.content or ""
            total_tokens += (resp.usage.total_tokens if resp.usage else 0)

            parsed = _parse_response(text)

            think_text = parsed["think"]
            think_block = None
            if think_text:
                think_block = ThinkBlock(
                    full_text=think_text,
                    sentences=_split_sentences(think_text),
                )

            if parsed["final_answer"] is not None:
                step = AgentStep(
                    step_idx=step_idx,
                    think_block=think_block,
                    tool_calls=[],
                    observation=None,
                )
                log.steps.append(step)
                log.final_answer = parsed["final_answer"]
                break

            action = parsed["action"]
            action_input = parsed["action_input"]

            if action and action in reg.list_names():
                record = reg.execute(action, action_input)

                step = AgentStep(
                    step_idx=step_idx,
                    think_block=think_block,
                    tool_calls=[record],
                    observation=record.result or record.error,
                )
                log.steps.append(step)

                observation = record.result if not record.error else f"Error: {record.error}"
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            elif action:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Observation: Error — tool '{action}' not found. "
                        f"Available tools: {reg.list_names()}"
                    ),
                })
                step = AgentStep(
                    step_idx=step_idx,
                    think_block=think_block,
                    tool_calls=[ToolCallRecord(
                        tool_name=action,
                        arguments=action_input,
                        error=f"Tool '{action}' not found",
                    )],
                    observation=f"Tool '{action}' not found",
                )
                log.steps.append(step)

            else:
                if not log.final_answer:
                    log.final_answer = text
                break

        else:
            if not log.final_answer:
                log.final_answer = "(Max steps reached)"

    log.total_tokens = total_tokens
    log.wall_clock_ms = timer.elapsed_ms

    if ground_truth and log.final_answer:
        log.task_success = ground_truth.lower() in log.final_answer.lower()

    return log


# ---------------------------------------------------------------------------
# Build a ToolRegistry from ToolBench API specs
# ---------------------------------------------------------------------------

def _build_toolbench_registry(api_specs: list[dict[str, Any]]) -> ToolRegistry:
    from pact.tools.toolbench_tools import build_toolbench_registry
    return build_toolbench_registry(api_specs)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    tasks: list[dict[str, Any]],
    config: PACTConfig | None = None,
    save_dir: str | None = None,
) -> list[TrajectoryLog]:
    """
    Run Agent A on a batch of tasks.

    Each task dict must have:
        - "prompt": str
    Optional keys:
        - "query_id": str
        - "ground_truth": str
        - "api_specs": list[dict]  (ToolBench API specs -> dynamic tools)
    """
    cfg = config or PACTConfig()
    out_dir = Path(save_dir or cfg.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: list[TrajectoryLog] = []
    for i, task in enumerate(tasks):
        qid = task.get("query_id", f"q{i:04d}")
        print(f"[Agent A] Running {qid}: {task['prompt'][:60]}...")

        task_registry = None
        if task.get("api_specs"):
            task_registry = _build_toolbench_registry(task["api_specs"])

        try:
            log = await run_agent_a(
                prompt=task["prompt"],
                config=cfg,
                query_id=qid,
                ground_truth=task.get("ground_truth"),
                registry=task_registry,
            )
        except Exception as e:
            print(f"  -> ERROR: {type(e).__name__}: {e}")
            log = TrajectoryLog(
                query_id=qid,
                prompt=task["prompt"],
                available_tools=[],
                mode="baseline",
                final_answer=f"Error: {e}",
                task_success=False,
            )

        log.save(out_dir / f"{qid}.json")
        logs.append(log)
        n_thinks = sum(1 for s in log.steps if s.think_block)
        print(f"  -> success={log.task_success}  steps={len(log.steps)}  "
              f"thinks={n_thinks}  wall_clock={log.wall_clock_ms:.0f}ms")

    return logs


# ---------------------------------------------------------------------------
# Sample tasks
# ---------------------------------------------------------------------------

SAMPLE_TASKS = [
    # S: Single-tool
    {"query_id": "s01", "prompt": "When did the Roman Empire fall and who was the last emperor?",
     "ground_truth": "476", "category": "single"},
    {"query_id": "s02", "prompt": "What is the speed of light in km/s?",
     "ground_truth": "299,792", "category": "single"},
    {"query_id": "s03", "prompt": "What is the current weather in Tokyo?",
     "ground_truth": "Clear", "category": "single"},
    {"query_id": "s04", "prompt": "What is the square root of 2025?",
     "ground_truth": "45", "category": "single"},
    # M: Multi-tool chain
    {"query_id": "m01", "prompt": "Find the speed of light from the document corpus (in km/s), then convert it to miles per hour.",
     "ground_truth": "670616", "category": "multi_chain"},
    {"query_id": "m02", "prompt": "Look up when DNA was first identified and when the double helix was described, then calculate how many years passed between those two events.",
     "ground_truth": "84", "category": "multi_chain"},
    {"query_id": "m03", "prompt": "Find where Python was created (which city/country), then look up the current weather there.",
     "ground_truth": "cloudy", "category": "multi_chain"},
    {"query_id": "m04", "prompt": "Search the corpus for the Ming Dynasty dates of the Great Wall, calculate how many years the Ming Dynasty lasted, then convert that duration to days.",
     "ground_truth": "100842", "category": "multi_chain"},
    {"query_id": "m05", "prompt": "Look up the Pythagorean theorem in the corpus, then write and run Python code to verify it for a 3-4-5 right triangle.",
     "ground_truth": "True", "category": "multi_chain"},
    {"query_id": "m06", "prompt": "Find when Einstein published general relativity from the corpus, then search the web for when it was first experimentally confirmed, and calculate how many years between publication and confirmation.",
     "ground_truth": "4", "category": "multi_chain"},
    # P: Multi-tool parallel
    {"query_id": "p01", "prompt": "I need two things at once: (1) the current weather in London and (2) the GDP of Germany from the countries database.",
     "ground_truth": "4259", "category": "multi_parallel"},
    {"query_id": "p02", "prompt": "Simultaneously look up the Treaty of Versailles in Wikipedia and search the web for its key terms and reparations amount.",
     "ground_truth": "132 billion", "category": "multi_parallel"},
    {"query_id": "p03", "prompt": "Get the weather in both San Francisco and Sydney, and tell me which city is warmer right now.",
     "ground_truth": "Sydney", "category": "multi_parallel"},
    # A: Action tasks
    {"query_id": "a01", "prompt": "Find out when the Roman Empire fell, then send an email to history@university.edu summarizing the key facts (date, last emperor, who deposed him).",
     "ground_truth": "Email sent", "category": "action"},
    {"query_id": "a02", "prompt": "Look up the Human Genome Project on Wikipedia to find when it was completed, then create a calendar event titled 'Genome Project Anniversary' on 2025-04-14 at 10:00 with a short description.",
     "ground_truth": "event", "category": "action"},
    {"query_id": "a03", "prompt": "Query the countries database for all European countries with GDP over 3000 billion USD, then email the results to finance@corp.com with subject 'Top European Economies'.",
     "ground_truth": "Email sent", "category": "action"},
    # C: Complex
    {"query_id": "c01", "prompt": "Search the corpus for the Amazon rainforest oxygen percentage, then use the calculator to figure out how many tons that is if Earth's total oxygen production is 3e11 tons/year, and finally search the web for the current Amazon deforestation rate.",
     "ground_truth": "6e+10", "category": "complex"},
    {"query_id": "c02", "prompt": "Find the number of base pairs in the human genome from the corpus, look up the Human Genome Project on Wikipedia for total cost, then calculate the cost per base pair in dollars.",
     "ground_truth": "0.9", "category": "complex"},
    {"query_id": "c03", "prompt": "Query the database for the GDP and population of all Asian countries, calculate GDP per capita for each, write Python code to find which Asian country has the highest GDP per capita, and report the result.",
     "ground_truth": "Japan", "category": "complex"},
    {"query_id": "c04", "prompt": "Find when the Treaty of Versailles was signed from the corpus, look up its terms on Wikipedia, search the web for additional details on reparations, calculate how many years ago it was signed (from 2025), and then email a summary to diplomacy@un.org.",
     "ground_truth": "106", "category": "complex"},
    # N: No-tool
    {"query_id": "n01", "prompt": "What is 2 + 2?", "ground_truth": "4", "category": "no_tool"},
    {"query_id": "n02", "prompt": "Say hello in French.", "ground_truth": "Bonjour", "category": "no_tool"},
]


async def main() -> None:
    config = PACTConfig(model_name="gpt-4o-mini")
    logs = await run_batch(SAMPLE_TASKS, config=config, save_dir="logs/agent_a")
    print(f"\nDone. {sum(l.task_success for l in logs)}/{len(logs)} succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
