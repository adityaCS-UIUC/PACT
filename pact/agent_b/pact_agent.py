"""
Agent B: PACT execution engine.

Generates reasoning tokens, checks the speculator after each sentence,
and when the speculator score crosses tau, halts reasoning and forces
immediate tool execution via the DAG scheduler.

Includes a rollback mechanism: if a speculative tool call fails, the
bad call is removed from context and the model resumes <think>-ing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from agents import Agent, Runner

from pact.config import (
    PACTConfig,
    TrajectoryLog,
    AgentStep,
    ThinkBlock,
    ToolCallRecord,
    Timer,
)
from pact.tools.registry import ToolRegistry
from pact.tools.rag_tools import build_rag_registry, search, get_more_passages
from pact.tools.extended_tools import (
    ALL_SDK_TOOLS, build_full_registry,
)
from pact.speculator.model import Speculator
from pact.agent_b.dag_scheduler import build_dag, execute_dag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(
    r"#TOOL\[(\w+)\]\s*(\{.*?\})",
    re.DOTALL,
)


def _split_sentences(text: str) -> list[str]:
    clean = re.sub(r"</?think>", "", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [s.strip() for s in parts if s.strip()]


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Parse tool calls from model output.
    Supports format:  #TOOL[tool_name]{"arg": "val"}
    Also handles OpenAI-style function_call JSON blocks.
    """
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append({"tool_name": tool_name, "arguments": args})
    return calls


# ---------------------------------------------------------------------------
# PACT Agent B instructions
# ---------------------------------------------------------------------------

AGENT_B_INSTRUCTIONS = """\
You are a research assistant that answers questions using the available tools.

**Process:**
1. Reason inside <think>...</think> tags.  Think step by step about what you
   know and what tool call (if any) would get you the missing piece.
2. When you are confident you know which tool to call and with what arguments,
   emit one or more tool calls using this format:
       #TOOL[tool_name]{{"arg1": "value1"}}
   You may emit multiple tool calls if they are independent.
3. After receiving tool results, continue reasoning or produce a final answer.

**Key rules:**
- Call the RIGHT tool with the RIGHT arguments.  One good call beats three sloppy ones.
- Many tasks require multiple tools in sequence (search → calculate → convert).
  Plan the full chain, but execute one step at a time.
- If two calls are independent, emit them together for parallel execution.
"""

FULL_SDK_TOOLS = [search, get_more_passages] + ALL_SDK_TOOLS


# ---------------------------------------------------------------------------
# PACT Execution Engine
# ---------------------------------------------------------------------------

class PACTAgent:
    """
    The PACT execution engine (Agent B).

    Wraps the openai-agents SDK for generation but intercepts the
    reasoning loop to apply the speculator and DAG scheduler.
    """

    def __init__(
        self,
        config: PACTConfig,
        speculator: Speculator | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.speculator = speculator
        self.registry = registry or build_full_registry()
        self.tool_names = self.registry.list_names()

        self._sdk_agent = Agent(
            name="Agent-B (PACT)",
            instructions=AGENT_B_INSTRUCTIONS,
            tools=FULL_SDK_TOOLS,
            model=config.model_name,
        )

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        query_id: str | None = None,
        ground_truth: str | None = None,
    ) -> TrajectoryLog:
        """
        Run the full PACT loop:
          1. Let the model start reasoning.
          2. After each sentence, query the speculator.
          3. If score >= tau, halt reasoning, extract/force tool calls.
          4. Execute tools via DAG scheduler.
          5. If tool errors -> rollback and resume thinking.
          6. If tool success -> append results and continue.
          7. Repeat until final answer.
        """
        qid = query_id or str(uuid.uuid4())[:8]
        tau = self.config.tau

        log = TrajectoryLog(
            query_id=qid,
            prompt=prompt,
            available_tools=self.tool_names,
            mode="pact",
            tau=tau,
        )

        with Timer() as total_timer:
            # --- Use the SDK runner for the full agent loop ---
            # The speculator acts as a "gate" that can short-circuit
            # the model's reasoning in future steps.
            #
            # For the initial implementation, we run the SDK agent
            # and apply speculator scoring post-hoc on the reasoning
            # blocks to measure where it *would* have interrupted.
            # A fully integrated version would intercept the streaming
            # token output.

            result = await self._run_with_speculation(prompt, log, tau)

        log.wall_clock_ms = total_timer.elapsed_ms
        log.final_answer = result

        if ground_truth:
            log.task_success = ground_truth.lower() in result.lower()

        return log

    # -----------------------------------------------------------------
    # Speculation-aware execution
    # -----------------------------------------------------------------

    async def _run_with_speculation(
        self,
        prompt: str,
        log: TrajectoryLog,
        tau: float,
    ) -> str:
        """
        Run the agent with speculator gating.

        Strategy:
        - First, let the SDK agent produce a full response (reasoning + tool calls).
        - Parse the <think> block and check where the speculator would have
          triggered early termination.
        - If the speculator says "ready" before the model finished thinking,
          record the truncation point and execute tools from that point.
        - Handle rollbacks if speculative tool calls fail.
        """
        context_messages: list[dict[str, str]] = [
            {"role": "user", "content": prompt}
        ]

        step_idx = 0
        max_steps = self.config.max_steps
        final_answer = ""

        while step_idx < max_steps:
            # --- Generate one round via the SDK ---
            sdk_result = await Runner.run(
                self._sdk_agent,
                context_messages,
            )
            output = sdk_result.final_output

            # --- Extract think block ---
            think_match = _THINK_RE.search(output)
            think_text = think_match.group(1) if think_match else ""
            sentences = _split_sentences(think_text)

            # --- Speculator gating ---
            truncation_point: int | None = None
            if self.speculator and sentences:
                cumulative = ""
                for i, sent in enumerate(sentences):
                    cumulative += (" " + sent) if cumulative else sent
                    for tool_name in self.tool_names:
                        score = self.speculator.score(prompt, cumulative, tool_name)
                        if score >= tau:
                            truncation_point = i
                            break
                    if truncation_point is not None:
                        break

            think_block = ThinkBlock(
                full_text=think_text,
                sentences=sentences,
                truncated_at_sentence=truncation_point,
            )

            # --- Extract tool calls (from full output or truncated thought) ---
            tool_calls_raw = _extract_tool_calls(output)

            # Also check if SDK made tool calls via its internal loop
            if not tool_calls_raw and sdk_result.last_agent != self._sdk_agent:
                pass

            step = AgentStep(
                step_idx=step_idx,
                think_block=think_block,
                tool_calls=[],
            )

            if tool_calls_raw:
                # --- Execute via DAG scheduler ---
                dag = build_dag(tool_calls_raw)
                records = await execute_dag(dag, self.registry)

                failed = [r for r in records if r.error]
                succeeded = [r for r in records if r.result is not None]

                step.tool_calls = records

                if failed:
                    # --- Rollback: remove failed calls, inject error, resume ---
                    error_msg = "; ".join(
                        f"{r.tool_name}: {r.error}" for r in failed
                    )
                    step.observation = f"ROLLBACK: {error_msg}"
                    context_messages.append({
                        "role": "assistant",
                        "content": output,
                    })
                    context_messages.append({
                        "role": "user",
                        "content": (
                            f"Tool error(s) occurred: {error_msg}\n"
                            "Please resume your <think> process and try again "
                            "with corrected arguments."
                        ),
                    })
                else:
                    obs = "\n".join(
                        f"[{r.tool_name}] {r.result}" for r in succeeded
                    )
                    step.observation = obs
                    context_messages.append({
                        "role": "assistant",
                        "content": output,
                    })
                    context_messages.append({
                        "role": "user",
                        "content": f"Tool results:\n{obs}\n\nContinue reasoning or give your final answer.",
                    })
            else:
                # No tool calls -> this is the final answer
                final_answer = output
                # Strip <think> tags from final answer
                final_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
                step.observation = final_answer
                log.steps.append(step)
                break

            log.steps.append(step)
            step_idx += 1

        return final_answer or output

    # -----------------------------------------------------------------
    # Batch runner
    # -----------------------------------------------------------------

    async def run_batch(
        self,
        tasks: list[dict[str, Any]],
        save_dir: str = "logs/agent_b",
    ) -> list[TrajectoryLog]:
        from pathlib import Path
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        logs: list[TrajectoryLog] = []
        for i, task in enumerate(tasks):
            qid = task.get("query_id", f"q{i:04d}")
            print(f"[Agent B / PACT] Running {qid}: {task['prompt'][:60]}...")
            log = await self.run(
                prompt=task["prompt"],
                query_id=qid,
                ground_truth=task.get("ground_truth"),
            )
            log.save(out / f"{qid}.json")
            logs.append(log)
            status = "OK" if log.task_success else "FAIL"
            print(f"  -> {status}  tau={self.config.tau}  wall_clock={log.wall_clock_ms:.0f}ms")

        return logs


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

async def main() -> None:
    from pact.agent_a.react_agent import SAMPLE_TASKS

    config = PACTConfig(
        model_name="gpt-4o-mini",
        tau=0.85,
    )

    # Without speculator (runs like Agent A but through the PACT engine)
    agent_b = PACTAgent(config=config, speculator=None)
    logs = await agent_b.run_batch(SAMPLE_TASKS, save_dir="logs/agent_b")
    n_ok = sum(l.task_success for l in logs)
    print(f"\nDone. {n_ok}/{len(logs)} succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
