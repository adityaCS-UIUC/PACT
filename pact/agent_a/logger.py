"""
Logging hooks for Agent A.

Implements openai-agents SDK RunHooks to capture every reasoning step,
tool call (with arguments), and observation into a TrajectoryLog.

The SDK's on_tool_start/on_tool_end hooks do not expose parsed arguments.
We work around this by capturing the raw model response in on_llm_end,
which contains the function-call payloads including argument JSON.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from agents import RunHooks, RunContextWrapper, AgentBase, Tool

from pact.config import (
    AgentStep,
    ThinkBlock,
    ToolCallRecord,
    TrajectoryLog,
)


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter (handles <think> blocks)."""
    clean = re.sub(r"</?think>", "", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [s.strip() for s in parts if s.strip()]


def _extract_think_block(text: str) -> ThinkBlock | None:
    """Extract a ThinkBlock from text that may contain <think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        full = match.group(1).strip()
        return ThinkBlock(full_text=full, sentences=_split_sentences(full))
    return None


class TrajectoryHooks(RunHooks[Any]):
    """
    Captures a full TrajectoryLog from an openai-agents SDK run.

    Usage:
        hooks = TrajectoryHooks(query_id="q1", prompt="...", tool_names=[...])
        result = await Runner.run(agent, prompt, hooks=hooks)
        log = hooks.finalize(result.final_output)
    """

    def __init__(
        self,
        query_id: str,
        prompt: str,
        tool_names: list[str],
    ) -> None:
        self.log = TrajectoryLog(
            query_id=query_id,
            prompt=prompt,
            available_tools=tool_names,
            mode="baseline",
        )
        self._step_counter = 0
        self._wall_start = time.perf_counter()
        self._tool_start: float | None = None

        self._pending_tool_args: dict[str, dict[str, Any]] = {}
        self._last_assistant_text: str = ""

    # ----- SDK hook callbacks -----

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: AgentBase,
        response: Any,
    ) -> None:
        """
        Capture function-call arguments from the raw model response.
        The response object contains output items including function_call
        entries with name and arguments fields.
        """
        self._pending_tool_args.clear()
        self._last_assistant_text = ""

        if not hasattr(response, "output"):
            return

        for item in response.output:
            item_type = getattr(item, "type", "")

            if item_type == "function_call":
                name = getattr(item, "name", "")
                args_str = getattr(item, "arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": str(args_str)}
                self._pending_tool_args[name] = args

            elif item_type == "message":
                content = getattr(item, "content", "")
                if isinstance(content, list):
                    for part in content:
                        if hasattr(part, "text"):
                            self._last_assistant_text += part.text
                elif isinstance(content, str):
                    self._last_assistant_text += content

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: AgentBase,
        tool: Tool,
    ) -> None:
        self._tool_start = time.perf_counter()

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: AgentBase,
        tool: Tool,
        result: str,
    ) -> None:
        latency = 0.0
        if self._tool_start is not None:
            latency = (time.perf_counter() - self._tool_start) * 1000

        arguments = self._pending_tool_args.pop(tool.name, {})

        think_block = _extract_think_block(self._last_assistant_text)

        step = AgentStep(
            step_idx=self._step_counter,
            think_block=think_block,
            tool_calls=[
                ToolCallRecord(
                    tool_name=tool.name,
                    arguments=arguments,
                    result=result,
                    latency_ms=latency,
                )
            ],
            observation=result,
        )
        self.log.steps.append(step)
        self._step_counter += 1

    # ----- finalization -----

    def finalize(
        self,
        final_output: str,
        task_success: bool = False,
        total_tokens: int = 0,
    ) -> TrajectoryLog:
        self.log.final_answer = final_output
        self.log.task_success = task_success
        self.log.total_tokens = total_tokens
        self.log.wall_clock_ms = (time.perf_counter() - self._wall_start) * 1000
        return self.log
