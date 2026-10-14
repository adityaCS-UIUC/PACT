"""
Tool registry: stores callable tools by name for both Agent A and Agent B.

Each tool is a plain Python callable wrapped with metadata (name, description,
parameter schema).  The registry also exposes an openai-agents-SDK-compatible
list so the same tools work with Runner.run() and with our custom PACT engine.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pact.config import ToolCallRecord, Timer


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-schema style
    callable: Callable[..., str]
    depends_on: list[str] = field(default_factory=list)


class ToolRegistry:
    """Central store for all tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ----- registration -----

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., str],
        depends_on: list[str] | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            callable=fn,
            depends_on=depends_on or [],
        )

    # ----- lookup -----

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def to_openai_tool_schemas(self) -> list[dict]:
        """Return tool descriptions in OpenAI function-calling schema format."""
        schemas = []
        for spec in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return schemas

    # ----- execution -----

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        """Execute a tool by name, returning a ToolCallRecord with timing."""
        spec = self._tools[name]
        record = ToolCallRecord(tool_name=name, arguments=arguments)
        with Timer() as t:
            try:
                sig = inspect.signature(spec.callable)
                params = sig.parameters
                # If the function accepts **kwargs, pass everything through.
                # Otherwise filter to only declared parameter names.
                has_var_keyword = any(
                    p.kind == p.VAR_KEYWORD for p in params.values()
                )
                if has_var_keyword:
                    bound = {k: v for k, v in arguments.items()
                             if not k.startswith("_")}
                else:
                    bound = {k: v for k, v in arguments.items()
                             if k in params}
                record.result = spec.callable(**bound)
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
        record.latency_ms = t.elapsed_ms
        return record
