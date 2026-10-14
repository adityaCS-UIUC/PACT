"""
DAG Scheduler: build a dependency graph of tool calls and execute
independent calls in parallel via asyncio.

Dependency detection:
  If tool X's arguments contain a placeholder like "$result_of_Y",
  then X depends on Y and must wait for Y to finish first.
  Tools with no dependencies run concurrently.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from pact.config import ToolCallRecord, Timer
from pact.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\$result_of_(\w+)")


def _find_deps(arguments: dict[str, Any]) -> set[str]:
    """Scan argument values for $result_of_<tool_name> placeholders."""
    deps: set[str] = set()
    for v in arguments.values():
        if isinstance(v, str):
            deps.update(_PLACEHOLDER_RE.findall(v))
    return deps


def _resolve_placeholders(
    arguments: dict[str, Any],
    results: dict[str, str],
) -> dict[str, Any]:
    """Replace $result_of_<tool> placeholders with actual results."""
    resolved = {}
    for k, v in arguments.items():
        if isinstance(v, str):
            for tool_name, result in results.items():
                v = v.replace(f"$result_of_{tool_name}", result)
        resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# DAG representation
# ---------------------------------------------------------------------------

class ToolNode:
    def __init__(self, tool_name: str, arguments: dict[str, Any], call_id: str) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.call_id = call_id
        self.deps: set[str] = _find_deps(arguments)
        self.record: ToolCallRecord | None = None


class DAG:
    """A directed acyclic graph of tool calls."""

    def __init__(self, nodes: list[ToolNode]) -> None:
        self.nodes = {n.call_id: n for n in nodes}
        self._adj: dict[str, set[str]] = defaultdict(set)
        for n in nodes:
            for dep_tool in n.deps:
                dep_ids = [
                    m.call_id for m in nodes
                    if m.tool_name == dep_tool and m.call_id != n.call_id
                ]
                for did in dep_ids:
                    self._adj[n.call_id].add(did)

    def topological_layers(self) -> list[list[str]]:
        """
        Return layers where each layer is a set of call_ids whose
        dependencies are satisfied by all prior layers.
        """
        in_degree: dict[str, int] = {cid: 0 for cid in self.nodes}
        for cid, deps in self._adj.items():
            in_degree[cid] = len(deps)

        layers: list[list[str]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            layer = [cid for cid in remaining if in_degree[cid] == 0]
            if not layer:
                raise RuntimeError("Cycle detected in tool-call DAG")
            layers.append(layer)
            for cid in layer:
                remaining.discard(cid)
                for other in remaining:
                    if cid in self._adj.get(other, set()):
                        in_degree[other] -= 1

        return layers


# ---------------------------------------------------------------------------
# Async parallel executor
# ---------------------------------------------------------------------------

async def execute_dag(
    dag: DAG,
    registry: ToolRegistry,
) -> list[ToolCallRecord]:
    """
    Execute tool calls respecting DAG dependencies.
    Independent tools within a layer run concurrently.
    Returns all ToolCallRecords (including errors).
    """
    results: dict[str, str] = {}
    all_records: list[ToolCallRecord] = []

    for layer in dag.topological_layers():
        async def _run_one(call_id: str) -> ToolCallRecord:
            node = dag.nodes[call_id]
            resolved_args = _resolve_placeholders(node.arguments, results)
            record = registry.execute(node.tool_name, resolved_args)
            node.record = record
            if record.result is not None:
                results[node.tool_name] = record.result
            return record

        records = await asyncio.gather(*[_run_one(cid) for cid in layer])
        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# Convenience: build DAG from a list of raw tool-call dicts
# ---------------------------------------------------------------------------

def build_dag(calls: list[dict[str, Any]]) -> DAG:
    """
    Build a DAG from a list of dicts like:
        [{"tool_name": "search", "arguments": {"query": "..."}}, ...]
    """
    nodes = []
    for i, c in enumerate(calls):
        nodes.append(ToolNode(
            tool_name=c["tool_name"],
            arguments=c.get("arguments", {}),
            call_id=f"{c['tool_name']}_{i}",
        ))
    return DAG(nodes)
