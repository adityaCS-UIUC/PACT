"""
Dynamic tool registration from ToolBench API specs.

Each ToolBench task comes with its own set of APIs.  This module:
  1. Builds callable tools on-the-fly from API specs.
  2. Backs each tool with a GPT-based API simulator (following StableToolBench's
     approach) that generates realistic responses from API documentation.
  3. Caches every simulated response for reproducibility.
  4. Exposes a per-task ToolRegistry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import openai

from pact.tools.registry import ToolRegistry


def _sanitize_name(raw: str) -> str:
    """OpenAI requires tool names to match ^[a-zA-Z0-9_-]+$."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tool"


# ---------------------------------------------------------------------------
# API Simulator — generates realistic responses using GPT-4o-mini
# ---------------------------------------------------------------------------

_SIMULATOR_SYSTEM_PROMPT = """\
You are an API Server operating within a specialized tool, which contains a \
collection of distinct APIs. Your role is to deeply understand the function of \
each API based on their descriptions in the API documentation. As you receive \
specific inputs for individual API calls within this tool, analyze these inputs \
to determine their intended purpose. Your task is to craft a JSON formatted \
response that aligns with the expected output of the API. The JSON schema is:
{
    "error": "",
    "response": ""
}

The error field should remain empty, indicating no errors in processing. The \
response field should contain the content you formulate based on the API's \
functionality and the input provided. Ensure that your responses are meaningful, \
directly addressing the API's intended functionality.

RULES:
- Your response MUST contain rich, realistic information given the API input parameters.
- Your response must be effective and have practical content — not placeholder text.
- Return ONLY the JSON object, nothing else.
- If the API would return a list of items, include 3-5 realistic example items.
- If the API would return details about a specific entity, include realistic details.
"""


def _build_simulator_prompt(spec: dict[str, Any], arguments: dict[str, Any]) -> str:
    """Build the user prompt for the API simulator, following StableToolBench format."""
    api_doc = {
        "api_name": spec.get("api_name", "unknown"),
        "api_description": spec.get("api_description", ""),
        "required_parameters": spec.get("required_parameters", []),
        "optional_parameters": spec.get("optional_parameters", []),
        "tool_description": spec.get("tool_description", ""),
        "tool_name": spec.get("tool_name", ""),
        "category_name": spec.get("category_name", ""),
    }

    template = spec.get("template_response", {})
    if template:
        api_doc["template_response"] = template

    request_data = {
        "category": spec.get("category_name", ""),
        "tool_name": spec.get("tool_name", ""),
        "api_name": spec.get("api_name", ""),
        "tool_input": json.dumps(arguments, default=str),
    }

    return f"API doc:\n{json.dumps(api_doc, indent=2, default=str)}\n\nRequest:\n{json.dumps(request_data, indent=2, default=str)}"


class APISimulator:
    """
    GPT-based API simulator with persistent caching.

    On a tool call:
      1. Check the local cache for a previous response.
      2. On cache miss, call GPT-4o-mini with the API documentation + arguments
         to generate a realistic response.
      3. Cache the result for reproducibility.

    This follows StableToolBench's "GPT-based caching system" approach.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._cache: dict[str, str] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/api_cache")
        self._model = model
        self._client: openai.OpenAI | None = None
        self._specs: dict[str, dict[str, Any]] = {}
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

        if self._cache_dir.exists():
            self._load_cache()

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def _load_cache(self) -> None:
        for p in self._cache_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                key = data.get("key", p.stem)
                self._cache[key] = data.get("response", "")
            except (json.JSONDecodeError, KeyError):
                continue

    def register_spec(self, api_name: str, spec: dict[str, Any]) -> None:
        self._specs[api_name] = spec

    @staticmethod
    def _make_key(api_name: str, arguments: dict[str, Any]) -> str:
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        h = hashlib.md5(f"{api_name}:{args_str}".encode()).hexdigest()[:12]
        return f"{api_name}__{h}"

    def get_cached(self, api_name: str, arguments: dict[str, Any]) -> str | None:
        key = self._make_key(api_name, arguments)
        return self._cache.get(key)

    def _save_to_cache(
        self, api_name: str, arguments: dict[str, Any], response: str
    ) -> None:
        key = self._make_key(api_name, arguments)
        self._cache[key] = response
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / f"{key}.json").write_text(
            json.dumps({
                "key": key,
                "api_name": api_name,
                "arguments": arguments,
                "response": response,
            }, default=str)
        )

    def _simulate(self, api_name: str, arguments: dict[str, Any]) -> str:
        spec = self._specs.get(api_name, {})
        if not spec:
            spec = {"api_name": api_name, "api_description": f"API named {api_name}"}

        user_prompt = _build_simulator_prompt(spec, arguments)

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SIMULATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or '{"error": "", "response": ""}'
        except Exception as e:
            self._stats["errors"] += 1
            return json.dumps({
                "error": f"Simulator error: {type(e).__name__}: {e}",
                "response": "",
            })

    def execute(self, api_name: str, arguments: dict[str, Any]) -> str:
        cached = self.get_cached(api_name, arguments)
        if cached is not None:
            self._stats["hits"] += 1
            return cached

        self._stats["misses"] += 1
        response = self._simulate(api_name, arguments)
        self._save_to_cache(api_name, arguments, response)
        return response

    def print_stats(self) -> None:
        total = self._stats["hits"] + self._stats["misses"]
        print(f"  API Simulator: {total} calls "
              f"({self._stats['hits']} cache hits, "
              f"{self._stats['misses']} simulated, "
              f"{self._stats['errors']} errors)")


# Singleton simulator instance
_global_simulator = APISimulator()


def set_cache_dir(path: str | Path) -> None:
    """Point the global simulator cache at a directory."""
    global _global_simulator
    _global_simulator = APISimulator(cache_dir=path)


def get_simulator() -> APISimulator:
    """Access the global simulator (e.g. to print stats)."""
    return _global_simulator


# ---------------------------------------------------------------------------
# Build a callable function from an API spec
# ---------------------------------------------------------------------------

def _make_tool_fn(
    api_name: str,
    spec: dict[str, Any],
    simulator: APISimulator,
) -> Callable[..., str]:
    """Create a closure that calls the simulator-backed API."""
    simulator.register_spec(api_name, spec)

    def tool_fn(**kwargs: Any) -> str:
        return simulator.execute(api_name, kwargs)

    tool_fn.__name__ = api_name
    tool_fn.__doc__ = spec.get("api_description", f"Call the {api_name} API.")[:200]
    return tool_fn


def _spec_to_json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a ToolBench API spec into a JSON Schema for function parameters."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in spec.get("required_parameters", []):
        name = param.get("name", "")
        if not name:
            continue
        properties[name] = {
            "type": _map_type(param.get("type", "string")),
            "description": param.get("description", ""),
        }
        required.append(name)

    for param in spec.get("optional_parameters", []):
        name = param.get("name", "")
        if not name:
            continue
        prop: dict[str, Any] = {
            "type": _map_type(param.get("type", "string")),
            "description": param.get("description", ""),
        }
        if "default" in param and param["default"] is not None:
            prop["default"] = param["default"]
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _map_type(t: str) -> str:
    t_lower = t.lower().strip()
    mapping = {
        "string": "string",
        "str": "string",
        "number": "number",
        "float": "number",
        "integer": "integer",
        "int": "integer",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "object": "object",
        "dict": "object",
    }
    return mapping.get(t_lower, "string")


# ---------------------------------------------------------------------------
# Public: build per-task tools
# ---------------------------------------------------------------------------

def build_toolbench_registry(
    api_specs: list[dict[str, Any]],
    simulator: APISimulator | None = None,
) -> ToolRegistry:
    """
    Build a ToolRegistry from a list of ToolBench API specs.
    Each spec should have at minimum {api_name, api_description}.
    """
    sim = simulator or _global_simulator
    reg = ToolRegistry()

    seen_names: set[str] = set()
    for spec in api_specs:
        raw_name = spec.get("api_name", spec.get("tool_name", "unknown"))
        api_name = _sanitize_name(raw_name)
        if api_name in seen_names:
            api_name = f"{api_name}_{len(seen_names)}"
        seen_names.add(api_name)

        description = spec.get("api_description", spec.get("tool_description", ""))
        schema = _spec_to_json_schema(spec)
        fn = _make_tool_fn(raw_name, spec, sim)

        reg.register(
            name=api_name,
            description=description[:300],
            parameters=schema,
            fn=fn,
        )

    return reg
