"""
Oracle: determines whether a cumulative thought chunk contains enough
information to produce the known-correct tool call.

Two modes:
  1. Keyword oracle (free) — checks if the chunk mentions the tool name
     and all argument values.  Works for the built-in PACT tools.
  2. LLM oracle (costs ~$0.001/call) — asks gpt-4o-mini whether the
     chunk contains enough to produce the call.  Required for ToolBench
     APIs whose names are opaque (e.g. "get_user_by_id_for_rapidapi").
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _param_present(value: Any, text_norm: str) -> bool:
    """Check whether a parameter value is mentioned in the text."""
    if isinstance(value, str):
        val_norm = _normalize(value)
        tokens = val_norm.split()
        if len(tokens) <= 3:
            return val_norm in text_norm
        hits = sum(1 for t in tokens if t in text_norm)
        return hits / len(tokens) >= 0.7

    if isinstance(value, (int, float)):
        return str(value) in text_norm

    if isinstance(value, list):
        return all(_param_present(v, text_norm) for v in value)

    if isinstance(value, dict):
        return all(_param_present(v, text_norm) for v in value.values())

    return str(value).lower() in text_norm


# ---------------------------------------------------------------------------
# Keyword synonyms for built-in PACT tools
# ---------------------------------------------------------------------------

_KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "search": ["search", "look up", "find", "query", "retrieve", "corpus"],
    "get_more_passages": ["more passages", "more results", "refine", "additional"],
    "web_search": ["web", "search online", "internet", "browse", "google"],
    "calculator": ["calculate", "compute", "math", "evaluate", "arithmetic"],
    "code_execute": ["run code", "execute", "python", "write code", "script"],
    "weather_lookup": ["weather", "temperature", "forecast", "climate"],
    "unit_convert": ["convert", "conversion", "unit", "transform"],
    "wiki_lookup": ["wikipedia", "wiki", "encyclopedia", "look up", "biography"],
    "send_email": ["email", "send", "mail", "notify", "message"],
    "create_calendar_event": ["calendar", "schedule", "event", "meeting", "appointment"],
    "database_query": ["database", "query", "table", "sql", "data", "countries"],
}


# ---------------------------------------------------------------------------
# Mode 1: Keyword oracle (free, fast)
# ---------------------------------------------------------------------------

def oracle_check(
    chunk: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> bool:
    """
    Keyword-based oracle.  Return True if `chunk` mentions the correct
    tool (by name or synonym) AND all argument values.
    """
    text_norm = _normalize(chunk)

    tool_name_norm = _normalize(tool_name)
    tool_mentioned = (
        tool_name_norm in text_norm
        or tool_name_norm.replace("_", " ") in text_norm
    )

    if not tool_mentioned:
        synonyms = _KEYWORD_SYNONYMS.get(tool_name, [])
        tool_mentioned = any(syn in text_norm for syn in synonyms)

    if not tool_mentioned:
        return False

    for key, value in tool_arguments.items():
        if not _param_present(value, text_norm):
            return False

    return True


# ---------------------------------------------------------------------------
# Mode 2: LLM oracle (for ToolBench APIs with opaque names)
# ---------------------------------------------------------------------------

_LLM_ORACLE_SYSTEM = """\
You are a labeling oracle.  Given a reasoning chunk and a target tool call, \
decide whether the reasoning contains enough SPECIFIC information \
(tool intent + key argument values) to generate this exact call.
Respond with ONLY one word: YES or NO."""

_LLM_ORACLE_USER = """\
## Reasoning chunk
{chunk}

## Target tool call
Tool: {tool_name}
Arguments: {arguments}

Does the reasoning contain enough specific information to produce this \
exact tool call (correct tool + correct argument values)?"""


def oracle_check_llm(
    chunk: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    model: str = "gpt-4o-mini",
    client: OpenAI | None = None,
) -> bool:
    """
    LLM-based oracle for ToolBench APIs.
    Asks gpt-4o-mini whether the chunk is ready to execute the tool call.
    ~$0.001 per call.
    """
    c = client or OpenAI()
    args_str = json.dumps(tool_arguments, default=str)
    user_msg = _LLM_ORACLE_USER.format(
        chunk=chunk[:1500],
        tool_name=tool_name,
        arguments=args_str[:500],
    )

    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_ORACLE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=3,
        temperature=0.0,
    )

    answer = resp.choices[0].message.content.strip().upper()
    return answer == "YES"


# ---------------------------------------------------------------------------
# Mode 3: Adaptive oracle (keyword first, LLM fallback)
# ---------------------------------------------------------------------------

def oracle_check_adaptive(
    chunk: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    use_llm_for_unknown_tools: bool = True,
    model: str = "gpt-4o-mini",
    client: OpenAI | None = None,
) -> bool:
    """
    Try keyword oracle first.  If the tool name is not in the built-in
    synonym table, fall back to the LLM oracle.
    """
    is_known_tool = tool_name in _KEYWORD_SYNONYMS

    if is_known_tool:
        return oracle_check(chunk, tool_name, tool_arguments)

    if use_llm_for_unknown_tools:
        return oracle_check_llm(chunk, tool_name, tool_arguments,
                                model=model, client=client)

    return oracle_check(chunk, tool_name, tool_arguments)
