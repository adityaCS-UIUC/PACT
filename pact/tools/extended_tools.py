"""
Extended tool suite for multi-tool experiments.

Provides real-world-style tools beyond RAG: web search, calculator,
code execution, weather, unit conversion, and action tools (email,
calendar).  Each has both a plain-function version (for ToolRegistry /
Agent B) and an @function_tool version (for openai-agents SDK / Agent A).

Mock implementations are used for reproducibility; swap in real APIs
(SerpAPI, OpenWeatherMap, etc.) when ready.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import random
import re
import textwrap
from datetime import datetime, timedelta
from typing import Any

from agents import function_tool

from pact.tools.registry import ToolRegistry

# ===================================================================
# 1. WEB SEARCH (simulated — swap for SerpAPI / Tavily / Brave)
# ===================================================================

_WEB_INDEX: dict[str, list[str]] = {
    "roman empire fall": [
        "The fall of the Western Roman Empire in 476 CE marked the end of ancient Rome. Odoacer deposed Romulus Augustulus.",
        "Historians debate whether 476 CE or the fall of Constantinople in 1453 is the true 'end' of the Roman Empire.",
    ],
    "treaty of versailles terms": [
        "Key terms: Germany accepted war guilt, paid reparations of 132 billion gold marks, lost 13% of territory and 10% of population.",
        "The Treaty limited the German army to 100,000 troops and banned tanks, an air force, and submarines.",
    ],
    "quantum computing applications 2025": [
        "IBM and Google demonstrated quantum advantage in optimization and drug discovery simulations.",
        "Quantum computing is being applied to cryptography, materials science, and financial modeling.",
    ],
    "einstein general relativity experimental confirmation": [
        "The 1919 Eddington solar eclipse expedition confirmed light bending predicted by general relativity.",
        "Gravitational waves, predicted by Einstein in 1916, were first detected by LIGO in 2015.",
    ],
    "amazon deforestation rate": [
        "Amazon deforestation reached 13,235 km² in 2021. Brazil pledged to end illegal deforestation by 2030.",
        "Between 2001 and 2020, the Amazon lost approximately 8% of its forest area.",
    ],
    "python programming language history": [
        "Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands, released 1991.",
        "Python 3.0 was released in 2008; it is now the most popular language by many indices.",
    ],
    "great wall of china length": [
        "The Great Wall of China stretches approximately 21,196 km (13,171 miles) including all branches.",
        "The Ming Dynasty sections total about 8,850 km (5,500 miles).",
    ],
    "human genome project": [
        "The Human Genome Project completed in 2003, mapping roughly 3 billion DNA base pairs and ~20,500 genes.",
        "The project cost approximately $2.7 billion and took 13 years.",
    ],
    "speed of light applications": [
        "The speed of light (299,792 km/s) defines the meter and is fundamental to GPS satellite corrections.",
        "Light-year = 9.461 trillion km, used to measure astronomical distances.",
    ],
    "machine learning transformer architecture": [
        "The Transformer architecture was introduced in 'Attention Is All You Need' (Vaswani et al., 2017).",
        "Transformers power GPT, BERT, and most modern LLMs; self-attention scales quadratically with sequence length.",
    ],
}


def web_search_fn(query: str, num_results: int = 3) -> str:
    """Search the web for current information on any topic."""
    q_lower = query.lower()
    best_key = ""
    best_score = 0
    for key in _WEB_INDEX:
        tokens = key.split()
        score = sum(1 for t in tokens if t in q_lower)
        if score > best_score:
            best_score = score
            best_key = key
    if best_score == 0:
        return f"Web search for '{query}': No relevant results found."
    hits = _WEB_INDEX[best_key][:num_results]
    return "\n\n".join(f"[Web {i+1}] {h}" for i, h in enumerate(hits))


@function_tool
def web_search(query: str, num_results: int = 3) -> str:
    """Search the web for current, real-time information on any topic.
    Use when the answer requires up-to-date facts, statistics, news, or
    information beyond what the local document corpus contains.

    Args:
        query: Descriptive search query.
        num_results: Number of results to return (1-5).
    """
    return web_search_fn(query, num_results)


# ===================================================================
# 2. CALCULATOR (safe math eval)
# ===================================================================

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCS:
        val = _SAFE_FUNCS[node.id]
        if isinstance(val, (int, float)):
            return float(val)
        raise ValueError(f"'{node.id}' is a function, not a constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            fn = _SAFE_FUNCS[node.func.id]
            if callable(fn):
                args = [_safe_eval(a) for a in node.args]
                return float(fn(*args))
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator_fn(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        if result == int(result):
            return str(int(result))
        return f"{result:.6g}"
    except Exception as e:
        return f"Calculator error: {e}"


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e.
    Use when you need to compute numeric results from known values.

    Args:
        expression: Math expression to evaluate, e.g. '299792 * 3600' or 'sqrt(9 + 16)'.
    """
    return calculator_fn(expression)


# ===================================================================
# 3. CODE EXECUTION (restricted Python sandbox)
# ===================================================================

def code_execute_fn(code: str) -> str:
    """Execute a Python code snippet and capture its printed output."""
    allowed_builtins = {
        "print": print, "range": range, "len": len, "int": int,
        "float": float, "str": str, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "bool": bool, "abs": abs,
        "max": max, "min": min, "sum": sum, "sorted": sorted,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "round": round, "isinstance": isinstance, "type": type,
        "True": True, "False": False, "None": None,
    }
    import io, contextlib
    buf = io.StringIO()
    local_ns: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": allowed_builtins, "math": math}, local_ns)
        output = buf.getvalue().strip()
        return output if output else "(code executed successfully, no output)"
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {e}"


@function_tool
def code_execute(code: str) -> str:
    """Execute a Python code snippet in a sandboxed environment.
    Use when you need to run computation, verify a formula, process data,
    or test an algorithm.  The code's printed output is returned.

    Args:
        code: Python code to execute. Use print() to produce output.
    """
    return code_execute_fn(code)


# ===================================================================
# 4. WEATHER LOOKUP (simulated)
# ===================================================================

_WEATHER_DB: dict[str, dict[str, Any]] = {
    "amsterdam": {"temp_c": 12, "condition": "Partly cloudy", "humidity": 78, "wind_kph": 22},
    "new york": {"temp_c": 18, "condition": "Sunny", "humidity": 55, "wind_kph": 14},
    "san francisco": {"temp_c": 16, "condition": "Foggy", "humidity": 82, "wind_kph": 18},
    "london": {"temp_c": 10, "condition": "Rainy", "humidity": 88, "wind_kph": 25},
    "tokyo": {"temp_c": 22, "condition": "Clear", "humidity": 60, "wind_kph": 10},
    "beijing": {"temp_c": 20, "condition": "Hazy", "humidity": 50, "wind_kph": 8},
    "paris": {"temp_c": 14, "condition": "Overcast", "humidity": 70, "wind_kph": 12},
    "sydney": {"temp_c": 25, "condition": "Sunny", "humidity": 45, "wind_kph": 16},
    "cairo": {"temp_c": 35, "condition": "Sunny", "humidity": 20, "wind_kph": 6},
    "moscow": {"temp_c": 5, "condition": "Snowy", "humidity": 90, "wind_kph": 30},
    "netherlands": {"temp_c": 12, "condition": "Partly cloudy", "humidity": 78, "wind_kph": 22},
}


def weather_lookup_fn(city: str) -> str:
    """Get current weather for a city."""
    key = city.lower().strip()
    data = _WEATHER_DB.get(key)
    if not data:
        for k, v in _WEATHER_DB.items():
            if k in key or key in k:
                data = v
                break
    if not data:
        return f"Weather data not available for '{city}'."
    return (
        f"Weather in {city.title()}: {data['condition']}, "
        f"{data['temp_c']}°C, humidity {data['humidity']}%, "
        f"wind {data['wind_kph']} km/h"
    )


@function_tool
def weather_lookup(city: str) -> str:
    """Get the current weather for a given city.
    Use when the user asks about weather or you need weather data for planning.

    Args:
        city: City name (e.g. 'London', 'New York').
    """
    return weather_lookup_fn(city)


# ===================================================================
# 5. UNIT CONVERTER
# ===================================================================

_CONVERSIONS: dict[tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("km/s", "mph"): 2236.936,
    ("mph", "km/s"): 1 / 2236.936,
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,
    ("kg", "lbs"): 2.20462,
    ("lbs", "kg"): 0.453592,
    ("celsius", "fahrenheit"): None,  # special
    ("fahrenheit", "celsius"): None,
    ("km", "m"): 1000,
    ("m", "km"): 0.001,
    ("years", "days"): 365.25,
    ("days", "years"): 1 / 365.25,
    ("hours", "seconds"): 3600,
    ("seconds", "hours"): 1 / 3600,
}


def unit_convert_fn(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between units."""
    f, t = from_unit.lower().strip(), to_unit.lower().strip()
    if (f, t) == ("celsius", "fahrenheit"):
        return f"{value * 9/5 + 32:.2f} {to_unit}"
    if (f, t) == ("fahrenheit", "celsius"):
        return f"{(value - 32) * 5/9:.2f} {to_unit}"
    factor = _CONVERSIONS.get((f, t))
    if factor is None:
        return f"Conversion from '{from_unit}' to '{to_unit}' not supported."
    result = value * factor
    if result == int(result):
        return f"{int(result)} {to_unit}"
    return f"{result:.4g} {to_unit}"


@function_tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value from one unit to another.
    Supports: km/miles, m/ft, kg/lbs, celsius/fahrenheit, km\u2044s/mph, years/days, hours/seconds.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit (e.g. 'km/s', 'celsius').
        to_unit: Target unit (e.g. 'mph', 'fahrenheit').
    """
    return unit_convert_fn(value, from_unit, to_unit)


# ===================================================================
# 6. WIKIPEDIA LOOKUP (simulated)
# ===================================================================

_WIKI_ARTICLES: dict[str, str] = {
    "guido van rossum": (
        "Guido van Rossum (born 31 January 1956) is a Dutch programmer, "
        "creator of Python. He worked at CWI in Amsterdam, then at Google "
        "and Dropbox. Python was first released in 1991."
    ),
    "romulus augustulus": (
        "Romulus Augustulus (c. 461–507 CE) was the last Western Roman Emperor, "
        "ruling from 31 October 475 until 4 September 476 when he was deposed "
        "by Odoacer. He was about 16 years old."
    ),
    "ligo gravitational waves": (
        "LIGO first detected gravitational waves on 14 September 2015, "
        "from a binary black hole merger 1.3 billion light-years away. "
        "This confirmed a major prediction of Einstein's general relativity."
    ),
    "human genome project": (
        "The Human Genome Project (1990–2003) was an international effort "
        "that mapped all ~20,500 human genes across 3 billion base pairs. "
        "It cost $2.7 billion and enabled modern genomics and personalized medicine."
    ),
    "treaty of versailles": (
        "The Treaty of Versailles (28 June 1919) ended WWI. Germany accepted "
        "war guilt (Article 231), paid 132 billion gold marks in reparations, "
        "ceded Alsace-Lorraine to France, and was limited to a 100,000-man army."
    ),
}


def wiki_lookup_fn(topic: str) -> str:
    """Look up a topic in the encyclopedia."""
    t_lower = topic.lower().strip()
    for key, article in _WIKI_ARTICLES.items():
        if key in t_lower or t_lower in key:
            return f"[Wikipedia: {topic}] {article}"
    tokens = t_lower.split()
    for key, article in _WIKI_ARTICLES.items():
        if any(tok in key for tok in tokens):
            return f"[Wikipedia: {topic}] {article}"
    return f"No Wikipedia article found for '{topic}'."


@function_tool
def wiki_lookup(topic: str) -> str:
    """Look up a topic in Wikipedia for detailed background information.
    Use when you need biographical, historical, or encyclopedic detail
    beyond what the document corpus or web search provides.

    Args:
        topic: The topic or entity to look up.
    """
    return wiki_lookup_fn(topic)


# ===================================================================
# 7. ACTION TOOLS (simulated side effects)
# ===================================================================

_ACTION_LOG: list[dict[str, Any]] = []


def send_email_fn(to: str, subject: str, body: str) -> str:
    """Send an email."""
    record = {"action": "send_email", "to": to, "subject": subject, "body": body,
              "timestamp": datetime.now().isoformat()}
    _ACTION_LOG.append(record)
    return f"Email sent to {to} with subject '{subject}'."


@function_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    Use when the task requires notifying someone or sharing results via email.

    Args:
        to: Email address of the recipient.
        subject: Email subject line.
        body: Email body text.
    """
    return send_email_fn(to, subject, body)


def create_event_fn(title: str, date: str, time: str, description: str = "") -> str:
    """Create a calendar event."""
    record = {"action": "create_event", "title": title, "date": date,
              "time": time, "description": description,
              "timestamp": datetime.now().isoformat()}
    _ACTION_LOG.append(record)
    return f"Calendar event '{title}' created for {date} at {time}."


@function_tool
def create_calendar_event(title: str, date: str, time: str, description: str = "") -> str:
    """Create a calendar event.
    Use when the task requires scheduling a meeting, reminder, or event.

    Args:
        title: Event title.
        date: Event date (e.g. '2025-04-15').
        time: Event time (e.g. '14:00').
        description: Optional event description.
    """
    return create_event_fn(title, date, time, description)


# ===================================================================
# 8. DATABASE QUERY (simulated structured data)
# ===================================================================

_COUNTRY_DB = [
    {"country": "Germany", "gdp_billion_usd": 4259, "population_million": 83, "continent": "Europe"},
    {"country": "France", "gdp_billion_usd": 2937, "population_million": 67, "continent": "Europe"},
    {"country": "United Kingdom", "gdp_billion_usd": 3070, "population_million": 67, "continent": "Europe"},
    {"country": "United States", "gdp_billion_usd": 25463, "population_million": 331, "continent": "North America"},
    {"country": "China", "gdp_billion_usd": 17963, "population_million": 1412, "continent": "Asia"},
    {"country": "Japan", "gdp_billion_usd": 4231, "population_million": 125, "continent": "Asia"},
    {"country": "India", "gdp_billion_usd": 3385, "population_million": 1408, "continent": "Asia"},
    {"country": "Brazil", "gdp_billion_usd": 1920, "population_million": 214, "continent": "South America"},
    {"country": "Australia", "gdp_billion_usd": 1675, "population_million": 26, "continent": "Oceania"},
    {"country": "Italy", "gdp_billion_usd": 2186, "population_million": 59, "continent": "Europe"},
]


def database_query_fn(table: str, filters: str = "", columns: str = "") -> str:
    """Query a structured database table."""
    if table.lower() != "countries":
        return f"Table '{table}' not found. Available tables: countries."

    results = list(_COUNTRY_DB)

    if filters:
        for filt in filters.split(","):
            filt = filt.strip()
            for op_str, op_fn in [(">=", operator.ge), ("<=", operator.le),
                                   (">", operator.gt), ("<", operator.lt),
                                   ("=", operator.eq)]:
                if op_str in filt:
                    col, val = filt.split(op_str, 1)
                    col, val = col.strip(), val.strip()
                    try:
                        val_num = float(val)
                        results = [r for r in results if op_fn(r.get(col, 0), val_num)]
                    except ValueError:
                        val_clean = val.strip("'\"")
                        results = [r for r in results if op_fn(str(r.get(col, "")), val_clean)]
                    break

    cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

    if not results:
        return "Query returned 0 rows."

    rows = []
    for r in results:
        if cols:
            row = {c: r[c] for c in cols if c in r}
        else:
            row = r
        rows.append(row)

    return json.dumps(rows, indent=2)


@function_tool
def database_query(table: str, filters: str = "", columns: str = "") -> str:
    """Query a structured database table with optional filters.
    Available tables: 'countries' (columns: country, gdp_billion_usd, population_million, continent).

    Args:
        table: Table name (e.g. 'countries').
        filters: Comma-separated filters (e.g. "continent=Europe, gdp_billion_usd>3000").
        columns: Comma-separated column names to return (empty = all).
    """
    return database_query_fn(table, filters, columns)


# ===================================================================
# Registry builder with ALL tools
# ===================================================================

_PARAM_SCHEMAS: dict[str, dict] = {
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "num_results": {"type": "integer", "description": "1-5 results.", "default": 3},
        },
        "required": ["query"],
    },
    "calculator": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate."},
        },
        "required": ["expression"],
    },
    "code_execute": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to run."},
        },
        "required": ["code"],
    },
    "weather_lookup": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
        },
        "required": ["city"],
    },
    "unit_convert": {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Numeric value."},
            "from_unit": {"type": "string", "description": "Source unit."},
            "to_unit": {"type": "string", "description": "Target unit."},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    "wiki_lookup": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Topic to look up."},
        },
        "required": ["topic"],
    },
    "send_email": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email."},
            "subject": {"type": "string", "description": "Subject line."},
            "body": {"type": "string", "description": "Email body."},
        },
        "required": ["to", "subject", "body"],
    },
    "create_calendar_event": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title."},
            "date": {"type": "string", "description": "Date (YYYY-MM-DD)."},
            "time": {"type": "string", "description": "Time (HH:MM)."},
            "description": {"type": "string", "description": "Optional description.", "default": ""},
        },
        "required": ["title", "date", "time"],
    },
    "database_query": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "description": "Table name."},
            "filters": {"type": "string", "description": "Comma-separated filters.", "default": ""},
            "columns": {"type": "string", "description": "Comma-separated columns.", "default": ""},
        },
        "required": ["table"],
    },
}

_TOOL_FNS: dict[str, tuple[str, Any]] = {
    "web_search": ("Search the web for current information.", web_search_fn),
    "calculator": ("Evaluate a math expression.", calculator_fn),
    "code_execute": ("Execute Python code in a sandbox.", code_execute_fn),
    "weather_lookup": ("Get current weather for a city.", weather_lookup_fn),
    "unit_convert": ("Convert between measurement units.", unit_convert_fn),
    "wiki_lookup": ("Look up a topic in Wikipedia.", wiki_lookup_fn),
    "send_email": ("Send an email to a recipient.", send_email_fn),
    "create_calendar_event": ("Create a calendar event.", create_event_fn),
    "database_query": ("Query a structured database.", database_query_fn),
}


def build_extended_registry() -> ToolRegistry:
    """Return a ToolRegistry with ALL extended tools (no RAG)."""
    reg = ToolRegistry()
    for name, (desc, fn) in _TOOL_FNS.items():
        reg.register(name=name, description=desc, parameters=_PARAM_SCHEMAS[name], fn=fn)
    return reg


def build_full_registry() -> ToolRegistry:
    """Return a ToolRegistry with RAG + all extended tools."""
    from pact.tools.rag_tools import build_rag_registry
    reg = build_rag_registry()
    for name, (desc, fn) in _TOOL_FNS.items():
        reg.register(name=name, description=desc, parameters=_PARAM_SCHEMAS[name], fn=fn)
    return reg


ALL_SDK_TOOLS = [
    web_search,
    calculator,
    code_execute,
    weather_lookup,
    unit_convert,
    wiki_lookup,
    send_email,
    create_calendar_event,
    database_query,
]
