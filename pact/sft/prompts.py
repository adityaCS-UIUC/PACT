"""
Canonical prompt formats used by both SFT training and evaluation.

Keep this file as the single source of truth so the model is trained and
evaluated on byte-identical prompt structure. If the SFT scripts ever change
how they format inputs, update HERE and both training and evaluation pick it
up automatically.

Three prompts:

  1. SPECULATOR (READY / NOT_READY classifier)
     Input  = task prompt + cumulative thought
     Output = "READY" or "NOT_READY"

  2. INSTRUCTOR (tool-call generator)
     Input  = task prompt + cumulative thought
     Output = "\\n\\n### Tool Call\\n#TOOL[tool_name]{json_args}"

  3. REASONER (used by Agent B — generates the thought given a task)
     Input  = task prompt
     Output = free-form reasoning text
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Output prefix the SFT-trained instructor expects to emit.
# Used for both training (it gets generated) and inference (we strip it).
# ---------------------------------------------------------------------------

INSTRUCTOR_OUTPUT_PREFIX = "\n\n### Tool Call\n"


# ---------------------------------------------------------------------------
# 1) SPECULATOR — R/NR classifier prompt
# ---------------------------------------------------------------------------

def build_speculator_input(task_prompt: str, thought: str) -> str:
    """Same format as pact/sft/toolcall_sft_with_prompt.py and toolcall_sft.py."""
    return (
        "### Task\n"
        f"{task_prompt.strip()}\n\n"
        "### Thought\n"
        f"{thought.strip()}\n\n"
        "Is the agent ready to call a tool? Answer READY or NOT_READY."
    )


SPECULATOR_OUTPUT_READY = "READY"
SPECULATOR_OUTPUT_NOT_READY = "NOT_READY"


# ---------------------------------------------------------------------------
# 2) INSTRUCTOR — tool-call generator prompt
# ---------------------------------------------------------------------------

def build_instructor_input(task_prompt: str, thought: str) -> str:
    """Same format as pact/sft/toolcall_pred_sft.py."""
    return (
        "### Task\n"
        f"{task_prompt.strip()}\n\n"
        "### Thought\n"
        f"{thought.strip()}\n\n"
        "The agent is READY. Output the tool call."
    )


def format_tool_call_target(tool_name: str, arguments: dict[str, Any]) -> str:
    """SFT target string (without the OUTPUT_PREFIX, which is added at training time)."""
    args_json = json.dumps(arguments or {}, ensure_ascii=False, separators=(",", ":"))
    return f"#TOOL[{tool_name}]{args_json}"


# ---------------------------------------------------------------------------
# 3) REASONER — free-form thought generator (no SFT counterpart; used as the
#    upstream stage that produces a thought for the speculator/instructor).
# ---------------------------------------------------------------------------

REASONER_SYSTEM_PROMPT = (
    "You are a tool-using assistant. Given the user's task and a list of "
    "available tools, reason step by step about which tool from the list to "
    "call and with what arguments. Produce ONLY your reasoning text. Do not "
    "emit the tool call yourself — another component will produce the "
    "structured call from your reasoning. Always reason about ONE of the "
    "listed tools — do not invent new tool names."
)


def _format_api_specs(api_specs: list[dict[str, Any]]) -> str:
    """Render api_specs list into a compact human-readable block for the Reasoner.

    Handles two parameter formats:
      - Old format: spec["parameters"]["properties"] dict
      - New format: spec["required_parameters"] / spec["optional_parameters"] lists
    """
    lines: list[str] = []
    for spec in api_specs:
        name = spec.get("api_name", "")
        desc = spec.get("api_description", "").strip()
        # Strip boilerplate subfunction prefix to keep prompts shorter
        if "This is the subfunction for tool" in desc:
            parts = desc.split('"')
            quoted = [
                p.strip() for p in parts
                if p.strip()
                and not p.strip().startswith(",")
                and "subfunction" not in p
                and "you can use" not in p
            ]
            desc = quoted[-1].lstrip(":").strip() if quoted else desc

        lines.append(f"**{name}** — {desc}")

        # ── Old format: parameters.properties ────────────────────────────
        params_obj = spec.get("parameters", {})
        if params_obj and "properties" in params_obj:
            properties = params_obj.get("properties", {})
            required = set(params_obj.get("required", []))
            if properties:
                for param_name, param_info in properties.items():
                    ptype = param_info.get("type", "string")
                    pdesc = param_info.get("description", "").strip()
                    req_marker = "required" if param_name in required else "optional"
                    line = f"  - {param_name} ({ptype}, {req_marker})"
                    if pdesc:
                        line += f": {pdesc}"
                    lines.append(line)
            else:
                lines.append("  (no parameters)")

        # ── New format: required_parameters / optional_parameters lists ───
        else:
            req_params  = spec.get("required_parameters", [])
            opt_params  = spec.get("optional_parameters", [])
            if req_params or opt_params:
                for p in req_params:
                    pname = p.get("name", "")
                    ptype = p.get("type", "STRING")
                    pdesc = p.get("description", "").strip()
                    line  = f"  - {pname} ({ptype}, required)"
                    if pdesc:
                        line += f": {pdesc}"
                    lines.append(line)
                for p in opt_params:
                    pname = p.get("name", "")
                    ptype = p.get("type", "STRING")
                    pdesc = p.get("description", "").strip()
                    line  = f"  - {pname} ({ptype}, optional)"
                    if pdesc:
                        line += f": {pdesc}"
                    lines.append(line)
            else:
                lines.append("  (no parameters)")

        lines.append("")  # blank line between tools

    return "\n".join(lines).rstrip()


def build_reasoner_user(
    task_prompt: str,
    available_tools: list[str] | None = None,
    api_specs: list[dict[str, Any]] | None = None,
) -> str:
    """Build the reasoner user message.

    When `api_specs` is provided, it renders the full tool signatures (name,
    description, parameters) so the reasoner knows exactly what arguments to
    produce.  Falls back to a plain name list when specs are absent.
    """
    if api_specs:
        tools_block = f"### Available tools\n{_format_api_specs(api_specs)}\n\n"
    elif available_tools:
        tools = "\n".join(f"- {t}" for t in available_tools)
        tools_block = f"### Available tools\n{tools}\n\n"
    else:
        tools_block = ""
    return (
        f"{tools_block}"
        "### Task\n"
        f"{task_prompt.strip()}\n\n"
        "Reason about which tool from the list to call and why."
    )


def build_instructor_input_with_tools(
    task_prompt: str,
    thought: str,
    available_tools: list[str] | None = None,
) -> str:
    """Defensive variant of build_instructor_input that re-states the tool list.

    Use this for zero-shot instructors. SFT'd instructors should use
    build_instructor_input directly (matches their training distribution).
    """
    base = build_instructor_input(task_prompt, thought)
    if not available_tools:
        return base
    tools = "\n".join(f"- {t}" for t in available_tools)
    return (
        f"### Available tools\n{tools}\n\n"
        f"{base}"
    )


# ---------------------------------------------------------------------------
# 4) BASELINE — single-pass tool-call generator
# ---------------------------------------------------------------------------
# Structured single-shot prompt for the baseline run.
# Same `### Task / ### Available tools` framing as the SFT prompts so the model
# sees a consistent input style, but no separate reasoning stage. The model
# thinks internally and outputs the tool call directly. Easier to extract,
# avoids the two-pass accuracy regression.

BASELINE_SYSTEM_PROMPT = (
    "You are a tool-using assistant. Pick exactly one tool from the provided "
    "list and emit a single tool call. You may reason briefly, but your final "
    "output line MUST be in this format:\n"
    "    #TOOL[<tool_name>]{\"arg\": \"value\"}\n"
    "Arguments must be valid JSON. Use only tools from the list."
)


def build_baseline_input(task_prompt: str, available_tools: list[str] | None = None) -> str:
    """Single-pass baseline prompt — task + tools, model emits the tool call directly."""
    tools_block = ""
    if available_tools:
        tools = "\n".join(f"- {t}" for t in available_tools)
        tools_block = f"### Available tools\n{tools}\n\n"
    return (
        f"{tools_block}"
        "### Task\n"
        f"{task_prompt.strip()}\n\n"
        "Reason briefly about which tool to call, then output the tool call as "
        "the last line in the format: #TOOL[tool_name]{\"arg\": \"value\"}."
    )
