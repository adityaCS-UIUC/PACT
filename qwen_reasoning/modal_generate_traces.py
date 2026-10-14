"""
Generate Qwen QwQ-32B reasoning traces for PACT speculator retraining.

Pipeline
--------
For each unique query_id in the input JSONL:
  1. Build a prompt with the task prompt, available tools, and full API specs.
  2. Run Qwen QwQ-32B to produce: <free-form reasoning> + #TOOL[name]{args}
  3. Parse and save the full trace, reasoning, and tool call.

The saved traces are fed to the LLM judge (GPT-4.1-mini) to label the optimal
READY point in each reasoning trace.  Those labeled traces become the SFT
training signal for retraining Phi-3-mini so it understands Qwen's reasoning
style.

Input  : data/main_ds_part00.jsonl  (query_id, prompt, available_tools, api_specs, relevant_apis)
Output : Modal volume 'pact-results' → qwen_reasoning/<run_name>/
           - <query_id>.json   one file per task (resumable)
           - traces.jsonl      consolidated append-only log

Usage
-----
    # Small trial
    modal run -m qwen_reasoning.modal_generate_traces --limit 5

    # Full run (~10K tasks, parallel across containers)
    modal run -m qwen_reasoning.modal_generate_traces \\
        --input-path data/shared_00.jsonl \\
        --run-name qwen_traces_v1 \\
        --batch-size 20
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

APP_NAME = "qwen-reasoning-traces"
GPU_TYPE = "H100"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("pact")
)

model_cache_vol = modal.Volume.from_name("pact-model-cache", create_if_missing=True)
results_vol     = modal.Volume.from_name("pact-results",     create_if_missing=True)

VOLUMES = {
    "/cache":   model_cache_vol,
    "/results": results_vol,
}

SECRETS = [modal.Secret.from_name("huggingface")]

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REASONER_MODEL = "Qwen/QwQ-32B-Preview"

# Matches the last #TOOL[...]{...} in the output (handles multi-line JSON args)
_TOOL_RE = re.compile(r"#TOOL\[([\w./ -]+)\]\s*(\{[^#]*?\})", re.DOTALL)

# System prompt — instructs Qwen to reason then emit exactly one #TOOL call.
# The MUST / WARNING lines are intentional: QwQ-32B tends to trail off without
# them when max_new_tokens is hit before it has written the tool call.
SYSTEM_PROMPT = (
    "You are a tool-using assistant. Given a task and available tools with "
    "their API specs, identify the best tool and arguments in as few words as "
    "possible. Be brief and direct — do NOT over-explain.\n\n"
    "Format:\n"
    "  1. One or two sentences identifying the right tool and why.\n"
    "  2. One sentence stating the argument values.\n"
    "  3. Immediately output the tool call on its own line:\n"
    "     #TOOL[tool_name]{\"arg1\": \"value1\"}\n\n"
    "RULES (strictly enforced):\n"
    "- Keep your reasoning under 150 words — be concise.\n"
    "- The tool_name in #TOOL[...] MUST be the EXACT api_name string shown in "
    "the list (e.g. #TOOL[GetClimateScoreByGps]{...}). Copy it character for "
    "character — do NOT paraphrase or use a description.\n"
    "- Use ONLY tool names from the provided list — never invent new names.\n"
    "- Arguments must be valid JSON matching the tool's parameter spec.\n"
    "- For tools with no parameters use an empty object: #TOOL[tool_name]{}\n"
    "- The #TOOL line MUST be the very last line of your response.\n"
    "- WARNING: A response without a valid #TOOL call is considered invalid."
)


# ---------------------------------------------------------------------------
# Remote Modal function — one H100 per batch
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=60 * 120,          # 2 h — 100 tasks × ~30s each = ~50 min typical
    scaledown_window=600,
    max_containers=10,      # never exceed 10 simultaneous GPU containers
)
def generate_traces_batch(
    tasks: list[dict[str, Any]],
    run_name: str,
    model_id: str = REASONER_MODEL,
    max_new_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Generate Qwen reasoning traces for a batch of tasks on a single H100.

    Each task dict must have:
        query_id, prompt, available_tools, api_specs, target_tool_call
    """
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pact.sft.prompts import build_reasoner_user

    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/cache/hf")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="/cache/hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model ready. Processing {len(tasks)} tasks.")

    out_dir = Path(f"/results/qwen_reasoning/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_results: list[dict[str, Any]] = []

    for task in tasks:
        qid      = task["query_id"]
        out_file = out_dir / f"{qid}.json"

        if out_file.exists():
            print(f"[skip] {qid}")
            batch_results.append(json.loads(out_file.read_text()))
            continue

        prompt           = task["prompt"]
        available_tools  = task.get("available_tools", [])
        api_specs        = task.get("api_specs", [])
        # target_tool_call is pre-normalised to {tool_name, arguments} from step 0
        target_tool_call = task.get("target_tool_call", {})

        # Strip fields that waste tokens but add no reasoning value:
        # template_response (large schema), method (GET/POST), category_name
        trimmed_specs = [
            {k: v for k, v in s.items()
             if k not in {"template_response", "method", "category_name", "tool_name"}}
            for s in api_specs
        ]

        # User message: trimmed API specs go to Qwen so it knows the parameter names
        user_msg = (
            build_reasoner_user(prompt, available_tools, trimmed_specs)
            + "\n\nIMPORTANT: End your response with a #TOOL call."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen_ms = (time.perf_counter() - t0) * 1000.0

        full_output = tok.decode(
            out[0, ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        # Find the LAST #TOOL[...]{...} in the output — Qwen sometimes outputs
        # multiple before settling on one; we want the final answer.
        all_matches = list(_TOOL_RE.finditer(full_output))
        if all_matches:
            tool_match    = all_matches[-1]
            raw_reasoning = full_output[:tool_match.start()].strip()
            tool_call_raw = tool_match.group(0)
            try:
                tool_args = json.loads(tool_match.group(2))
            except json.JSONDecodeError:
                tool_args = {}
            tool_call_parsed = {
                "tool_name": tool_match.group(1).strip(),
                "arguments": tool_args,
            }
        else:
            raw_reasoning    = full_output
            tool_call_raw    = None
            tool_call_parsed = None

        gt_name   = target_tool_call.get("tool_name", "")
        pred_name = tool_call_parsed["tool_name"] if tool_call_parsed else ""
        # Accept any relevant api as correct (not just the first one)
        relevant_names = {a[1] for a in task.get("relevant_apis", [])} or {gt_name}
        tool_correct = pred_name in relevant_names if pred_name else False

        record = {
            # ── Identifiers ──────────────────────────────────────────────
            "query_id": qid,
            # ── Input context (needed for judge + SFT) ───────────────────
            "prompt":          prompt,
            "available_tools": available_tools,
            "api_specs":       api_specs,
            # ── Ground truth ─────────────────────────────────────────────
            # target_tool_call = first step; gt_calls = all steps (no thought text)
            "target_tool_call": target_tool_call,
            "gt_calls":         task.get("gt_calls", []),
            "relevant_apis":    task.get("relevant_apis", []),
            # ── Qwen output ──────────────────────────────────────────────
            "qwen_full_output":  full_output,
            "qwen_reasoning":    raw_reasoning,
            "tool_call_raw":     tool_call_raw,
            "tool_call_parsed":  tool_call_parsed,
            # ── Quick accuracy signal ────────────────────────────────────
            # tool_name_correct = True if pred matches any relevant_api
            "tool_name_correct": tool_correct,
            "has_tool_call":     tool_call_parsed is not None,
            # ── Timing / token counts ────────────────────────────────────
            "generation_ms": round(gen_ms, 1),
            "reasoning_tokens": len(
                tok.encode(raw_reasoning, add_special_tokens=False)
            ) if raw_reasoning else 0,
            "total_output_tokens": len(
                tok.encode(full_output, add_special_tokens=False)
            ),
        }

        out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        batch_results.append(record)

        status = "✓" if record["tool_name_correct"] else ("⚠" if record["has_tool_call"] else "✗")
        print(
            f"[{status}] {qid}  "
            f"pred={pred_name or 'NONE'}  "
            f"gt={gt_name}  "
            f"ms={gen_ms:.0f}"
        )

    # Append to consolidated JSONL
    traces_file = out_dir / "traces.jsonl"
    with traces_file.open("a") as fh:
        for r in batch_results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    results_vol.commit()
    return batch_results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    input_path:     str = "data/main_ds_part00.jsonl",
    run_name:       str = "qwen_traces",
    limit:          int = 0,        # 0 = process all
    batch_size:     int = 100,      # 1000 tasks / 100 = 10 containers = 1 per GPU (perfect fit)
    model_id:       str = REASONER_MODEL,
    max_new_tokens: int = 1024,
) -> None:
    """Read input JSONL, deduplicate by query_id, dispatch batches to Modal.

    Batches run in parallel across Modal H100 containers — ~500 containers for
    10K tasks at batch_size=20.  Each container loads Qwen once and processes
    its batch sequentially.
    """
    # ── Deduplicate: one task per query_id (data is already query-level) ──
    # shard_00.jsonl schema:
    #   query_id, prompt, available_tools, api_specs, ground_truth_path
    # ground_truth_path is a list of {thought, action, action_input} steps.
    # We target step 0 (first tool call) for trace generation.
    seen: set[str] = set()
    tasks: list[dict[str, Any]] = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            if qid in seen:
                continue
            seen.add(qid)

            # Ground truth: prefer ground_truth_path (action/action_input),
            # fall back to relevant_apis (api_name only, no arguments) which is
            # what shared_00.jsonl uses.
            gtp = row.get("ground_truth_path", [])
            relevant_apis = row.get("relevant_apis", [])

            if gtp:
                # Strip thought, keep action + action_input only
                gt_calls = [
                    {"action": s.get("action", ""),
                     "action_input": s.get("action_input", {})}
                    for s in gtp
                ]
                first = gtp[0]
                target_tool_call = {
                    "tool_name": first.get("action", ""),
                    "arguments": first.get("action_input", {}),
                }
            elif relevant_apis:
                # relevant_apis = [["Category", "api_name"], ...]
                gt_calls = [{"action": a[1], "action_input": {}} for a in relevant_apis]
                target_tool_call = {
                    "tool_name": relevant_apis[0][1],
                    "arguments": {},
                }
            else:
                gt_calls = []
                target_tool_call = {"tool_name": "", "arguments": {}}

            tasks.append({
                "query_id":          qid,
                "prompt":            row.get("prompt", ""),
                "available_tools":   row.get("available_tools", []),
                "api_specs":         row.get("api_specs", []),
                "relevant_apis":     relevant_apis,
                "gt_calls":          gt_calls,
                "target_tool_call":  target_tool_call,
            })

    if limit and limit > 0:
        tasks = tasks[:limit]

    n_total = len(tasks)

    # ── Skip already-completed tasks by checking the volume ───────────────
    # This avoids dispatching containers for work that's already done,
    # making re-runs after failures or disconnects essentially free.
    volume_prefix = f"qwen_reasoning/{run_name}"
    try:
        existing_ids: set[str] = {
            Path(entry.path).stem          # filename without .json == query_id
            for entry in results_vol.listdir(volume_prefix, recursive=False)
            if entry.path.endswith(".json")
        }
        tasks = [t for t in tasks if t["query_id"] not in existing_ids]
        print(
            f"Volume check: {len(existing_ids)} already done, "
            f"{len(tasks)} remaining out of {n_total}."
        )
    except Exception as exc:
        print(f"Could not read volume (first run?): {exc}. Processing all {n_total} tasks.")

    if not tasks:
        print("Nothing to do — all tasks already completed.")
        return

    batches = [
        tasks[i : i + batch_size]
        for i in range(0, len(tasks), batch_size)
    ]

    print(
        f"Input       : {input_path}\n"
        f"To process  : {len(tasks)} tasks\n"
        f"Batches     : {len(batches)} × up to {batch_size} tasks each\n"
        f"Run name    : {run_name}\n"
        f"Max tokens  : {max_new_tokens}"
    )

    # ── Dispatch all batches in parallel via Modal .map() ─────────────────
    total_done    = 0
    total_correct = 0
    total_no_call = 0

    for batch_results in generate_traces_batch.map(
        batches,
        kwargs={
            "run_name":       run_name,
            "model_id":       model_id,
            "max_new_tokens": max_new_tokens,
        },
        return_exceptions=True,
    ):
        if isinstance(batch_results, Exception):
            print(f"[batch error] {batch_results}")
            continue
        for r in batch_results:
            total_done += 1
            if r.get("tool_name_correct"):
                total_correct += 1
            if not r.get("has_tool_call"):
                total_no_call += 1

    print(
        f"\n── Results ────────────────────────────────\n"
        f"  Newly generated   : {total_done}\n"
        f"  Tool name correct : {total_correct}/{total_done} "
        f"({total_correct/max(total_done, 1):.1%})\n"
        f"  Missing #TOOL call: {total_no_call} "
        f"({total_no_call/max(total_done, 1):.1%})\n"
        f"  Results volume    : pact-results → qwen_reasoning/{run_name}/"
    )
