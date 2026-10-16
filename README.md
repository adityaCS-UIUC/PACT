# PACT

**Pruned Agent Call Throughput.** A speculative early-exit pipeline for LLM tool-use agents, from CS 598 (Systems for Generative AI) at UIUC, Spring 2026.

A large reasoner writes a tool-use trace one sentence at a time. A small speculator scores each prefix as ready or not ready. When its confidence crosses a threshold τ, a second small model turns that partial thought into the next tool call, instead of waiting for the full trace.

```
User query
    │
    ▼
Reasoner (QwQ-32B)
    │  sentence prefix
    ▼
Speculator (Phi-3-mini + LoRA)   READY / NOT_READY at threshold τ
    │
    ▼
Instructor (Phi-3-mini + LoRA)   partial thought → tool call
    │
    ▼
Tool call
```

This repository is the local project tree. Training shards, trace dumps, logs, and virtual environments are not included.

## Layout

| Path | Role |
| --- | --- |
| `pact/` | Agents, speculator model, SFT, ToolBench data pipeline, evaluation |
| `qwen_reasoning/` | Modal job that generates QwQ reasoning traces |
| `modal_speculator_baseline/` | Modal SFT and threshold sweeps for the speculator |
| `modal_phi3_baseline_eval/` | Phi-3 baseline evaluations |
| `test_pipeline.py` | Local pipeline smoke test |
| `main.md` | Course pipeline notes |
| `draft.tex`, `CS_598_Final_Report.pdf` | Write-up |

`main.md` still names a few design paths (`adapters/`, `pact/eval/run_pact.py`) that are not in this tree. The scripts that are checked in live under `pact/`, `pact/evaluation/`, and the `modal_*` folders.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set secrets in the environment, not in the repo:

| Variable | Use |
| --- | --- |
| `OPENAI_API_KEY` | GPT judge that labels prefixes READY or NOT_READY |
| `HF_TOKEN` | Gated Hugging Face models such as Phi-3 |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal GPU jobs |
