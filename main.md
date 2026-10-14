# PACT: Pruned Agent Call Throughput

**CS 598 SYS-GEN-AI — Spring 2026, UIUC**

PACT is a three-model speculative early-exit pipeline for LLM tool-use agents. It speeds up inference by having a lightweight **Speculator** decide when a reasoning trace is "good enough" to hand off to a smaller **Instructor** model, rather than always waiting for the full reasoning from a large **Reasoner**.

## Architecture

```
User Query
    │
    ▼
Reasoner (QwQ-32B)          ← generates reasoning sentence by sentence
    │  each sentence prefix
    ▼
Speculator (Phi-3-mini + LoRA)  ← scores: READY or NOT_READY?
    │  fires at threshold τ
    ▼
Instructor (Phi-3-mini + LoRA)  ← translates thought → #TOOL[name]{args}
    │
    ▼
Tool Call
```

## Repository Structure

```
adapters/
  speculator/       ← fine-tuned Phi-3-mini LoRA adapter (READY/NOT_READY)
  instructor/       ← fine-tuned Phi-3-mini LoRA adapter (tool call generation)
pact/
  eval/             ← evaluation scripts: run_baseline.py, run_pact.py, run_sweep.py, grade.py
  sft/              ← prompt builders and SFT data utilities
  data_pipeline/    ← data extraction and preprocessing
modal_speculator_baseline/  ← SFT training scripts (Modal GPU)
modal_phi3_baseline_eval/   ← baseline evaluation scripts
extract_ground_truth_cumulative.py  ← split reasoning traces into cumulative sentence prefixes
judge_cumulative_ready.py           ← GPT-4.1 mini labels each prefix READY/NOT_READY
data/eval/tasks.jsonl               ← 500-task eval set (ToolBench queries)
requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy your API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET

# Authenticate with Modal
modal token set --token-id <MODAL_TOKEN_ID> --token-secret <MODAL_TOKEN_SECRET>
```

## Full Pipeline (end-to-end)

### Step 1 — Generate QwQ-32B reasoning traces

Run QwQ-32B on Modal H100s to generate reasoning traces for ToolBench tasks:

```bash
modal run pact/eval/run_baseline.py \
  --tasks-path data/eval/tasks.jsonl \
  --run-name qwen_traces_v1
```

Download results from the `pact-results` Modal volume:

```bash
modal volume get pact-results qwen_reasoning/qwen_traces_v1/ results/qwen_traces_v1/
```

### Step 2 — Extract cumulative sentence prefixes

Split each reasoning trace into s1, s1+s2, s1+s2+s3, … prefixes for speculator training:

```bash
python extract_ground_truth_cumulative.py \
  --input results/qwen_traces_v1/ \
  --output data/sft/cumulative_prefixes.jsonl
```

### Step 3 — Label prefixes READY / NOT_READY (GPT judge)

Use GPT-4.1 mini to label each prefix — READY if the prefix contains enough context to make the correct tool call, NOT_READY otherwise:

```bash
export OPENAI_API_KEY="sk-..."
python judge_cumulative_ready.py \
  --input data/sft/cumulative_prefixes.jsonl \
  --output data/sft/judged.jsonl

# Resume a stopped run:
python judge_cumulative_ready.py \
  --input data/sft/cumulative_prefixes.jsonl \
  --output data/sft/judged.jsonl \
  --resume
```

### Step 4 — Split into train/test

```bash
python modal_speculator_baseline/split_judged_train_test.py \
  --input data/sft/judged.jsonl \
  --train data/sft/train.jsonl \
  --test data/sft/test.jsonl
```

### Step 5 — Fine-tune Speculator (Phi-3-mini, READY/NOT_READY)

Upload training data to Modal volume then run SFT on an A100:

```bash
modal volume put pact-sft-data train.jsonl data/sft/train.jsonl

modal run modal_speculator_baseline/modal_sft_r_nr_prompt.py \
  --data-file train.jsonl \
  --run-name phi3_speculator_v1
```

Download the trained adapter:

```bash
modal volume get pact-sft-ckpts phi3_speculator_v1/ adapters/speculator/
```

### Step 6 — Fine-tune Instructor (Phi-3-mini, tool call generation)

```bash
modal run modal_speculator_baseline/modal_sft_toolcall.py \
  --data-file train.jsonl \
  --run-name phi3_instructor_v1
```

Download the trained adapter:

```bash
modal volume get pact-sft-ckpts phi3_instructor_v1/ adapters/instructor/
```

### Step 7 — Run PACT evaluation (tau sweep)

Evaluate PACT on the 500-task eval set across multiple tau thresholds:

```bash
modal run pact/eval/run_sweep.py \
  --tasks data/eval/tasks.jsonl \
  --run-name pact_eval_v1 \
  --taus 0.25 0.33 0.55 0.8
```

Or run a single tau:

```bash
modal run pact/eval/run_pact.py \
  --tasks-path data/eval/tasks.jsonl \
  --run-name pact_tau0.55 \
  --tau 0.55
```

### Step 8 — Grade results

```bash
python pact/eval/grade.py \
  --log-dir results/pact_eval_v1_pact_tau=0.55/ \
  --tasks data/eval/tasks.jsonl \
  --baseline-log-dir results/pact_eval_v1_baseline/ \
  --output results/graded_tau0.55.jsonl

python pact/eval/metrics.py results/graded_tau0.55.jsonl
```

## Pre-trained Adapters

Pre-trained LoRA adapters are included in this repository:

| Adapter | Path | Base model | Task |
|---------|------|-----------|------|
| Speculator | `adapters/speculator/` | Phi-3-mini-4k-instruct | READY / NOT_READY binary classification |
| Instructor | `adapters/instructor/` | Phi-3-mini-4k-instruct | Tool call generation (#TOOL[name]{args}) |

To use pre-trained adapters, skip Steps 5–6 and point `--speculator-adapter` / `--instructor-adapter` at these directories in run_pact.py.

## Tau (τ) explained

τ is the speculator's confidence threshold. When the speculator's P(READY) ≥ τ, it fires and passes the partial thought to the Instructor.

- **Low τ (e.g. 0.25)**: fires early → faster, less accurate
- **High τ (e.g. 0.8)**: fires late → slower, more accurate
- **τ = 1.0**: never fires early → equivalent to full reasoning

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | For GPT-4.1 mini judge (Step 3) |
| `MODAL_TOKEN_ID` | Modal authentication |
| `MODAL_TOKEN_SECRET` | Modal authentication |
