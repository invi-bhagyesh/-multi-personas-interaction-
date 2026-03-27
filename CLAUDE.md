# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying **persona-induced bias in multi-agent LLM interactions** — specifically "epistemic collusion," where persona-conditioned agents coordinate in ways that degrade downstream user accuracy. Paper: https://arxiv.org/abs/2511.11789

All experiments live under `Persona-Induced-Bias-in-MAS/`.

## Running Experiments

```bash
cd Persona-Induced-Bias-in-MAS

# Run the full pipeline (accuracy → debate → collaboration)
./run.sh

# Single-agent accuracy baseline
python code/accuracy.py --group <0-2> --model <gpt|claude|deepseek|gemini|llama>

# Two-agent debate (CPS)
python code/cps.py --group <0-2> --model <model> --persona1 <0-9> --persona2 <0-9> --turn <1-16> --initial <T|F>

# Multi-agent collaboration/persuasion
python code/collaboration.py --dataset <cps|persuade> --group <0-2> --model <model> \
  --num1 <int> --num2 <int> --persona1 <0-9> --persona2 <0-9> --turn <1-16> --initial <T|F|support|oppose>
```

## Evaluation (post-hoc analysis)

```bash
python code/evaluation/eval_accuracy.py    # Accuracy per persona
python code/evaluation/eval_cps.py         # Debate convergence analysis
python code/evaluation/eval_persuade.py    # Persuasion conformity analysis
```

## Key CLI Parameters

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `--group` | 0–2 | Persona group: 0=neutral, 1=gender identities, 2=racial identities |
| `--model` | gpt, claude, deepseek, gemini, llama | LLM backend |
| `--persona1/2` | 0–9 | Index into the selected persona group |
| `--initial` | T, F, support, oppose | Which agent starts with the correct/support stance |
| `--turn` | 1–16 | Max debate/persuasion rounds |

## Architecture

### Core Modules

**`code/client.py`** — LLM dispatcher. `send_client(model, ...)` routes to `send_openai/claude/deepseek/gemini/llama()`. All backends include retry logic (up to 5 attempts). Requires a `KEY` module (not in repo) with an `api_key` dict containing `"uniapi"`, `"claude"`, etc.

**`code/utils.py`** — Shared state. Defines the `Agent` class (holds persona, system prompt, task prompt, output history). Contains persona group lists and parsing helpers: `extract_stance()`, `extract_message()`, `extract_option()`.

**`code/prompts.py`** — All prompt templates. Behavioral personas are injected via `personas_prompt()`. Agents use structured XML-like tags (`<message>`, `<stance>`, `<other_people_message>`) for parseable outputs.

### Experiment Scripts

- **`accuracy.py`**: Single-agent baseline over `gpqa_455.json`. Uses 32-process pool (batch size 15). Outputs to `results/accuracy/{model}_group{group}.json`.
- **`cps.py`**: Pairwise debate — Agent 1 starts with correct answer, Agent 2 with wrong (or vice versa). Early-stops when both agents converge. 16-process pool (batch size 20). Outputs to `results/cps/{model}/{group}_{p1}{p2}_{initial}_turn{N}/`.
- **`persuade.py`**: Two-agent persuasion. Persuader tries to flip persuadee's stance. Early-stops on stance flip.
- **`collaboration.py`**: General multi-agent orchestrator. Handles both CPS (N debaters with split initial answers) and persuasion (1 persuadee + M persuaders).

### Data Files (`data/`)

- `gpqa_455.json` — 455 GPQA benchmark questions with correct option
- `gpqa_TF.json` — Same questions with pre-generated correct/wrong reasoning chains for debate seeding
- `filtered_claim.json` — Controversial claims for persuasion task
- `persuade_cw.json` — Claims with pre-generated support/oppose arguments and counterarguments

### Results Layout

```
results/
  accuracy/{model}_group{group}.json
  cps/{model}/{group}_{persona1}{persona2}_{initial}_turn{N}/*.json
  persuade/{model}/...
```

## Dependencies

No `requirements.txt` is tracked. Key packages: `openai`, `anthropic`, `google-genai`, `vllm`, `tqdm`. The `KEY` module with API keys must be created locally and is excluded from git.

Llama requires a local vLLM server running at the configured endpoint before running experiments with `--model llama`.
