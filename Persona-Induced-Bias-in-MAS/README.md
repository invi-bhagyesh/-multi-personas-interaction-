# Persona-Induced Bias in Multi-Agent Systems

Based on [Li et al. 2025](https://arxiv.org/abs/2511.11789). Extended with OpenCharacter LoRA-based persona experiments.

## Setup

```bash
cd Persona-Induced-Bias-in-MAS
pip install -r requirements.txt
```

For vLLM backend (recommended):
```bash
pip install vllm
```

## Quick Start

Run everything (prepare data + accuracy + CPS):
```bash
python code/run.py --vllm
```

## Step-by-Step

### 1. Prepare Data

Downloads MMLU-Pro and generates correct/wrong reasoning chains:
```bash
python code/run.py --experiments prepare_data
```

Requires `OPENAI_API_KEY` env var (uses gpt-4o-mini for TF generation).
Skips if data files already exist.

### 2. Accuracy Baseline

Single-agent accuracy per persona:
```bash
python code/run.py --experiments accuracy --vllm
```

Output: `code/results/accuracy/opencharacter_mmlu_pro.json`

### 3. CPS (Table 1 + Table 2)

Runs trustworthiness, insistence, and conformity matrix:
```bash
python code/run.py --experiments cps --vllm
```

Output: `code/results/cps_opencharacter/table1.json`, `table2.json`

### 4. Prompt-Based Comparison

Same experiments using system-prompt personas instead of LoRA:
```bash
# CPS
python code/run.py --experiments cps --vllm --prompt-based

# Accuracy
python code/run.py --experiments accuracy --vllm --prompt-based
```

Output goes to `*_prompt_based` paths.

## Flags

| Flag | Effect |
|------|--------|
| `--vllm` | Use vLLM backend (fast) |
| `--prompt-based` | System-prompt personas instead of LoRA |
| `--experiments X Y` | Run specific experiments: `prepare_data`, `accuracy`, `cps` |
| `--config path` | Custom config file (default: `config.yaml`) |

## Examples

```bash
# OpenCharacter LoRA, fast
python code/run.py --experiments cps --vllm

# Prompt-based, fast
python code/run.py --experiments cps --vllm --prompt-based

# OpenCharacter LoRA, HF backend (slow, no vLLM needed)
python code/run.py --experiments cps

# Accuracy + CPS together
python code/run.py --experiments accuracy cps --vllm

# Full pipeline
python code/run.py --vllm
```

## Config

Edit `config.yaml` to change personas, benchmark, number of questions, etc.

## Estimated Runtimes (A40, 50 questions, 4 personas)

| Experiment | vLLM | HF batched |
|-----------|------|-----------|
| Accuracy | ~3 min | ~10 min |
| CPS Table 1 | ~5 min | ~20 min |
| CPS Table 2 | ~15 min | ~60+ min |
