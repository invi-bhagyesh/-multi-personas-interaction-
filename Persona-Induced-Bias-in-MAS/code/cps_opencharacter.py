"""
Experiment 1 — Trustworthiness & Insistence (OpenCharacter)
=============================================================
Replicates the §4 single-persona dyadic interaction from the paper but using
LoRA behavioral personas instead of prompt-injected demographic personas.

Two modes controlled by --mode:

  trustworthiness  (default)
      Agent_p  = persona model  → starts with the WRONG answer
      Agent_d  = base model     → starts with the CORRECT answer
      Measure: does Agent_d (default) conform to Agent_p?
      T(p) = fraction of cases where Agent_d switches to Agent_p's answer

  insistence
      Agent_p  = persona model  → starts with the CORRECT answer
      Agent_d  = base model     → starts with the WRONG answer
      Measure: does Agent_p resist conforming to Agent_d?
      I(p) = 1 - fraction of cases where Agent_p switches to Agent_d's answer

Both use a single interaction round (--turn 1), mirroring §4 of the paper.

Usage:
    # measure trustworthiness for all personas
    python cps_opencharacter.py --mode trustworthiness --n 50

    # measure insistence for specific personas
    python cps_opencharacter.py --mode insistence --personas sarcasm sycophancy nonchalance --n 50

    # both metrics at once
    python cps_opencharacter.py --mode both --n 100
"""

import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option, Agent
from prompts import gpqa_task_prompt, gpqa_other_answer, gpqa_interaction_prompt, personas_prompt
from client import send_opencharacter

REPO    = "maius/llama-3.1-8b-it-personas"
BASE_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ALL_PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]

# Label used in prompts when referring to the other agent.
# The original paper used demographic labels; here we use the persona name.
DEFAULT_LABEL = "another agent"


def load_base_model():
    print("[loading] base model (no adapter) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def load_persona_model(persona: str, base_tokenizer, base_weights):
    """Stack a LoRA adapter on top of the already-loaded base weights."""
    print(f"[loading] '{persona}' adapter ...")
    model = PeftModel.from_pretrained(base_weights, REPO, subfolder=persona)
    model.eval()
    return base_tokenizer, model


def ask_model(tokenizer, model, memory: list, max_new_tokens: int = 512) -> str:
    """Send a chat-formatted memory list to a HuggingFace model and return the reply."""
    inputs = tokenizer.apply_chat_template(
        memory, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = out[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_initial_memory(task_instr: str, initial_response: str, other_label: str) -> list:
    """Build the conversation up to (but not including) the interaction turn."""
    return [
        {"role": "user",      "content": task_instr},
        {"role": "assistant", "content": initial_response},
        {"role": "user",      "content":
            gpqa_other_answer(other_label, "")  # placeholder; filled below
        },
    ]


def run_one_case(
    data: dict,
    persona: str,
    mode: str,
    tokenizer_p, model_p,
    tokenizer_d, model_d,
    max_new_tokens: int = 512,
) -> dict:
    """
    Run a single debate case.

    mode='trustworthiness': persona_agent has WRONG answer, default has CORRECT.
                            We track whether the default agent conforms.
    mode='insistence':      persona_agent has CORRECT answer, default has WRONG.
                            We track whether the persona agent resists.
    """
    if mode == "trustworthiness":
        initial_p = data["wrong"]
        initial_d = data["correct"]
        correct_option = data["correct_option"]
        wrong_option   = data["wrong_option"]
    else:  # insistence
        initial_p = data["correct"]
        initial_d = data["wrong"]
        correct_option = data["correct_option"]
        wrong_option   = data["wrong_option"]

    # ── Persona agent sees default's initial answer, then updates ──────────
    task_p = gpqa_task_prompt(DEFAULT_LABEL, data)
    memory_p = [
        {"role": "user",      "content": task_p},
        {"role": "assistant", "content": initial_p},
        {"role": "user",      "content":
            gpqa_other_answer(DEFAULT_LABEL, initial_d) +
            gpqa_interaction_prompt(DEFAULT_LABEL)},
    ]
    reply_p = ask_model(tokenizer_p, model_p, memory_p, max_new_tokens)

    # ── Default agent sees persona's initial answer, then updates ──────────
    task_d = gpqa_task_prompt(persona, data)
    memory_d = [
        {"role": "user",      "content": task_d},
        {"role": "assistant", "content": initial_d},
        {"role": "user",      "content":
            gpqa_other_answer(persona, initial_p) +
            gpqa_interaction_prompt(persona)},
    ]
    reply_d = ask_model(tokenizer_d, model_d, memory_d, max_new_tokens)

    option_p = extract_option(reply_p)
    option_d = extract_option(reply_d)

    if mode == "trustworthiness":
        # Did the default agent conform to the persona agent (adopt the wrong answer)?
        conformed = option_d == wrong_option
    else:
        # Did the persona agent resist (keep the correct answer)?
        resisted = option_p == correct_option

    return {
        "case":           data["case"],
        "correct_option": correct_option,
        "wrong_option":   wrong_option,
        "reply_persona":  reply_p,
        "reply_default":  reply_d,
        "option_persona": option_p,
        "option_default": option_d,
        # metric flag
        "conformed_default" if mode == "trustworthiness" else "resisted_persona":
            conformed if mode == "trustworthiness" else resisted,
    }


def run_persona_experiment(
    persona: str,
    mode: str,
    data: list,
    tokenizer_d, model_d,
    max_new_tokens: int = 512,
) -> dict:
    # Load persona adapter on top of the base weights
    tokenizer_p, model_p = load_persona_model(persona, tokenizer_d, model_d)

    cases = []
    for item in tqdm(data, desc=f"{persona} ({mode})"):
        result = run_one_case(
            item, persona, mode,
            tokenizer_p, model_p,
            tokenizer_d, model_d,
            max_new_tokens,
        )
        cases.append(result)

    if mode == "trustworthiness":
        score = sum(c["conformed_default"] for c in cases) / len(cases)
        metric_name = "trustworthiness"
    else:
        score = sum(c["resisted_persona"] for c in cases) / len(cases)
        metric_name = "insistence"

    print(f"  {persona:15s}  {metric_name}: {score:.3f}")

    # Unload the adapter to free GPU memory
    model_p = model_p.unload()  # returns the base model, freeing LoRA weights
    torch.cuda.empty_cache()

    return {"persona": persona, metric_name: score, "cases": cases}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trustworthiness", "insistence", "both"],
                        default="trustworthiness")
    parser.add_argument("--personas", nargs="+", default=ALL_PERSONAS,
                        choices=ALL_PERSONAS, metavar="PERSONA")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N questions (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default="results/cps_opencharacter")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    with open("../data/gpqa_TF.json") as f:
        data = json.load(f)
    if args.n:
        data = data[: args.n]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model once; reuse across all personas
    tokenizer_d, model_d = load_base_model()

    modes = ["trustworthiness", "insistence"] if args.mode == "both" else [args.mode]

    for mode in modes:
        all_results = []
        print(f"\n{'='*50}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*50}")

        for persona in args.personas:
            result = run_persona_experiment(
                persona, mode, data,
                tokenizer_d, model_d,
                args.max_new_tokens,
            )
            all_results.append(result)

            out_path = os.path.join(args.output_dir, f"{mode}.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=4)

        print(f"\n=== {mode.upper()} Summary ===")
        metric = "trustworthiness" if mode == "trustworthiness" else "insistence"
        for r in all_results:
            print(f"  {r['persona']:15s}  {r[metric]:.3f}")


if __name__ == "__main__":
    main()
