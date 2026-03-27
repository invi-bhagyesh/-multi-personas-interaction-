"""
Experiment 1 — Single-Persona Accuracy Baseline (OpenCharacter)
================================================================
Replicates accuracy.py but uses LoRA persona adapters from
maius/llama-3.1-8b-it-personas instead of prompt-injected demographic personas.

Each persona model answers every GPQA question independently (no interaction).
This gives us the per-persona accuracy baseline analogous to Table 3 in the paper.

Usage:
    python accuracy_opencharacter.py [--personas sarcasm humor ...] [--n 50]

    --personas   subset of personas to run (default: all 10)
    --n          number of questions to evaluate (default: all 455)
    --output     path for results JSON (default: results/accuracy/opencharacter.json)
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
from utils import extract_option

REPO     = "maius/llama-3.1-8b-it-personas"
BASE_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ALL_PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]

TASK_PROMPT = (
    "Answer the following question. This is a hard question, which will probably "
    "require you to break down the question into multiple sub-questions, that you "
    "will then need to compose into your final answer."
    "{question}: A) {A}, B) {B}, C) {C}, D) {D}\n"
    "Explain your answer, and ensure that your final answer (A) or (B) or (C) or (D) "
    "is positioned at the very end of your output inside parentheses, adhering to the "
    "format 'Final answer: (answer)'."
)


def load_model(persona: str):
    print(f"\n[loading] base model + '{persona}' adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, REPO, subfolder=persona)
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, question_text: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": question_text}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # temperature=0 equivalent
        )
    new_tokens = out[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_persona(persona: str, data: list, max_new_tokens: int = 512) -> dict:
    tokenizer, model = load_model(persona)
    results = []
    correct = 0
    for item in tqdm(data, desc=persona):
        prompt = TASK_PROMPT.format(
            question=item["question"],
            A=item["options"][0],
            B=item["options"][1],
            C=item["options"][2],
            D=item["options"][3],
        )
        raw = ask(tokenizer, model, prompt, max_new_tokens)
        predicted = extract_option(raw)
        is_correct = predicted == item["answer"] if predicted else False
        if is_correct:
            correct += 1
        results.append({
            "case":      item["case"],
            "answer":    item["answer"],
            "predicted": predicted,
            "correct":   is_correct,
            "raw":       raw,
        })
    accuracy = correct / len(data) if data else 0.0
    print(f"  {persona}: {correct}/{len(data)} = {accuracy:.3f}")
    # free GPU memory before loading the next adapter
    del model
    torch.cuda.empty_cache()
    return {"persona": persona, "accuracy": accuracy, "results": results}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", nargs="+", default=ALL_PERSONAS,
                        choices=ALL_PERSONAS, metavar="PERSONA")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N questions (default: all)")
    parser.add_argument("--data", type=str, default="../data/mmlu_questions.json")
    parser.add_argument("--output", type=str,
                        default="results/accuracy/opencharacter_mmlu.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data) as f:
        data = json.load(f)
    if args.n:
        data = data[: args.n]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = []
    for persona in args.personas:
        persona_result = run_persona(persona, data)
        all_results.append(persona_result)
        # checkpoint after each persona
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=4)

    print("\n=== Summary ===")
    for r in all_results:
        print(f"  {r['persona']:15s}  {r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
