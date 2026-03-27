"""
Single-persona accuracy baseline (OpenCharacter).

Each persona-LoRA model answers MMLU questions independently.
Also runs the base model (no adapter) as the no-persona baseline.

Produces a JSON file that maps to the paper's accuracy table (Table 3 appendix).
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

REPO    = "maius/llama-3.1-8b-it-personas"
BASE_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

TASK_PROMPT = (
    "Answer the following question. This is a hard question, which will probably "
    "require you to break down the question into multiple sub-questions, that you "
    "will then need to compose into your final answer. "
    "{question}: A) {A}, B) {B}, C) {C}, D) {D}\n"
    "Explain your answer, and ensure that your final answer (A) or (B) or (C) or (D) "
    "is positioned at the very end of your output inside parentheses, adhering to the "
    "format 'Final answer: (answer)'."
)


def load_base(base_id=BASE_ID):
    """Load base model and tokenizer once."""
    print(f"[loading] base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def apply_adapter(base_model, persona, repo=REPO):
    """Stack a LoRA adapter onto the base model. Returns PeftModel."""
    print(f"[loading] adapter: {persona}")
    model = PeftModel.from_pretrained(base_model, repo, subfolder=persona)
    model.eval()
    return model


def generate(tokenizer, model, memory, max_new_tokens=512):
    """Generate a reply from a chat-formatted message list."""
    inputs = tokenizer.apply_chat_template(
        memory, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def run_persona(persona, data, tokenizer, base_model, max_new_tokens=512, repo=REPO):
    """Run accuracy evaluation for one persona (or 'base' for no adapter)."""
    if persona == "base":
        model = base_model
    else:
        model = apply_adapter(base_model, persona, repo)

    results = []
    correct = 0
    for item in tqdm(data, desc=persona):
        prompt = TASK_PROMPT.format(
            question=item["question"],
            A=item["options"][0], B=item["options"][1],
            C=item["options"][2], D=item["options"][3],
        )
        memory = [{"role": "user", "content": prompt}]
        raw = generate(tokenizer, model, memory, max_new_tokens)
        predicted = extract_option(raw)
        is_correct = (predicted == item["answer"]) if predicted else False
        if is_correct:
            correct += 1
        results.append({
            "case": item["case"],
            "answer": item["answer"],
            "predicted": predicted,
            "correct": is_correct,
            "raw": raw,
        })

    accuracy = correct / len(data) if data else 0.0
    print(f"  {persona}: {correct}/{len(data)} = {accuracy:.3f}")

    # Unload adapter if we loaded one
    if persona != "base" and isinstance(model, PeftModel):
        model.unload()
        torch.cuda.empty_cache()

    return {"persona": persona, "accuracy": accuracy, "results": results}
