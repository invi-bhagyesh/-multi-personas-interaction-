import os
import json
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from model_utils import load_base, apply_adapter, unload_adapter, generate, generate_batch, REPO
from prepare_mmlu import task_prompt

BATCH_SIZE = 16


def run_persona(persona, data, tokenizer, base_model, max_new_tokens=512, repo=REPO):
    if persona == "base":
        model = base_model
    else:
        model = apply_adapter(base_model, persona, repo)

    # Build all prompts
    memories = []
    for item in data:
        prompt = task_prompt(item)
        memories.append([{"role": "user", "content": prompt}])

    # Batched generation
    print(f"  {persona}: generating {len(memories)} replies...")
    replies = generate_batch(tokenizer, model, memories, max_new_tokens, BATCH_SIZE)

    results = []
    correct = 0
    for item, raw in zip(data, replies):
        predicted = extract_option(raw)
        is_correct = predicted == item["answer"] if predicted else False
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

    if persona != "base":
        unload_adapter(model)

    return {"persona": persona, "accuracy": accuracy, "results": results}
