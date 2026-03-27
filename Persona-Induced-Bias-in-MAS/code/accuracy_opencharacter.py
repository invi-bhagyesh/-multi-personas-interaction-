import os
import json
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from model_utils import load_base, apply_adapter, unload_adapter, generate, REPO

TASK_PROMPT = "Answer the following question. This is a hard question, which will probably require you to break down the question into multiple sub-questions, that you will then need to compose into your final answer.{}: A) {}, B) {}, C) {}, D) {} \nExplain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."


def run_persona(persona, data, tokenizer, base_model, max_new_tokens=512, repo=REPO):
    if persona == "base":
        model = base_model
    else:
        model = apply_adapter(base_model, persona, repo)

    results = []
    correct = 0
    for item in tqdm(data, desc=persona):
        prompt = TASK_PROMPT.format(
            item["question"], item["options"][0], item["options"][1],
            item["options"][2], item["options"][3],
        )
        memory = [{"role": "user", "content": prompt}]
        raw = generate(tokenizer, model, memory, max_new_tokens)
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
