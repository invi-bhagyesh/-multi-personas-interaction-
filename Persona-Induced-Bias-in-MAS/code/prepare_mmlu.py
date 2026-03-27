"""
Prepare MMLU data files for OpenCharacter experiments.

Produces two files (mirroring the GPQA originals):
  data/mmlu_questions.json   — analogous to gpqa_455.json
  data/mmlu_TF.json          — analogous to gpqa_TF.json
                               (correct + wrong reasoning chains pre-generated)

The TF file requires a model to generate reasoning chains.  By default it uses
the base Llama-3.1-8B-Instruct model locally.  Pass --api gpt/deepseek/etc to
use an API model instead (requires KEY module with api_key dict).

Usage:
    # Step 1: build mmlu_questions.json (no model needed)
    python prepare_mmlu.py --step questions --subjects high_school_biology --n 100

    # Step 2: generate TF reasoning chains (needs GPU or API)
    python prepare_mmlu.py --step tf --model_type local
    python prepare_mmlu.py --step tf --model_type api --api_model gpt

    # Both steps at once
    python prepare_mmlu.py --step both --subjects high_school_biology --n 100 --model_type local
"""

import os
import json
import argparse
import random
from tqdm import tqdm

TASK_PROMPT = (
    "Answer the following question. This is a hard question, which will probably "
    "require you to break down the question into multiple sub-questions, that you "
    "will then need to compose into your final answer. "
    "{question}: A) {A}, B) {B}, C) {C}, D) {D}\n"
    "Explain your answer, and ensure that your final answer (A) or (B) or (C) or (D) "
    "is positioned at the very end of your output inside parentheses, adhering to the "
    "format 'Final answer: (answer)'."
)

WRONG_PROMPT = (
    "Answer the following question, but your final answer MUST be {wrong_option}. "
    "Construct a plausible-sounding but incorrect reasoning that leads to ({wrong_option}). "
    "{question}: A) {A}, B) {B}, C) {C}, D) {D}\n"
    "Explain your (incorrect) reasoning, and ensure that your final answer "
    "is positioned at the very end inside parentheses, adhering to the format "
    "'Final answer: (answer)'."
)

OPTION_LETTERS = ["A", "B", "C", "D"]


# ── MMLU download ─────────────────────────────────────────────────────────────

def load_mmlu(subjects: list, split: str = "test", n: int = None) -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Install the datasets library first:\n  pip install datasets"
        )

    records = []
    case_id = 0
    for subject in subjects:
        print(f"  loading mmlu/{subject} ({split}) ...")
        ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
        for row in ds:
            choices = row["choices"]          # list of 4 strings
            answer_idx = row["answer"]        # 0-3 int
            if len(choices) != 4:
                continue
            records.append({
                "case":    case_id,
                "question": row["question"],
                "options":  choices,
                "answer":   OPTION_LETTERS[answer_idx],
                "subject":  subject,
            })
            case_id += 1
            if n and case_id >= n:
                break
        if n and case_id >= n:
            break
    return records


# ── Reasoning chain generation ────────────────────────────────────────────────

def generate_local(prompt: str, tokenizer, model, max_new_tokens: int = 512) -> str:
    import torch
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_api(prompt: str, api_model: str, max_tokens: int = 512) -> str:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from client import send_client
    memory = [{"role": "user", "content": prompt}]
    return send_client(api_model, memory, max_tokens=max_tokens, temperature=0.0)


def build_tf_entry(item: dict, generate_fn) -> dict | None:
    """Generate correct and wrong reasoning chains for one question."""
    correct_option = item["answer"]
    # Pick a wrong option (randomly, different from correct)
    wrong_candidates = [o for o in OPTION_LETTERS if o != correct_option]
    wrong_option = random.choice(wrong_candidates)

    correct_prompt = TASK_PROMPT.format(
        question=item["question"],
        A=item["options"][0], B=item["options"][1],
        C=item["options"][2], D=item["options"][3],
    )
    wrong_prompt = WRONG_PROMPT.format(
        wrong_option=wrong_option,
        question=item["question"],
        A=item["options"][0], B=item["options"][1],
        C=item["options"][2], D=item["options"][3],
    )

    correct_chain = generate_fn(correct_prompt)
    wrong_chain   = generate_fn(wrong_prompt)

    if correct_chain is None or wrong_chain is None:
        return None

    return {
        "case":           item["case"],
        "question":       item["question"],
        "options":        item["options"],
        "correct_option": correct_option,
        "wrong_option":   wrong_option,
        "correct":        correct_chain,
        "wrong":          wrong_chain,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

DEFAULT_SUBJECTS = [
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["questions", "tf", "both"],
                        default="both")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None,
                        help="Max total questions (default: all)")
    parser.add_argument("--model_type", choices=["local", "api"], default="local",
                        help="How to generate TF reasoning chains")
    parser.add_argument("--api_model", default="gpt",
                        help="API model name when --model_type api")
    parser.add_argument("--base_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace base model for local generation")
    parser.add_argument("--questions_out", default="../data/mmlu_questions.json")
    parser.add_argument("--tf_out",        default="../data/mmlu_TF.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs("../data", exist_ok=True)

    # ── Step 1: build questions file ──────────────────────────────────────
    if args.step in ("questions", "both"):
        print("Loading MMLU ...")
        records = load_mmlu(args.subjects, args.split, args.n)
        with open(args.questions_out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {len(records)} questions → {args.questions_out}")

    # ── Step 2: generate TF reasoning chains ─────────────────────────────
    if args.step in ("tf", "both"):
        with open(args.questions_out) as f:
            questions = json.load(f)

        if args.model_type == "local":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"Loading {args.base_id} for chain generation ...")
            tokenizer = AutoTokenizer.from_pretrained(args.base_id)
            model = AutoModelForCausalLM.from_pretrained(
                args.base_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            model.eval()
            generate_fn = lambda p: generate_local(p, tokenizer, model)
        else:
            generate_fn = lambda p: generate_api(p, args.api_model)

        tf_records = []
        for item in tqdm(questions, desc="generating TF chains"):
            entry = build_tf_entry(item, generate_fn)
            if entry:
                tf_records.append(entry)
            # checkpoint every 50
            if len(tf_records) % 50 == 0:
                with open(args.tf_out, "w") as f:
                    json.dump(tf_records, f, indent=4)

        with open(args.tf_out, "w") as f:
            json.dump(tf_records, f, indent=4)
        print(f"Saved {len(tf_records)} TF entries → {args.tf_out}")


if __name__ == "__main__":
    main()
