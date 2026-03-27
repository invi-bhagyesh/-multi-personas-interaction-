import os
import json
import argparse
import random
from tqdm import tqdm

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

DEFAULT_SUBJECTS = [
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
]


def format_options(options):
    return ", ".join(f"{OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def format_option_list(options):
    return " or ".join(f"({OPTION_LETTERS[i]})" for i in range(len(options)))


def task_prompt(item):
    opts = format_options(item["options"])
    opt_list = format_option_list(item["options"])
    return (
        "Answer the following question. This is a hard question, which will probably "
        "require you to break down the question into multiple sub-questions, that you "
        f"will then need to compose into your final answer.{item['question']}: {opts}\n"
        f"Explain your answer, and ensure that your final answer {opt_list} is positioned "
        "at the very end of your output inside parentheses, adhering to the format "
        "'Final answer: (answer)'."
    )


def wrong_prompt(item, wrong_option):
    opts = format_options(item["options"])
    opt_list = format_option_list(item["options"])
    return (
        f"Answer the following question, but your final answer MUST be {wrong_option}. "
        f"Construct a plausible-sounding but incorrect reasoning that leads to ({wrong_option}). "
        f"{item['question']}: {opts}\n"
        "Explain your (incorrect) reasoning, and ensure that your final answer "
        "is positioned at the very end inside parentheses, adhering to the format "
        "'Final answer: (answer)'."
    )


# ── data loaders ─────────────────────────────────────────────────────────────

def load_mmlu(subjects, split="test", n=None):
    from datasets import load_dataset

    records = []
    case_id = 0
    for subject in subjects:
        print(f"  loading mmlu/{subject} ({split}) ...")
        ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
        for row in ds:
            choices = row["choices"]
            answer_idx = row["answer"]
            if len(choices) != 4:
                continue
            records.append({
                "case": case_id,
                "question": row["question"],
                "options": choices,
                "answer": OPTION_LETTERS[answer_idx],
                "subject": subject,
            })
            case_id += 1
            if n and case_id >= n:
                break
        if n and case_id >= n:
            break
    return records


def load_mmlu_pro(categories=None, split="test", n=None):
    from datasets import load_dataset

    print(f"  loading MMLU-Pro ({split}) ...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split, trust_remote_code=True)

    records = []
    case_id = 0
    for row in ds:
        if categories and row["category"] not in categories:
            continue
        records.append({
            "case": case_id,
            "question": row["question"],
            "options": row["options"],
            "answer": OPTION_LETTERS[row["answer_index"]],
            "category": row["category"],
        })
        case_id += 1
        if n and case_id >= n:
            break
    return records


# ── generation ───────────────────────────────────────────────────────────────

def generate_local(prompt, tokenizer, model, max_new_tokens=512):
    import torch
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def generate_api(prompt, api_model, max_tokens=512):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


def build_tf_entry(item, generate_fn):
    correct_option = item["answer"]
    num_options = len(item["options"])
    wrong_candidates = [OPTION_LETTERS[i] for i in range(num_options) if OPTION_LETTERS[i] != correct_option]
    wrong_opt = random.choice(wrong_candidates)

    correct_chain = generate_fn(task_prompt(item))
    wrong_chain = generate_fn(wrong_prompt(item, wrong_opt))
    if correct_chain is None or wrong_chain is None:
        return None

    return {
        "case": item["case"],
        "question": item["question"],
        "options": item["options"],
        "correct_option": correct_option,
        "wrong_option": wrong_opt,
        "correct": correct_chain,
        "wrong": wrong_chain,
    }


def build_tf_entries_parallel(questions, generate_fn, workers=10):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_tf_entry, item, generate_fn): i
                   for i, item in enumerate(questions)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [error] case {idx}: {e}")
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(questions)} done")

    return [r for r in results if r is not None]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["questions", "tf", "both"], default="both")
    parser.add_argument("--benchmark", choices=["mmlu", "mmlu_pro"], default="mmlu")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--model_type", choices=["local", "api"], default="local")
    parser.add_argument("--api_model", default="gpt-4o-mini")
    parser.add_argument("--base_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--questions_out", default=None)
    parser.add_argument("--tf_out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs("../data", exist_ok=True)

    if args.questions_out is None:
        args.questions_out = f"../data/{args.benchmark}_questions.json"
    if args.tf_out is None:
        args.tf_out = f"../data/{args.benchmark}_TF.json"

    if args.step in ("questions", "both"):
        if args.benchmark == "mmlu":
            print("Loading MMLU ...")
            records = load_mmlu(args.subjects, args.split, args.n)
        else:
            print("Loading MMLU-Pro ...")
            records = load_mmlu_pro(args.categories, args.split, args.n)
        with open(args.questions_out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {len(records)} questions -> {args.questions_out}")

    if args.step in ("tf", "both"):
        with open(args.questions_out) as f:
            questions = json.load(f)

        if args.model_type == "local":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"Loading {args.base_id} ...")
            tokenizer = AutoTokenizer.from_pretrained(args.base_id)
            model = AutoModelForCausalLM.from_pretrained(
                args.base_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            model.eval()
            gen_fn = lambda p: generate_local(p, tokenizer, model)
        else:
            gen_fn = lambda p: generate_api(p, args.api_model)

        if args.model_type == "api":
            tf_records = build_tf_entries_parallel(questions, gen_fn, workers=10)
        else:
            tf_records = []
            for item in tqdm(questions, desc="TF chains"):
                entry = build_tf_entry(item, gen_fn)
                if entry:
                    tf_records.append(entry)
                if len(tf_records) % 50 == 0 and tf_records:
                    with open(args.tf_out, "w") as f:
                        json.dump(tf_records, f, indent=4)

        with open(args.tf_out, "w") as f:
            json.dump(tf_records, f, indent=4)
        print(f"Saved {len(tf_records)} TF entries -> {args.tf_out}")


if __name__ == "__main__":
    main()
