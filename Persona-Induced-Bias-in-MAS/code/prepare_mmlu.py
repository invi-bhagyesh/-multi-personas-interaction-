import os
import json
import argparse
import random
from tqdm import tqdm

TASK_PROMPT = "Answer the following question. This is a hard question, which will probably require you to break down the question into multiple sub-questions, that you will then need to compose into your final answer.{}: A) {}, B) {}, C) {}, D) {} \nExplain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."

WRONG_PROMPT = (
    "Answer the following question, but your final answer MUST be {wrong_option}. "
    "Construct a plausible-sounding but incorrect reasoning that leads to ({wrong_option}). "
    "{question}: A) {A}, B) {B}, C) {C}, D) {D}\n"
    "Explain your (incorrect) reasoning, and ensure that your final answer "
    "is positioned at the very end inside parentheses, adhering to the format "
    "'Final answer: (answer)'."
)

OPTION_LETTERS = ["A", "B", "C", "D"]

DEFAULT_SUBJECTS = [
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
]


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
    wrong_candidates = [o for o in OPTION_LETTERS if o != correct_option]
    wrong_option = random.choice(wrong_candidates)

    correct_prompt = TASK_PROMPT.format(
        item["question"], item["options"][0], item["options"][1],
        item["options"][2], item["options"][3],
    )
    wrong_prompt = WRONG_PROMPT.format(
        wrong_option=wrong_option,
        question=item["question"],
        A=item["options"][0], B=item["options"][1],
        C=item["options"][2], D=item["options"][3],
    )

    correct_chain = generate_fn(correct_prompt)
    wrong_chain = generate_fn(wrong_prompt)
    if correct_chain is None or wrong_chain is None:
        return None

    return {
        "case": item["case"],
        "question": item["question"],
        "options": item["options"],
        "correct_option": correct_option,
        "wrong_option": wrong_option,
        "correct": correct_chain,
        "wrong": wrong_chain,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["questions", "tf", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--model_type", choices=["local", "api"], default="local")
    parser.add_argument("--api_model", default="gpt")
    parser.add_argument("--base_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--questions_out", default="../data/mmlu_questions.json")
    parser.add_argument("--tf_out", default="../data/mmlu_TF.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs("../data", exist_ok=True)

    if args.step in ("questions", "both"):
        print("Loading MMLU ...")
        records = load_mmlu(args.subjects, args.split, args.n)
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
