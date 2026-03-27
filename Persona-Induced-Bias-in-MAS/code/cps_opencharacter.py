import os
import json
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from prompts import gpqa_task_prompt, gpqa_other_answer, gpqa_interaction_prompt
from model_utils import load_base, apply_adapter, unload_adapter, generate, REPO

LABEL_BASE = "another agent"
LABEL_PERSONA = "the other agent"


def run_one_case(data, initial, tokenizer, model_1, model_2,
                 label_1, label_2, max_new_tokens=512):
    if initial == "T":
        init_1, init_2 = data["correct"], data["wrong"]
    else:
        init_1, init_2 = data["wrong"], data["correct"]

    correct_opt = data["correct_option"]
    wrong_opt = data["wrong_option"]

    memory_1 = [
        {"role": "user", "content": gpqa_task_prompt(label_1, data)},
        {"role": "assistant", "content": init_1},
        {"role": "user", "content": gpqa_other_answer(label_1, init_2) + gpqa_interaction_prompt(label_1)},
    ]
    reply_1 = generate(tokenizer, model_1, memory_1, max_new_tokens)

    memory_2 = [
        {"role": "user", "content": gpqa_task_prompt(label_2, data)},
        {"role": "assistant", "content": init_2},
        {"role": "user", "content": gpqa_other_answer(label_2, init_1) + gpqa_interaction_prompt(label_2)},
    ]
    reply_2 = generate(tokenizer, model_2, memory_2, max_new_tokens)

    opt_1 = extract_option(reply_1)
    opt_2 = extract_option(reply_2)

    if initial == "T":
        agent1_conformed = (opt_1 == wrong_opt)
        agent2_conformed = (opt_2 == correct_opt)
    else:
        agent1_conformed = (opt_1 == correct_opt)
        agent2_conformed = (opt_2 == wrong_opt)

    return {
        "case": data["case"],
        "initial": initial,
        "correct_option": correct_opt,
        "wrong_option": wrong_opt,
        "option_1": opt_1,
        "option_2": opt_2,
        "agent1_conformed": agent1_conformed,
        "agent2_conformed": agent2_conformed,
    }


def run_table1_persona(persona, data, initials, tokenizer, base_model,
                       max_new_tokens=512, repo=REPO):
    persona_model = apply_adapter(base_model, persona, repo)

    cases = []
    for initial in initials:
        for item in tqdm(data, desc=f"{persona} (init={initial})"):
            result = run_one_case(
                item, initial, tokenizer,
                model_1=persona_model, model_2=base_model,
                label_1=LABEL_BASE, label_2=LABEL_PERSONA,
                max_new_tokens=max_new_tokens,
            )
            result["persona"] = persona
            cases.append(result)

    trustworthiness = sum(c["agent2_conformed"] for c in cases) / len(cases)
    insistence = 1.0 - sum(c["agent1_conformed"] for c in cases) / len(cases)
    print(f"  {persona:15s}  T={trustworthiness:.3f}  I={insistence:.3f}")

    unload_adapter(persona_model)
    return {
        "persona": persona,
        "trustworthiness": trustworthiness,
        "insistence": insistence,
        "n_cases": len(cases),
        "cases": cases,
    }


def run_table2_pair(p1, p2, data, initials, tokenizer, base_model,
                    max_new_tokens=512, repo=REPO):
    model_1 = apply_adapter(base_model, p1, repo)

    cases = []
    for initial in initials:
        for item in tqdm(data, desc=f"{p1} vs {p2} (init={initial})"):
            if initial == "T":
                init_1, init_2 = item["correct"], item["wrong"]
            else:
                init_1, init_2 = item["wrong"], item["correct"]

            memory_1 = [
                {"role": "user", "content": gpqa_task_prompt(p2, item)},
                {"role": "assistant", "content": init_1},
                {"role": "user", "content": gpqa_other_answer(p2, init_2) + gpqa_interaction_prompt(p2)},
            ]
            reply_1 = generate(tokenizer, model_1, memory_1, max_new_tokens)
            opt_1 = extract_option(reply_1)

            cases.append({
                "case": item["case"],
                "initial": initial,
                "correct_option": item["correct_option"],
                "wrong_option": item["wrong_option"],
                "option_1": opt_1,
                "option_2": None,
                "reply_1": reply_1,
                "reply_2": None,
            })

    unload_adapter(model_1)

    model_2 = apply_adapter(base_model, p2, repo)
    idx = 0
    for initial in initials:
        for item in tqdm(data, desc=f"{p2} vs {p1} (reply, init={initial})"):
            if initial == "T":
                init_1, init_2 = item["correct"], item["wrong"]
            else:
                init_1, init_2 = item["wrong"], item["correct"]

            memory_2 = [
                {"role": "user", "content": gpqa_task_prompt(p1, item)},
                {"role": "assistant", "content": init_2},
                {"role": "user", "content": gpqa_other_answer(p1, init_1) + gpqa_interaction_prompt(p1)},
            ]
            reply_2 = generate(tokenizer, model_2, memory_2, max_new_tokens)
            cases[idx]["option_2"] = extract_option(reply_2)
            cases[idx]["reply_2"] = reply_2
            idx += 1

    unload_adapter(model_2)

    conformed = 0
    for c in cases:
        if c["initial"] == "T" and c["option_1"] == c["wrong_option"]:
            conformed += 1
        elif c["initial"] == "F" and c["option_1"] == c["correct_option"]:
            conformed += 1

    conformity = conformed / len(cases) if cases else 0.0
    print(f"  C({p1}->{p2}) = {conformity:.3f}")

    return {
        "persona1": p1,
        "persona2": p2,
        "conformity": conformity,
        "n_cases": len(cases),
        "cases": cases,
    }
