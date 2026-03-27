"""
Pairwise debate experiments (OpenCharacter).

Supports two setups matching the paper:

  Table 1 (§4): One persona-LoRA agent vs one base-model agent, 1 round.
    - Both T and F initial conditions, aggregated.
    - Trustworthiness T(p) = rate at which base conforms to persona.
    - Insistence I(p) = 1 - rate at which persona conforms to base.

  Table 2 (§5): Two persona-LoRA agents interact, 1 round.
    - All ordered pairs including same-persona (diagonal).
    - Conformity rate C(p1→p2) for each pair.
    - "Intra" (diagonal) vs "All" comparison for in-group favoritism.

Key design: mirrors the original cps.py — each case has pre-generated
correct + wrong reasoning chains. One agent starts with correct, the other
with wrong (initial=T) or vice versa (initial=F). After 1 round of seeing
each other's answer, both agents produce an updated answer.
"""

import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from prompts import gpqa_task_prompt, gpqa_other_answer, gpqa_interaction_prompt

REPO    = "maius/qwen-2.5-7b-it-personas"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

# How agents refer to each other in prompts (no demographic label needed)
LABEL_BASE    = "another agent"
LABEL_PERSONA = "the other agent"


def load_base(base_id=BASE_ID):
    print(f"[loading] base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def apply_adapter(base_model, persona, repo=REPO):
    print(f"[loading] adapter: {persona}")
    model = PeftModel.from_pretrained(base_model, repo, subfolder=persona)
    model.eval()
    return model


def unload_adapter(model):
    if isinstance(model, PeftModel):
        model = model.unload()
        torch.cuda.empty_cache()
    return model


def generate(tokenizer, model, memory, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        memory, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def run_one_case(data, initial, tokenizer, model_1, model_2,
                 label_1, label_2, max_new_tokens=512):
    """
    Run a single 1-round debate between two agents on one question.

    Args:
        data: dict with 'correct', 'wrong', 'correct_option', 'wrong_option', etc.
        initial: 'T' or 'F'.
            T → agent1 starts correct, agent2 starts wrong.
            F → agent1 starts wrong, agent2 starts correct.
        model_1, model_2: the two models (base or PeftModel).
        label_1, label_2: how each agent refers to the other in prompts.

    Returns:
        dict with both agents' final options and whether each conformed.
    """
    if initial == "T":
        init_1, init_2 = data["correct"], data["wrong"]
    else:
        init_1, init_2 = data["wrong"], data["correct"]

    correct_opt = data["correct_option"]
    wrong_opt   = data["wrong_option"]

    # Agent 1 sees agent 2's initial answer, produces updated answer
    memory_1 = [
        {"role": "user",      "content": gpqa_task_prompt(label_1, data)},
        {"role": "assistant", "content": init_1},
        {"role": "user",      "content":
            gpqa_other_answer(label_1, init_2) +
            gpqa_interaction_prompt(label_1)},
    ]
    reply_1 = generate(tokenizer, model_1, memory_1, max_new_tokens)

    # Agent 2 sees agent 1's initial answer, produces updated answer
    memory_2 = [
        {"role": "user",      "content": gpqa_task_prompt(label_2, data)},
        {"role": "assistant", "content": init_2},
        {"role": "user",      "content":
            gpqa_other_answer(label_2, init_1) +
            gpqa_interaction_prompt(label_2)},
    ]
    reply_2 = generate(tokenizer, model_2, memory_2, max_new_tokens)

    opt_1 = extract_option(reply_1)
    opt_2 = extract_option(reply_2)

    # "Conformed" = agent switched to the OTHER agent's initial answer.
    # Agent 1 started with init_1; if initial=T, that's correct. Conforming
    # means agent 1 now holds wrong_opt (adopted agent 2's position).
    if initial == "T":
        agent1_conformed = (opt_1 == wrong_opt)
        agent2_conformed = (opt_2 == correct_opt)
    else:
        agent1_conformed = (opt_1 == correct_opt)
        agent2_conformed = (opt_2 == wrong_opt)

    return {
        "case":             data["case"],
        "initial":          initial,
        "correct_option":   correct_opt,
        "wrong_option":     wrong_opt,
        "option_1":         opt_1,
        "option_2":         opt_2,
        "agent1_conformed": agent1_conformed,
        "agent2_conformed": agent2_conformed,
    }


# ── Table 1: persona vs base ─────────────────────────────────────────────────

def run_table1_persona(persona, data, initials, tokenizer, base_model,
                       max_new_tokens=512, repo=REPO):
    """
    Run Table 1 for one persona: persona-LoRA agent vs base agent.

    For each question and each initial condition:
      - Agent 1 = persona model, Agent 2 = base model.
      - Trustworthiness: did the base agent (agent2) conform to persona?
      - Insistence: did the persona agent (agent1) resist conforming to base?

    Returns dict with trustworthiness, insistence scores, and per-case details.
    """
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

    # Trustworthiness = rate at which base (agent2) conformed to persona
    trustworthiness = sum(c["agent2_conformed"] for c in cases) / len(cases)
    # Insistence = 1 - rate at which persona (agent1) conformed to base
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


# ── Table 2: persona vs persona ──────────────────────────────────────────────

def run_table2_pair(p1, p2, data, initials, tokenizer, base_model,
                    max_new_tokens=512, repo=REPO):
    """
    Run Table 2 for one ordered pair: persona1 vs persona2.

    Both agents have LoRA adapters. Measures C(p1→p2) = rate at which
    agent1 (p1) conforms to agent2 (p2).

    Returns dict with conformity rate and per-case details.
    """
    model_1 = apply_adapter(base_model, p1, repo)

    # For p2, we need a separate PeftModel. But we can't stack two adapters
    # on the same base simultaneously. We'll generate agent1's reply first
    # with model_1, then swap to model_2 for agent2's reply.
    # This means we can't use run_one_case directly — we need sequential generation.

    cases = []
    for initial in initials:
        for item in tqdm(data, desc=f"{p1} vs {p2} (init={initial})"):
            if initial == "T":
                init_1, init_2 = item["correct"], item["wrong"]
            else:
                init_1, init_2 = item["wrong"], item["correct"]

            correct_opt = item["correct_option"]
            wrong_opt   = item["wrong_option"]

            # Agent 1 (p1) generates reply
            memory_1 = [
                {"role": "user",      "content": gpqa_task_prompt(p2, item)},
                {"role": "assistant", "content": init_1},
                {"role": "user",      "content":
                    gpqa_other_answer(p2, init_2) +
                    gpqa_interaction_prompt(p2)},
            ]
            reply_1 = generate(tokenizer, model_1, memory_1, max_new_tokens)
            opt_1 = extract_option(reply_1)

            cases.append({
                "case":           item["case"],
                "initial":        initial,
                "correct_option": correct_opt,
                "wrong_option":   wrong_opt,
                "option_1":       opt_1,
                "option_2":       None,   # filled in agent2 pass
                "reply_1":        reply_1,
                "reply_2":        None,
            })

    # Done with agent1 — swap adapter
    unload_adapter(model_1)

    # Agent 2 (p2) pass
    model_2 = apply_adapter(base_model, p2, repo)

    idx = 0
    for initial in initials:
        for item in tqdm(data, desc=f"{p2} vs {p1} (reply, init={initial})"):
            if initial == "T":
                init_1, init_2 = item["correct"], item["wrong"]
            else:
                init_1, init_2 = item["wrong"], item["correct"]

            memory_2 = [
                {"role": "user",      "content": gpqa_task_prompt(p1, item)},
                {"role": "assistant", "content": init_2},
                {"role": "user",      "content":
                    gpqa_other_answer(p1, init_1) +
                    gpqa_interaction_prompt(p1)},
            ]
            reply_2 = generate(tokenizer, model_2, memory_2, max_new_tokens)
            opt_2 = extract_option(reply_2)

            cases[idx]["option_2"] = opt_2
            cases[idx]["reply_2"]  = reply_2
            idx += 1

    unload_adapter(model_2)

    # Compute conformity: C(p1→p2) = rate at which p1 adopted p2's initial answer
    conformed_count = 0
    for c in cases:
        if c["initial"] == "T":
            # p1 started correct; conforming = p1 switched to wrong
            if c["option_1"] == c["wrong_option"]:
                conformed_count += 1
        else:
            # p1 started wrong; conforming = p1 switched to correct
            if c["option_1"] == c["correct_option"]:
                conformed_count += 1

    conformity = conformed_count / len(cases) if cases else 0.0
    print(f"  C({p1}→{p2}) = {conformity:.3f}")

    return {
        "persona1": p1,
        "persona2": p2,
        "conformity": conformity,
        "n_cases": len(cases),
        "cases": cases,
    }
