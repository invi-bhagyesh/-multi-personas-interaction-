import os
import json
import itertools
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from prompts import gpqa_task_prompt, gpqa_other_answer, gpqa_interaction_prompt
from model_utils import load_base, apply_adapter, unload_adapter, generate, generate_batch, REPO

LABEL_BASE = "another agent"
LABEL_PERSONA = "the other agent"

BATCH_SIZE = 16


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_memory(label, data, init_self, init_other, num_options):
    return [
        {"role": "user", "content": gpqa_task_prompt(label, data)},
        {"role": "assistant", "content": init_self},
        {"role": "user", "content": gpqa_other_answer(label, init_other)
                                    + gpqa_interaction_prompt(label, num_options)},
    ]


def _inits(item, initial):
    if initial == "T":
        return item["correct"], item["wrong"]
    return item["wrong"], item["correct"]


# ── single-case (kept for base-vs-base baseline) ────────────────────────────

def run_one_case(data, initial, tokenizer, model_1, model_2,
                 label_1, label_2, max_new_tokens=512):
    init_1, init_2 = _inits(data, initial)
    correct_opt = data["correct_option"]
    wrong_opt = data["wrong_option"]
    num_opts = len(data["options"])

    reply_1 = generate(tokenizer, model_1,
                       _build_memory(label_1, data, init_1, init_2, num_opts),
                       max_new_tokens)
    reply_2 = generate(tokenizer, model_2,
                       _build_memory(label_2, data, init_2, init_1, num_opts),
                       max_new_tokens)

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


# ── Table 1: persona vs base (batched) ──────────────────────────────────────

def run_table1_persona(persona, data, initials, tokenizer, base_model,
                       max_new_tokens=512, repo=REPO):
    persona_model = apply_adapter(base_model, persona, repo)

    # Build all prompts for both models
    jobs = []  # (item, initial, correct_opt, wrong_opt)
    memories_persona = []
    memories_base = []
    for initial in initials:
        for item in data:
            init_p, init_b = _inits(item, initial)
            num_opts = len(item["options"])
            memories_persona.append(
                _build_memory(LABEL_BASE, item, init_p, init_b, num_opts))
            memories_base.append(
                _build_memory(LABEL_PERSONA, item, init_b, init_p, num_opts))
            jobs.append((item, initial))

    # Batch generate persona side
    print(f"  {persona}: generating persona replies ({len(jobs)} cases)...")
    replies_persona = generate_batch(tokenizer, persona_model, memories_persona,
                                     max_new_tokens, BATCH_SIZE)
    unload_adapter(persona_model)

    # Batch generate base side
    print(f"  {persona}: generating base replies ({len(jobs)} cases)...")
    replies_base = generate_batch(tokenizer, base_model, memories_base,
                                  max_new_tokens, BATCH_SIZE)

    # Assemble results
    cases = []
    for i, (item, initial) in enumerate(jobs):
        correct_opt = item["correct_option"]
        wrong_opt = item["wrong_option"]
        opt_1 = extract_option(replies_persona[i])
        opt_2 = extract_option(replies_base[i])

        if initial == "T":
            agent1_conformed = (opt_1 == wrong_opt)
            agent2_conformed = (opt_2 == correct_opt)
        else:
            agent1_conformed = (opt_1 == correct_opt)
            agent2_conformed = (opt_2 == wrong_opt)

        cases.append({
            "case": item["case"],
            "initial": initial,
            "correct_option": correct_opt,
            "wrong_option": wrong_opt,
            "option_1": opt_1,
            "option_2": opt_2,
            "agent1_conformed": agent1_conformed,
            "agent2_conformed": agent2_conformed,
            "persona": persona,
        })

    trustworthiness = sum(c["agent2_conformed"] for c in cases) / len(cases)
    insistence = 1.0 - sum(c["agent1_conformed"] for c in cases) / len(cases)
    print(f"  {persona:15s}  T={trustworthiness:.3f}  I={insistence:.3f}")

    return {
        "persona": persona,
        "trustworthiness": trustworthiness,
        "insistence": insistence,
        "n_cases": len(cases),
        "cases": cases,
    }


# ── Table 2: persona pairs (adapter-batched) ────────────────────────────────

def run_table2_all(personas, data, initials, tokenizer, base_model,
                   max_new_tokens=512, repo=REPO):
    """Run all persona pairs with minimal adapter load/unloads.

    Strategy: for each persona p, load its adapter once, then generate
    all replies where p is the responder — covering every pair where
    p appears on either side.
    """
    pairs = list(itertools.product(personas, repeat=2))

    # For each pair (p1, p2), we need:
    #   - p1's reply (seeing p2's init) → p1 is model_1
    #   - p2's reply (seeing p1's init) → p2 is model_2
    # Group by which adapter needs to be loaded.

    # Pre-build all jobs indexed by (p1, p2, item_idx, initial)
    # reply_store[persona][(partner, item_idx, initial)] = reply
    reply_store = {p: {} for p in personas}

    for persona in personas:
        model = apply_adapter(base_model, persona, repo)

        # Collect all memories where this persona needs to respond
        keys = []
        memories = []
        for partner in personas:
            for initial in initials:
                for idx, item in enumerate(data):
                    # As model_1 in pair (persona, partner)
                    init_1, init_2 = _inits(item, initial)
                    num_opts = len(item["options"])
                    mem = _build_memory(partner, item, init_1, init_2, num_opts)
                    memories.append(mem)
                    keys.append(("as_p1", partner, idx, initial))

                    # As model_2 in pair (partner, persona)
                    init_1, init_2 = _inits(item, initial)
                    mem = _build_memory(partner, item, init_2, init_1, num_opts)
                    memories.append(mem)
                    keys.append(("as_p2", partner, idx, initial))

        print(f"  Table 2: generating {len(memories)} replies for {persona}...")
        replies = generate_batch(tokenizer, model, memories, max_new_tokens, BATCH_SIZE)
        unload_adapter(model)

        for key, reply in zip(keys, replies):
            role, partner, idx, initial = key
            reply_store[persona][(role, partner, idx, initial)] = reply

    # Assemble results per pair
    results = []
    for p1, p2 in pairs:
        cases = []
        for initial in initials:
            for idx, item in enumerate(data):
                correct_opt = item["correct_option"]
                wrong_opt = item["wrong_option"]

                reply_1 = reply_store[p1][("as_p1", p2, idx, initial)]
                reply_2 = reply_store[p2][("as_p2", p1, idx, initial)]
                opt_1 = extract_option(reply_1)
                opt_2 = extract_option(reply_2)

                cases.append({
                    "case": item["case"],
                    "initial": initial,
                    "correct_option": correct_opt,
                    "wrong_option": wrong_opt,
                    "option_1": opt_1,
                    "option_2": opt_2,
                    "reply_1": reply_1,
                    "reply_2": reply_2,
                })

        conformed = 0
        for c in cases:
            if c["initial"] == "T" and c["option_1"] == c["wrong_option"]:
                conformed += 1
            elif c["initial"] == "F" and c["option_1"] == c["correct_option"]:
                conformed += 1

        conformity = conformed / len(cases) if cases else 0.0
        print(f"  C({p1}->{p2}) = {conformity:.3f}")

        results.append({
            "persona1": p1,
            "persona2": p2,
            "conformity": conformity,
            "n_cases": len(cases),
            "cases": cases,
        })

    return results


# Keep old single-pair function for backwards compatibility
def run_table2_pair(p1, p2, data, initials, tokenizer, base_model,
                    max_new_tokens=512, repo=REPO):
    model_1 = apply_adapter(base_model, p1, repo)

    cases = []
    memories = []
    for initial in initials:
        for item in data:
            init_1, init_2 = _inits(item, initial)
            num_opts = len(item["options"])
            memories.append(_build_memory(p2, item, init_1, init_2, num_opts))
            cases.append({
                "case": item["case"],
                "initial": initial,
                "correct_option": item["correct_option"],
                "wrong_option": item["wrong_option"],
                "option_1": None,
                "option_2": None,
                "reply_1": None,
                "reply_2": None,
            })

    replies_1 = generate_batch(tokenizer, model_1, memories, max_new_tokens, BATCH_SIZE)
    for i, reply in enumerate(replies_1):
        cases[i]["reply_1"] = reply
        cases[i]["option_1"] = extract_option(reply)

    unload_adapter(model_1)

    model_2 = apply_adapter(base_model, p2, repo)
    memories_2 = []
    for initial in initials:
        for item in data:
            init_1, init_2 = _inits(item, initial)
            num_opts = len(item["options"])
            memories_2.append(_build_memory(p1, item, init_2, init_1, num_opts))

    replies_2 = generate_batch(tokenizer, model_2, memories_2, max_new_tokens, BATCH_SIZE)
    for i, reply in enumerate(replies_2):
        cases[i]["reply_2"] = reply
        cases[i]["option_2"] = extract_option(reply)

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
