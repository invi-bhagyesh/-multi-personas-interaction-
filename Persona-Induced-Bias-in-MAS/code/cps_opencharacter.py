import os
import json
import itertools
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import extract_option
from prompts import gpqa_task_prompt, gpqa_other_answer, gpqa_interaction_prompt, personas_prompt
from model_utils import load_base, apply_adapter, unload_adapter, generate, generate_batch, REPO, is_prompt_based, PERSONA_DESCRIPTIONS

LABEL_BASE = "another agent"
LABEL_PERSONA = "the other agent"

BATCH_SIZE = 32


# ── helpers ──────────────────────────────────────────────────────────────────

def _label(persona):
    """Return the label used to refer to an agent in prompts.

    In prompt-based mode, use a descriptive persona name (matching the
    original paper's approach). In OCT mode, use generic labels since
    the persona is embedded in the model weights, not named.
    """
    if is_prompt_based() and persona and persona != "base":
        desc = PERSONA_DESCRIPTIONS.get(persona, persona)
        # Strip leading article if present to avoid "a a ..." or "a an ..."
        for prefix in ("a ", "an "):
            if desc.startswith(prefix):
                desc = desc[len(prefix):]
                break
        return f"a {desc} agent"
    return None


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


def _check_conformed(initial, option, correct_opt, wrong_opt):
    """Check if an agent conformed to the other's position."""
    if initial == "T":
        return option == wrong_opt
    return option == correct_opt


def _compute_conformity(cases):
    """Compute conformity rate: fraction of cases where agent 1 switched."""
    if not cases:
        return 0.0
    conformed = sum(1 for c in cases
                    if _check_conformed(c["initial"], c["option_1"],
                                        c["correct_option"], c["wrong_option"]))
    return conformed / len(cases)


# ── Table 1: persona vs base (batched) ──────────────────────────────────────

def run_table1_persona(persona, data, initials, tokenizer, base_model,
                       max_new_tokens=512, repo=REPO):
    persona_model = apply_adapter(base_model, persona, repo)

    # In prompt-based mode, agents see each other's persona names (like original paper)
    # In OCT mode, agents see generic labels
    label_for_persona = _label(persona) or LABEL_PERSONA
    label_for_base = _label("base") or LABEL_BASE

    jobs = []
    memories_persona = []
    memories_base = []
    for initial in initials:
        for item in data:
            init_p, init_b = _inits(item, initial)
            num_opts = len(item["options"])
            memories_persona.append(
                _build_memory(label_for_base, item, init_p, init_b, num_opts))
            memories_base.append(
                _build_memory(label_for_persona, item, init_b, init_p, num_opts))
            jobs.append((item, initial))

    print(f"  {persona}: generating persona replies ({len(jobs)} cases)...")
    replies_persona = generate_batch(tokenizer, persona_model, memories_persona,
                                     max_new_tokens, BATCH_SIZE, persona=persona)
    unload_adapter(persona_model)

    print(f"  {persona}: generating base replies ({len(jobs)} cases)...")
    replies_base = generate_batch(tokenizer, base_model, memories_base,
                                  max_new_tokens, BATCH_SIZE, persona=None)

    cases = []
    for i, (item, initial) in enumerate(jobs):
        correct_opt = item["correct_option"]
        wrong_opt = item["wrong_option"]
        opt_1 = extract_option(replies_persona[i])
        opt_2 = extract_option(replies_base[i])

        cases.append({
            "case": item["case"],
            "initial": initial,
            "correct_option": correct_opt,
            "wrong_option": wrong_opt,
            "option_1": opt_1,
            "option_2": opt_2,
            "agent1_conformed": _check_conformed(initial, opt_1, correct_opt, wrong_opt),
            "agent2_conformed": _check_conformed(initial, opt_2, wrong_opt, correct_opt),
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


def run_baseline(data, initials, tokenizer, base_model, max_new_tokens=512):
    """Run base-vs-base (no persona) to get baseline T and I."""
    memories_1 = []
    memories_2 = []
    jobs = []
    for initial in initials:
        for item in data:
            init_1, init_2 = _inits(item, initial)
            num_opts = len(item["options"])
            memories_1.append(_build_memory(LABEL_BASE, item, init_1, init_2, num_opts))
            memories_2.append(_build_memory(LABEL_PERSONA, item, init_2, init_1, num_opts))
            jobs.append((item, initial))

    all_memories = memories_1 + memories_2
    print("  Running base-vs-base baseline...")
    all_replies = generate_batch(tokenizer, base_model, all_memories,
                                 max_new_tokens, persona=None)
    replies_1 = all_replies[:len(jobs)]
    replies_2 = all_replies[len(jobs):]

    cases = []
    for i, (item, initial) in enumerate(jobs):
        correct_opt = item["correct_option"]
        wrong_opt = item["wrong_option"]
        opt_1 = extract_option(replies_1[i])
        opt_2 = extract_option(replies_2[i])
        cases.append({
            "initial": initial,
            "correct_option": correct_opt,
            "wrong_option": wrong_opt,
            "option_1": opt_1,
            "option_2": opt_2,
            "agent1_conformed": _check_conformed(initial, opt_1, correct_opt, wrong_opt),
            "agent2_conformed": _check_conformed(initial, opt_2, wrong_opt, correct_opt),
        })

    t_base = sum(c["agent2_conformed"] for c in cases) / len(cases)
    i_base = 1.0 - sum(c["agent1_conformed"] for c in cases) / len(cases)
    print(f"  {'base':15s}  T={t_base:.3f}  I={i_base:.3f}")
    return t_base, i_base


# ── Table 2: persona pairs (adapter-batched) ────────────────────────────────

def run_table2_all(personas, data, initials, tokenizer, base_model,
                   max_new_tokens=512, repo=REPO):
    """Run all persona pairs with minimal adapter load/unloads.

    Strategy: for each persona p, load its adapter once, then generate
    all replies where p is the responder — covering every pair where
    p appears on either side.
    """
    pairs = list(itertools.product(personas, repeat=2))

    reply_store = {p: {} for p in personas}

    for persona in personas:
        model = apply_adapter(base_model, persona, repo)

        keys = []
        memories = []
        for partner in personas:
            partner_label = _label(partner) or partner
            for initial in initials:
                for idx, item in enumerate(data):
                    init_1, init_2 = _inits(item, initial)
                    num_opts = len(item["options"])

                    # As model_1 in pair (persona, partner)
                    memories.append(_build_memory(partner_label, item, init_1, init_2, num_opts))
                    keys.append(("as_p1", partner, idx, initial))

                    # As model_2 in pair (partner, persona)
                    memories.append(_build_memory(partner_label, item, init_2, init_1, num_opts))
                    keys.append(("as_p2", partner, idx, initial))

        print(f"  Table 2: generating {len(memories)} replies for {persona}...")
        replies = generate_batch(tokenizer, model, memories, max_new_tokens, BATCH_SIZE,
                                 persona=persona)
        unload_adapter(model)

        for key, reply in zip(keys, replies):
            role, partner, idx, initial = key
            reply_store[persona][(role, partner, idx, initial)] = reply

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

        conformity = _compute_conformity(cases)
        print(f"  C({p1}->{p2}) = {conformity:.3f}")

        results.append({
            "persona1": p1,
            "persona2": p2,
            "conformity": conformity,
            "n_cases": len(cases),
            "cases": cases,
        })

    return results
