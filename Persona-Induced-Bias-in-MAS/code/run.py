import argparse
import itertools
import json
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(SCRIPT_DIR, rel_path))


def get_personas(cfg):
    chosen = cfg["personas"].get("run")
    if chosen is None:
        return cfg["personas"]["all"]
    return chosen


def get_data_paths(cfg):
    name = cfg["benchmark"]["name"]
    if name == "mmlu":
        return resolve_path(cfg["data"]["mmlu_questions"]), resolve_path(cfg["data"]["mmlu_tf"])
    elif name == "gpqa":
        return resolve_path(cfg["data"]["gpqa_questions"]), resolve_path(cfg["data"]["gpqa_tf"])
    raise ValueError(f"Unknown benchmark: {name}")


def get_n(cfg):
    name = cfg["benchmark"]["name"]
    return cfg["benchmark"][name].get("n")


# ── prepare_data ─────────────────────────────────────────────────────────────

def run_prepare_data(cfg):
    import random
    from tqdm import tqdm
    from prepare_mmlu import load_mmlu, build_tf_entry, build_tf_entries_parallel, generate_local, generate_api

    name = cfg["benchmark"]["name"]
    if name != "mmlu":
        print("[prepare_data] Only MMLU prep is automated. Skipping.")
        return

    random.seed(42)
    bcfg = cfg["benchmark"]["mmlu"]
    tfcfg = cfg["tf_generation"]
    q_out = resolve_path(cfg["data"]["mmlu_questions"])
    tf_out = resolve_path(cfg["data"]["mmlu_tf"])
    n = bcfg.get("n")

    os.makedirs(os.path.dirname(q_out) or ".", exist_ok=True)

    if not os.path.exists(q_out):
        print("Downloading MMLU ...")
        records = load_mmlu(bcfg["subjects"], bcfg.get("split", "test"), n)
        with open(q_out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"  saved {len(records)} questions -> {q_out}")
    else:
        print(f"  [skip] {q_out} exists")

    if not os.path.exists(tf_out):
        with open(q_out) as f:
            questions = json.load(f)

        if tfcfg["model_type"] == "local":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            base_id = cfg["model"]["base_id"]
            print(f"Loading {base_id} for TF generation ...")
            tok = AutoTokenizer.from_pretrained(base_id)
            mdl = AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", torch_dtype=torch.bfloat16)
            mdl.eval()
            gen_fn = lambda p: generate_local(p, tok, mdl)
        else:
            gen_fn = lambda p: generate_api(p, tfcfg["api_model"])

        if tfcfg["model_type"] == "api":
            tf_records = build_tf_entries_parallel(questions, gen_fn, workers=10)
        else:
            tf_records = []
            for item in tqdm(questions, desc="TF chains"):
                entry = build_tf_entry(item, gen_fn)
                if entry:
                    tf_records.append(entry)
                if len(tf_records) % 50 == 0 and tf_records:
                    with open(tf_out, "w") as f:
                        json.dump(tf_records, f, indent=4)
        with open(tf_out, "w") as f:
            json.dump(tf_records, f, indent=4)
        print(f"  saved {len(tf_records)} TF entries -> {tf_out}")
    else:
        print(f"  [skip] {tf_out} exists")


# ── accuracy ─────────────────────────────────────────────────────────────────

def run_accuracy(cfg):
    import accuracy_opencharacter as acc

    personas = get_personas(cfg)
    q_path, _ = get_data_paths(cfg)
    n = get_n(cfg)
    out_path = resolve_path(cfg["output"]["accuracy"])

    with open(q_path) as f:
        data = json.load(f)
    if n:
        data = data[:n]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tokenizer, base_model = acc.load_base(cfg["model"]["base_id"])

    results = []
    for persona in personas:
        r = acc.run_persona(persona, data, tokenizer, base_model,
                            max_new_tokens=cfg["model"]["max_new_tokens"],
                            repo=cfg["model"]["repo"])
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

    print("\n=== Accuracy ===")
    for r in results:
        print(f"  {r['persona']:15s}  {r['accuracy']:.3f}")


# ── cps ──────────────────────────────────────────────────────────────────────

def run_cps(cfg):
    import cps_opencharacter as cps

    personas = get_personas(cfg)
    _, tf_path = get_data_paths(cfg)
    n = get_n(cfg)
    mnt = cfg["model"]["max_new_tokens"]
    repo = cfg["model"]["repo"]
    out_dir = resolve_path(cfg["output"]["cps_dir"])

    with open(tf_path) as f:
        data = json.load(f)
    if n:
        data = data[:n]

    os.makedirs(out_dir, exist_ok=True)
    tokenizer, base_model = cps.load_base(cfg["model"]["base_id"])
    initials = ["T", "F"]

    # Table 1: each persona vs base
    table1 = []
    for persona in personas:
        r = cps.run_table1_persona(persona, data, initials, tokenizer, base_model,
                                   max_new_tokens=mnt, repo=repo)
        table1.append(r)
        with open(os.path.join(out_dir, "table1.json"), "w") as f:
            json.dump(table1, f, indent=4, default=str)

    print(f"\n  {'Persona':15s}  {'Trust':>7s}  {'Insist':>7s}")
    for r in table1:
        print(f"  {r['persona']:15s}  {r['trustworthiness']:7.3f}  {r['insistence']:7.3f}")

    # Table 2: all persona pairs
    active = [p for p in personas if p != "base"]
    pairs = list(itertools.product(active, repeat=2))

    table2 = []
    for p1, p2 in pairs:
        r = cps.run_table2_pair(p1, p2, data, initials, tokenizer, base_model,
                                max_new_tokens=mnt, repo=repo)
        table2.append(r)
        with open(os.path.join(out_dir, "table2.json"), "w") as f:
            json.dump(table2, f, indent=4, default=str)

    conf = {(r["persona1"], r["persona2"]): r["conformity"] for r in table2}
    print(f"\n  {'C(row->col)':15s}" + "".join(f"  {p[:7]:>7s}" for p in active))
    all_scores, intra_scores = [], []
    for p1 in active:
        row = f"  {p1:15s}"
        for p2 in active:
            c = conf.get((p1, p2), float('nan'))
            all_scores.append(c)
            if p1 == p2:
                intra_scores.append(c)
                row += f"  [{c:5.3f}]"
            else:
                row += f"  {c:7.3f}"
        print(row)

    avg_all = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_intra = sum(intra_scores) / len(intra_scores) if intra_scores else 0
    print(f"\n  All: {avg_all:.3f}   Intra: {avg_intra:.3f}   "
          f"Delta: {avg_intra - avg_all:+.3f}")


# ── main ─────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "prepare_data": run_prepare_data,
    "accuracy": run_accuracy,
    "cps": run_cps,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_DIR, "config.yaml"))
    parser.add_argument("--experiments", nargs="+",
                        choices=list(EXPERIMENTS.keys()), default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.experiments:
        to_run = args.experiments
    else:
        to_run = []
        if cfg["experiments"].get("prepare_data"):
            to_run.append("prepare_data")
        if cfg["experiments"].get("accuracy"):
            to_run.append("accuracy")
        if cfg["experiments"].get("cps", {}).get("run"):
            to_run.append("cps")

    print(f"Model:     {cfg['model']['base_id']}")
    print(f"Adapters:  {cfg['model']['repo']}")
    print(f"Benchmark: {cfg['benchmark']['name']}")
    print(f"Personas:  {get_personas(cfg)}")
    print(f"Running:   {to_run}\n")

    for exp in to_run:
        EXPERIMENTS[exp](cfg)


if __name__ == "__main__":
    main()
