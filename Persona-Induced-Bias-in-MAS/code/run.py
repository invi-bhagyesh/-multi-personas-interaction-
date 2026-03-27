"""
Central runner — loads config.yaml and dispatches to experiment scripts.

Usage:
    python code/run.py                          # uses ../config.yaml
    python code/run.py --config ../config.yaml  # explicit path
    python code/run.py --config ../config.yaml --experiments accuracy
    python code/run.py --config ../config.yaml --experiments prepare_data cps
"""

import argparse
import os
import sys
import yaml

# Make sibling modules importable
sys.path.insert(0, os.path.dirname(__file__))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_personas(cfg: dict) -> list:
    chosen = cfg["personas"]["run"]
    if chosen is None:
        return cfg["personas"]["all"]
    return chosen


def resolve_data_paths(cfg: dict) -> tuple[str, str]:
    """Return (questions_path, tf_path) for the configured benchmark."""
    name = cfg["benchmark"]["name"]
    if name == "mmlu":
        return cfg["data"]["mmlu_questions"], cfg["data"]["mmlu_tf"]
    elif name == "gpqa":
        return cfg["data"]["gpqa_questions"], cfg["data"]["gpqa_tf"]
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def resolve_n(cfg: dict) -> int | None:
    name = cfg["benchmark"]["name"]
    return cfg["benchmark"][name].get("n")


# ── Sub-tasks ─────────────────────────────────────────────────────────────────

def run_prepare_data(cfg: dict):
    from prepare_mmlu import load_mmlu, build_tf_entry, main as mmlu_main
    import types, sys as _sys

    name = cfg["benchmark"]["name"]
    if name != "mmlu":
        print("[prepare_data] Only MMLU preparation is automated. Skipping.")
        return

    # Build a fake argparse Namespace so prepare_mmlu.main() is reusable
    import argparse, random
    from prepare_mmlu import load_mmlu, build_tf_entry, generate_local, generate_api, OPTION_LETTERS
    import json, os, random
    from tqdm import tqdm

    random.seed(42)
    bcfg   = cfg["benchmark"]["mmlu"]
    tfcfg  = cfg["tf_generation"]
    q_out  = cfg["data"]["mmlu_questions"]
    tf_out = cfg["data"]["mmlu_tf"]
    n      = bcfg.get("n")

    os.makedirs(os.path.dirname(q_out) if os.path.dirname(q_out) else ".", exist_ok=True)

    # Step 1 — questions
    if not os.path.exists(q_out):
        print("Downloading MMLU ...")
        records = load_mmlu(bcfg["subjects"], bcfg.get("split", "test"), n)
        with open(q_out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"  saved {len(records)} questions → {q_out}")
    else:
        print(f"  [skip] {q_out} already exists")

    # Step 2 — TF chains
    if not os.path.exists(tf_out):
        with open(q_out) as f:
            questions = json.load(f)

        if tfcfg["model_type"] == "local":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            base_id = cfg["model"]["base_id"]
            print(f"Loading {base_id} for TF chain generation ...")
            tokenizer = AutoTokenizer.from_pretrained(base_id)
            model = AutoModelForCausalLM.from_pretrained(
                base_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            model.eval()
            generate_fn = lambda p: generate_local(p, tokenizer, model)
        else:
            api_model = tfcfg["api_model"]
            generate_fn = lambda p: generate_api(p, api_model)

        tf_records = []
        for item in tqdm(questions, desc="TF chains"):
            entry = build_tf_entry(item, generate_fn)
            if entry:
                tf_records.append(entry)
            if len(tf_records) % 50 == 0 and tf_records:
                with open(tf_out, "w") as f:
                    json.dump(tf_records, f, indent=4)

        with open(tf_out, "w") as f:
            json.dump(tf_records, f, indent=4)
        print(f"  saved {len(tf_records)} TF entries → {tf_out}")
    else:
        print(f"  [skip] {tf_out} already exists")


def run_accuracy(cfg: dict):
    import json, os
    from tqdm import tqdm
    import accuracy_opencharacter as acc_mod

    questions_path, _ = resolve_data_paths(cfg)
    personas = resolve_personas(cfg)
    n = resolve_n(cfg)
    output = cfg["output"]["accuracy"]
    max_new_tokens = cfg["model"]["max_new_tokens"]

    with open(questions_path) as f:
        data = json.load(f)
    if n:
        data = data[:n]

    os.makedirs(os.path.dirname(output), exist_ok=True)

    all_results = []
    for persona in personas:
        result = acc_mod.run_persona(persona, data, max_new_tokens)
        all_results.append(result)
        with open(output, "w") as f:
            json.dump(all_results, f, indent=4)

    print("\n=== Accuracy Summary ===")
    for r in all_results:
        print(f"  {r['persona']:15s}  {r['accuracy']:.3f}")


def run_cps(cfg: dict):
    import json, os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import cps_opencharacter as cps_mod

    _, tf_path = resolve_data_paths(cfg)
    personas = resolve_personas(cfg)
    n = resolve_n(cfg)
    cps_cfg = cfg["experiments"]["cps"]
    modes = (
        ["trustworthiness", "insistence"]
        if cps_cfg["mode"] == "both"
        else [cps_cfg["mode"]]
    )
    output_dir = cfg["output"]["cps_dir"]
    max_new_tokens = cfg["model"]["max_new_tokens"]
    base_id = cfg["model"]["base_id"]

    with open(tf_path) as f:
        data = json.load(f)
    if n:
        data = data[:n]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {base_id}")
    tokenizer_d, model_d = cps_mod.load_base_model()

    for mode in modes:
        all_results = []
        print(f"\n{'='*50}\n  Mode: {mode.upper()}\n{'='*50}")
        for persona in personas:
            result = cps_mod.run_persona_experiment(
                persona, mode, data, tokenizer_d, model_d, max_new_tokens
            )
            all_results.append(result)
            with open(os.path.join(output_dir, f"{mode}.json"), "w") as f:
                json.dump(all_results, f, indent=4)

        metric = "trustworthiness" if mode == "trustworthiness" else "insistence"
        print(f"\n=== {mode.upper()} Summary ===")
        for r in all_results:
            print(f"  {r['persona']:15s}  {r[metric]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

EXPERIMENT_MAP = {
    "prepare_data": run_prepare_data,
    "accuracy":     run_accuracy,
    "cps":          run_cps,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENT_MAP.keys()),
        default=None,
        help="Override which experiments to run (default: use config.yaml flags)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Determine which experiments to run
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

    print(f"Config: {args.config}")
    print(f"Benchmark: {cfg['benchmark']['name']}")
    print(f"Personas: {resolve_personas(cfg)}")
    print(f"Running: {to_run}\n")

    for exp in to_run:
        print(f"\n{'#'*60}\n# {exp.upper()}\n{'#'*60}")
        EXPERIMENT_MAP[exp](cfg)


if __name__ == "__main__":
    main()
