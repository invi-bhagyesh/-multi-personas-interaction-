"""
Evaluate results and produce Tables 1 & 2 matching the paper format.

Usage:
    python evaluation/eval_opencharacter.py
    python evaluation/eval_opencharacter.py --table1 results/table1/table1.json
    python evaluation/eval_opencharacter.py --table2_dir results/table2/
"""

import json
import os
import argparse
import glob
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def eval_table1(path):
    data = load_json(path)

    print("\n" + "=" * 60)
    print("  TABLE 1 — Persona-induced variations in T and I")
    print("  (Trustworthiness & Insistence, persona vs base, 1 round)")
    print("=" * 60)

    print(f"\n  {'Persona':15s}  {'Trust':>7s}  {'Insist':>7s}")
    print(f"  {'-'*33}")

    t_scores, i_scores = [], []
    for entry in data:
        t = entry["trustworthiness"]
        i = entry["insistence"]
        t_scores.append(t)
        i_scores.append(i)
        print(f"  {entry['persona']:15s}  {t:7.1%}  {i:7.1%}")

    print(f"  {'-'*33}")
    if len(t_scores) > 1:
        print(f"  {'Δmax-min':15s}  {(max(t_scores)-min(t_scores)):7.1%}  "
              f"{(max(i_scores)-min(i_scores)):7.1%}")
        print(f"  {'Δavg':15s}  {np.std(t_scores):7.1%}  "
              f"{np.std(i_scores):7.1%}")
    print(f"  {'mean':15s}  {np.mean(t_scores):7.1%}  "
          f"{np.mean(i_scores):7.1%}")


def eval_table2(table2_dir):
    # Load all pair results
    combined = os.path.join(table2_dir, "table2_all.json")
    if os.path.exists(combined):
        results = load_json(combined)
    else:
        results = []
        for f in sorted(glob.glob(os.path.join(table2_dir, "*.json"))):
            if "table2_all" in f:
                continue
            results.append(load_json(f))

    if not results:
        print(f"[skip] No table2 results in {table2_dir}")
        return

    # Build conformity dict
    conf = {}
    for r in results:
        conf[(r["persona1"], r["persona2"])] = r["conformity"]

    personas = sorted(set(r["persona1"] for r in results))

    print("\n" + "=" * 60)
    print("  TABLE 2 — Conformity rates C(p1→p2)")
    print("  (both agents have persona LoRA, 1 round)")
    print("=" * 60)

    # Header
    col_w = max(8, max(len(p) for p in personas) + 1)
    header = f"\n  {'C(row→col)':15s}" + "".join(f"  {p:>{col_w}s}" for p in personas)
    print(header)
    print(f"  {'-' * (15 + (col_w + 2) * len(personas))}")

    all_scores = []
    intra_scores = []
    for p1 in personas:
        row = f"  {p1:15s}"
        for p2 in personas:
            c = conf.get((p1, p2), float('nan'))
            all_scores.append(c)
            if p1 == p2:
                intra_scores.append(c)
                row += f"  [{c:{col_w-2}.1%}]"
            else:
                row += f"  {c:{col_w}.1%}"
        print(row)

    avg_all   = np.mean(all_scores)
    avg_intra = np.mean(intra_scores)
    delta = avg_intra - avg_all

    print(f"\n  All: {avg_all:.1%}   Intra: {avg_intra:.1%}   "
          f"Δ(Intra-All): {delta:+.1%}")
    if delta > 0:
        print("  → In-group favoritism: agents conform MORE to same-persona partners")
    elif delta < 0:
        print("  → Out-group preference: agents conform MORE to different-persona partners")
    else:
        print("  → No in-group effect detected")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table1", default="results/table1/table1.json")
    parser.add_argument("--table2_dir", default="results/table2/")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.table1):
        eval_table1(args.table1)
    else:
        print(f"[skip] {args.table1} not found")

    if os.path.isdir(args.table2_dir):
        eval_table2(args.table2_dir)
    else:
        print(f"[skip] {args.table2_dir} not found")


if __name__ == "__main__":
    main()
