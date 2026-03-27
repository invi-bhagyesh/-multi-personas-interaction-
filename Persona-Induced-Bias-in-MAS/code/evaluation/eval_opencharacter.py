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
        print(f"  {'range':15s}  {max(t_scores)-min(t_scores):7.1%}  "
              f"{max(i_scores)-min(i_scores):7.1%}")
        print(f"  {'std':15s}  {np.std(t_scores):7.1%}  {np.std(i_scores):7.1%}")
    print(f"  {'mean':15s}  {np.mean(t_scores):7.1%}  {np.mean(i_scores):7.1%}")


def eval_table2(cps_dir):
    path = os.path.join(cps_dir, "table2.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return

    results = load_json(path)
    if not results:
        print(f"[skip] No table2 results in {cps_dir}")
        return

    conf = {}
    for r in results:
        conf[(r["persona1"], r["persona2"])] = r["conformity"]

    personas = sorted(set(r["persona1"] for r in results))
    col_w = max(8, max(len(p) for p in personas) + 1)

    print(f"\n  {'C(row->col)':15s}" + "".join(f"  {p:>{col_w}s}" for p in personas))
    print(f"  {'-' * (15 + (col_w + 2) * len(personas))}")

    all_scores, intra_scores = [], []
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

    avg_all = np.mean(all_scores)
    avg_intra = np.mean(intra_scores)
    delta = avg_intra - avg_all
    print(f"\n  All: {avg_all:.1%}   Intra: {avg_intra:.1%}   Delta: {delta:+.1%}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table1", default="results/cps_opencharacter/table1.json")
    parser.add_argument("--cps_dir", default="results/cps_opencharacter/")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.table1):
        eval_table1(args.table1)
    else:
        print(f"[skip] {args.table1} not found")

    if os.path.isdir(args.cps_dir):
        eval_table2(args.cps_dir)
    else:
        print(f"[skip] {args.cps_dir} not found")


if __name__ == "__main__":
    main()
