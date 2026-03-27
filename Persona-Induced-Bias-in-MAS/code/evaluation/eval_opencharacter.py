"""
Evaluate OpenCharacter Experiment 1 results.

Reads the JSON files produced by accuracy_opencharacter.py and
cps_opencharacter.py and prints a summary table comparable to
Tables 1, 3, and 4 in the paper.

Usage:
    python eval_opencharacter.py
    python eval_opencharacter.py --accuracy_path results/accuracy/opencharacter.json \
                                  --cps_dir results/cps_opencharacter
"""

import json
import os
import argparse
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def eval_accuracy(path: str):
    data = load_json(path)
    print("\n=== Single-Persona Accuracy (Experiment 1 baseline) ===")
    print(f"{'Persona':15s}  {'Accuracy':>8s}")
    print("-" * 28)
    accuracies = []
    for entry in data:
        print(f"  {entry['persona']:15s}  {entry['accuracy']:.3f}")
        accuracies.append(entry['accuracy'])
    print("-" * 28)
    print(f"  {'Δmax-min':15s}  {max(accuracies) - min(accuracies):.3f}")
    print(f"  {'mean':15s}  {np.mean(accuracies):.3f}")


def eval_cps(cps_dir: str):
    for mode in ["trustworthiness", "insistence"]:
        path = os.path.join(cps_dir, f"{mode}.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        metric = mode  # key name matches
        scores = [e[metric] for e in data]

        print(f"\n=== {mode.capitalize()} Scores ===")
        print(f"{'Persona':15s}  {metric[:8]:>8s}")
        print("-" * 28)
        for entry in sorted(data, key=lambda x: -x[metric]):
            print(f"  {entry['persona']:15s}  {entry[metric]:.3f}")
        print("-" * 28)
        print(f"  {'Δmax-min':15s}  {max(scores) - min(scores):.3f}")
        print(f"  {'mean':15s}  {np.mean(scores):.3f}")

        # Highlight extremes
        top = max(data, key=lambda x: x[metric])
        bot = min(data, key=lambda x: x[metric])
        print(f"\n  Highest {mode}: {top['persona']} ({top[metric]:.3f})")
        print(f"  Lowest  {mode}: {bot['persona']} ({bot[metric]:.3f})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy_path", type=str,
                        default="results/accuracy/opencharacter.json")
    parser.add_argument("--cps_dir", type=str,
                        default="results/cps_opencharacter")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.accuracy_path):
        eval_accuracy(args.accuracy_path)
    else:
        print(f"[skip] accuracy results not found at {args.accuracy_path}")

    if os.path.isdir(args.cps_dir):
        eval_cps(args.cps_dir)
    else:
        print(f"[skip] cps results not found at {args.cps_dir}")


if __name__ == "__main__":
    main()
