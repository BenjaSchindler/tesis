#!/usr/bin/env python3
"""Analyze results of exp_classifier_tuning.py."""

import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

RESULTS = Path(__file__).parent.parent / "results" / "classifier_tuning"
PATH = RESULTS / "final_results.json"
if not PATH.exists():
    PATH = RESULTS / "partial_results.json"
    print(f"[!] Using partial results from {PATH}")


def load():
    with open(PATH) as f:
        data = json.load(f)
    return data["results"]


def by_key(results, fn):
    out = defaultdict(list)
    for r in results:
        out[fn(r)].append(r)
    return out


def f1(r):
    return r["f1_macro"]


def paired_deltas(results, method_a, method_b, group_keys):
    """For each group, compute mean F1 for method_a and method_b, return paired deltas."""
    by_group = by_key(results, lambda r: tuple(r[k] for k in group_keys))
    pairs = []
    for grp, items in by_group.items():
        a = [r for r in items if r["augmentation_method"] == method_a]
        b = [r for r in items if r["augmentation_method"] == method_b]
        if not a or not b:
            continue
        # Average across seeds
        a_mean = np.mean([r["f1_macro"] for r in a])
        b_mean = np.mean([r["f1_macro"] for r in b])
        pairs.append((grp, a_mean - b_mean, a_mean, b_mean))
    return pairs


def main():
    results = load()
    print(f"Loaded {len(results)} results")
    print(f"Classifiers: {sorted(set(r['classifier'] for r in results))}")
    print(f"Methods:     {sorted(set(r['augmentation_method'] for r in results))}")
    print(f"Datasets:    {sorted(set(r['dataset'] for r in results))}")
    print()

    # =============================================
    # Per-classifier: soft_weighted vs smote
    # =============================================
    print("=" * 70)
    print("PAIRED DELTAS: soft_weighted vs SMOTE (per classifier)")
    print("Each pair = average F1 over 5 seeds, per (dataset, n_shot)")
    print("=" * 70)
    print(f"{'Classifier':<25}{'mean Δ (pp)':>14}{'Win %':>10}{'p-val':>10}{'d':>8}")
    print("-" * 70)
    for clf in sorted(set(r['classifier'] for r in results)):
        sub = [r for r in results if r['classifier'] == clf]
        pairs = paired_deltas(sub, 'soft_weighted', 'smote',
                              ['dataset', 'n_shot'])
        if not pairs:
            continue
        deltas = np.array([p[1] for p in pairs]) * 100
        winrate = (deltas > 0).mean() * 100
        if len(deltas) > 1 and deltas.std() > 0:
            t, p = stats.ttest_1samp(deltas, 0)
            d = deltas.mean() / deltas.std()
        else:
            t, p, d = 0, 1, 0
        sig = " *" if p < 0.05 else ""
        print(f"{clf:<25}{deltas.mean():>+12.3f}  {winrate:>7.1f}%{p:>10.4f}{d:>7.2f}{sig}")

    print()
    print("=" * 70)
    print("PAIRED DELTAS: soft_weighted vs no_augmentation (per classifier)")
    print("=" * 70)
    print(f"{'Classifier':<25}{'mean Δ (pp)':>14}{'Win %':>10}{'p-val':>10}{'d':>8}")
    print("-" * 70)
    for clf in sorted(set(r['classifier'] for r in results)):
        sub = [r for r in results if r['classifier'] == clf]
        pairs = paired_deltas(sub, 'soft_weighted', 'no_augmentation',
                              ['dataset', 'n_shot'])
        if not pairs:
            continue
        deltas = np.array([p[1] for p in pairs]) * 100
        winrate = (deltas > 0).mean() * 100
        if len(deltas) > 1 and deltas.std() > 0:
            t, p = stats.ttest_1samp(deltas, 0)
            d = deltas.mean() / deltas.std()
        else:
            t, p, d = 0, 1, 0
        sig = " *" if p < 0.05 else ""
        print(f"{clf:<25}{deltas.mean():>+12.3f}  {winrate:>7.1f}%{p:>10.4f}{d:>7.2f}{sig}")

    print()
    print("=" * 70)
    print("F1 means by classifier × method (averaged across all datasets, shots, seeds)")
    print("=" * 70)
    clfs = sorted(set(r['classifier'] for r in results))
    mths = sorted(set(r['augmentation_method'] for r in results))
    print(f"{'classifier':<25}" + "".join(f"{m:>18}" for m in mths))
    for clf in clfs:
        row = f"{clf:<25}"
        for m in mths:
            f1s = [r['f1_macro'] for r in results if r['classifier'] == clf and r['augmentation_method'] == m]
            row += f"{np.mean(f1s):>18.4f}"
        print(row)

    print()
    print("=" * 70)
    print("Per (n_shot) deltas: soft_weighted vs smote, per classifier")
    print("=" * 70)
    for n_shot in sorted(set(r['n_shot'] for r in results)):
        print(f"\n--- {n_shot}-shot ---")
        print(f"{'Classifier':<25}{'mean Δ (pp)':>14}{'Win %':>10}")
        for clf in sorted(set(r['classifier'] for r in results)):
            sub = [r for r in results if r['classifier'] == clf and r['n_shot'] == n_shot]
            pairs = paired_deltas(sub, 'soft_weighted', 'smote', ['dataset'])
            if not pairs:
                continue
            deltas = np.array([p[1] for p in pairs]) * 100
            winrate = (deltas > 0).mean() * 100
            print(f"  {clf:<23}{deltas.mean():>+12.3f}  {winrate:>7.1f}%")


if __name__ == "__main__":
    main()
