#!/usr/bin/env python3
"""Analyze exp_classifier_tuning_v2.py — attribution decomposition + rigorous stats.

Produces:
  1. Δ vs SMOTE per (classifier x method) with paired t-test, Wilcoxon, Holm, d.
  2. The causal decomposition (per classifier):
        LLM effect    = llm_unfiltered - no_augmentation
        FILTER effect = binary_filter  - llm_unfiltered   <-- the thesis claim
        WEIGHT effect = soft_weighted  - binary_filter
  3. Breakdown by n-shot.
  4. CV-vs-fixed comparison (did honest tuning hold up?).
"""

import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results" / "classifier_tuning_v2"


def load():
    final = RESULTS_DIR / "final_results.json"
    partial = RESULTS_DIR / "partial_results.json"
    path = final if final.exists() else partial
    if not path.exists():
        sys.exit(f"No results found in {RESULTS_DIR}")
    if path == partial:
        print(f"[!] Using PARTIAL results from {path}")
    data = json.load(open(path))
    res = data["results"]
    print(f"Loaded {len(res)} results\n")
    return res


def key(r):
    return (r["dataset"], r["n_shot"], r["seed"], r["classifier"])


def cohen_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    # sd==0 with a non-zero mean => effect size is UNDEFINED (a constant difference),
    # not zero. Return nan so it never reads as a misleading d=0.00 downstream.
    return float(diffs.mean() / sd) if sd > 1e-12 else float("nan")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        prev = max(prev, min(val, 1.0))
        adj[i] = prev
    return adj


def paired(by_key, method_a, method_b):
    """Paired (method_a - method_b) over shared keys. Returns dict of stats."""
    a, b = [], []
    for k, methods in by_key.items():
        if method_a in methods and method_b in methods:
            a.append(methods[method_a])
            b.append(methods[method_b])
    if len(a) < 3:
        return None
    a, b = np.array(a), np.array(b)
    diffs = a - b
    t_p = stats.ttest_rel(a, b).pvalue
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w_p = float("nan")
    return {
        "n": len(a),
        "mean_delta_pp": float(diffs.mean() * 100),
        "win_rate": float((diffs > 0).mean() * 100),
        "t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohen_d": cohen_d_paired(diffs),
    }


def build_index(res):
    by_key = defaultdict(dict)
    for r in res:
        by_key[key(r)][r["augmentation_method"]] = r["f1_macro"]
    return by_key


def main():
    res = load()
    by_key = build_index(res)
    classifiers = sorted({r["classifier"] for r in res})

    # ---- 1. Δ vs SMOTE per (classifier x method) ----
    print("=" * 78)
    print("1. Δ vs SMOTE  (paired, per classifier)")
    print("=" * 78)
    print(f"{'classifier':<20}{'method':<16}{'Δpp':>8}{'win%':>7}{'t_p':>10}{'wilc_p':>10}{'d':>7}")
    rows, pvals = [], []
    for clf in classifiers:
        bk = {k: m for k, m in by_key.items() if k[3] == clf}
        for method in ["no_augmentation", "llm_unfiltered", "binary_filter", "soft_weighted"]:
            st = paired(bk, method, "smote")
            if st:
                rows.append((clf, method, st))
                pvals.append(st["t_p"])
    adj = holm(pvals) if pvals else []
    for (clf, method, st), p_adj in zip(rows, adj):
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
        print(f"{clf:<20}{method:<16}{st['mean_delta_pp']:>+8.2f}{st['win_rate']:>6.0f}%"
              f"{st['t_p']:>10.4f}{st['wilcoxon_p']:>10.4f}{st['cohen_d']:>7.2f} {sig}")

    # ---- 2. Causal decomposition (FIXED classifiers ONLY) ----
    # The decomposition is only CLEAN on fixed classifiers, where hyperparameters are
    # identical across all 5 methods, so the ONLY thing varying between conditions is
    # the DATA. For *_cv classifiers, hyperparameters are selected per-method, which
    # would confound data-quality with tuning — so they are EXCLUDED here. (Block 4
    # reports the CV-vs-fixed tuning effect separately.) Holm-corrected within this
    # family, treated as separate from block 1 (vs-SMOTE is a different null family).
    print("\n" + "=" * 78)
    print("2. CAUSAL DECOMPOSITION  (core attribution test — FIXED classifiers only, Holm-corrected)")
    print("   LLM effect    = llm_unfiltered - no_augmentation")
    print("   FILTER effect = binary_filter  - llm_unfiltered   <-- THE THESIS")
    print("   WEIGHT effect = soft_weighted  - binary_filter")
    print("=" * 78)
    steps = [
        ("LLM    ", "llm_unfiltered", "no_augmentation"),
        ("FILTER ", "binary_filter", "llm_unfiltered"),
        ("WEIGHT ", "soft_weighted", "binary_filter"),
    ]
    fixed_classifiers = [c for c in classifiers if not c.endswith("_cv")]
    decomp_rows, decomp_pvals = [], []
    for clf in fixed_classifiers:
        bk = {k: m for k, m in by_key.items() if k[3] == clf}
        for label, a, b in steps:
            st = paired(bk, a, b)
            if st:
                decomp_rows.append((clf, label, st))
                decomp_pvals.append(st["t_p"])
    decomp_adj = holm(decomp_pvals) if decomp_pvals else []
    print(f"{'classifier':<20}{'effect':<9}{'Δpp':>8}{'win%':>7}{'t_p':>10}{'p_holm':>10}{'d':>7}")
    last_clf = None
    for (clf, label, st), p_adj in zip(decomp_rows, decomp_adj):
        if last_clf is not None and clf != last_clf:
            print()
        last_clf = clf
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
        print(f"{clf:<20}{label:<9}{st['mean_delta_pp']:>+8.2f}{st['win_rate']:>6.0f}%"
              f"{st['t_p']:>10.4f}{p_adj:>10.4f}{st['cohen_d']:>7.2f} {sig}")
    if not decomp_rows:
        print("  [!] no fixed classifiers found in results")

    # ---- 3. By n-shot (soft_weighted vs smote) ----
    print("=" * 78)
    print("3. soft_weighted vs SMOTE by n-shot x classifier")
    print("=" * 78)
    shots = sorted({r["n_shot"] for r in res})
    print(f"{'classifier':<20}" + "".join(f"{f'{s}-shot':>12}" for s in shots))
    for clf in classifiers:
        line = f"{clf:<20}"
        for s in shots:
            bk = {k: m for k, m in by_key.items() if k[3] == clf and k[1] == s}
            st = paired(bk, "soft_weighted", "smote")
            line += f"{st['mean_delta_pp']:>+11.2f}" if st else f"{'--':>12}"
        print(line)

    # ---- 4. CV vs fixed: did honest tuning beat the hand-picked default? ----
    print("\n" + "=" * 78)
    print("4. CV-selected vs fixed-default (honest tuning held up?)")
    print("=" * 78)
    pairs = [("random_forest_cv", "random_forest"), ("xgboost_cv", "xgboost")]
    for cv_name, fixed_name in pairs:
        # pair on (dataset, n_shot, seed, method)
        cv_by, fx_by = {}, {}
        for r in res:
            kk = (r["dataset"], r["n_shot"], r["seed"], r["augmentation_method"])
            if r["classifier"] == cv_name:
                cv_by[kk] = r["f1_macro"]
            elif r["classifier"] == fixed_name:
                fx_by[kk] = r["f1_macro"]
        shared = sorted(set(cv_by) & set(fx_by))
        if len(shared) < 3:
            continue
        a = np.array([cv_by[k] for k in shared])
        b = np.array([fx_by[k] for k in shared])
        d = a - b
        t_p = stats.ttest_rel(a, b).pvalue
        print(f"{cv_name:<20} vs {fixed_name:<18} Δ={d.mean()*100:>+6.2f}pp  "
              f"win={float((d>0).mean()*100):>3.0f}%  t_p={t_p:.4f}  (n={len(shared)})")


if __name__ == "__main__":
    main()
