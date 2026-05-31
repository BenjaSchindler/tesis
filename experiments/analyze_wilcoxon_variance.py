#!/usr/bin/env python3
"""Compute Wilcoxon (alongside t-test) + caching/variance decomposition.
Reads thesis_final (cached-linear → zero variance) and classifier_tuning_v2
(RF/XGB re-runs → REAL cross-seed variance). Prints only verified aggregates.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats

ROOT = Path("/home/benja/Desktop/Tesis/filters")
TF = json.load(open(ROOT / "results/thesis_final/final_results.json"))["results"]
V2 = json.load(open(ROOT / "results/classifier_tuning_v2/partial_results.json"))["results"]

DET = {"logistic_regression", "svc_linear", "ridge"}
STO = {"mlp", "random_forest"}
LLM = {"binary_filter", "soft_weighted"}


def paired(by_key, a, b):
    xa, xb = [], []
    for k, m in by_key.items():
        if a in m and b in m:
            xa.append(m[a]); xb.append(m[b])
    if len(xa) < 3:
        return None
    xa, xb = np.array(xa), np.array(xb)
    d = xa - xb
    t_p = stats.ttest_rel(xa, xb).pvalue
    try:
        w_p = stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue
    except ValueError:
        w_p = float("nan")
    sd = d.std(ddof=1)
    return dict(n=len(xa), delta=d.mean() * 100, win=(d > 0).mean() * 100,
               t_p=t_p, w_p=w_p, d=(d.mean() / sd if sd > 1e-12 else float("nan")))


def index(res, keyfields):
    bk = defaultdict(dict)
    for r in res:
        k = tuple(r[f] for f in keyfields)
        bk[k][r["augmentation_method"]] = r["f1_macro"]
    return bk


def show(tag, st):
    if st is None:
        print(f"  {tag:<42} insufficient"); return
    print(f"  {tag:<42} N={st['n']:<4} Δ={st['delta']:+.2f}pp  win={st['win']:.1f}%  "
          f"t_p={st['t_p']:.2e}  W_p={st['w_p']:.2e}  d={st['d']:.2f}")


print("=" * 96)
print("PART 1 — thesis_final: WILCOXON alongside paired t-test (pairing = dataset×classifier×seed)")
print("=" * 96)
bk = index(TF, ["dataset", "classifier", "seed"])
for m in ["soft_weighted", "binary_filter"]:
    show(f"{m} vs smote (ALL clf)", paired(bk, m, "smote"))
# by n-shot
print("  --- soft_weighted vs smote by n-shot ---")
for ns in [10, 25, 50]:
    bkn = index([r for r in TF if r["n_shot"] == ns], ["dataset", "classifier", "seed"])
    show(f"  10/25/50 -> {ns}-shot", paired(bkn, "soft_weighted", "smote"))

print("\n" + "=" * 96)
print("PART 2 — thesis_final: CACHING REDUCES VARIANCE (LLM gens cached → seed-invariant)")
print("=" * 96)
# cross-seed std of LLM methods on deterministic vs stochastic classifiers
g = defaultdict(list)
for r in TF:
    if r["augmentation_method"] in LLM:
        g[(r["dataset"], r["classifier"], r["augmentation_method"])].append(r["f1_macro"])
for label, cset in [("DETERMINISTIC clf (LR/SVC/Ridge)", DET), ("STOCHASTIC clf (MLP/RF)", STO)]:
    stds = [np.std(v, ddof=1) for (ds, c, mth), v in g.items() if c in cset and len(v) >= 2]
    pct0 = np.mean(np.array(stds) == 0) * 100
    print(f"  {label:<36} mean cross-seed std={np.mean(stds)*100:.4f}pp  "
          f"max={np.max(stds)*100:.4f}pp  %zero-variance={pct0:.0f}%  (n_groups={len(stds)})")
print("  --- Effect SURVIVES on stochastic-only subset (where variance is genuine): ---")
bk_sto = index([r for r in TF if r["classifier"] in STO], ["dataset", "classifier", "seed"])
show("soft_weighted vs smote (STOCHASTIC only)", paired(bk_sto, "soft_weighted", "smote"))
bk_det = index([r for r in TF if r["classifier"] in DET], ["dataset", "classifier", "seed"])
show("soft_weighted vs smote (DETERMINISTIC only)", paired(bk_det, "soft_weighted", "smote"))
# also seed-averaged stochastic pairing (matches analyze_seed_asymmetry granularity)
bk_sto_avg = defaultdict(dict)
tmp = defaultdict(lambda: defaultdict(list))
for r in TF:
    if r["classifier"] in STO:
        tmp[(r["dataset"], r["classifier"])][r["augmentation_method"]].append(r["f1_macro"])
for k, md in tmp.items():
    for mth, vals in md.items():
        bk_sto_avg[k][mth] = np.mean(vals)
show("soft_weighted vs smote (STOCH, seed-averaged)", paired(bk_sto_avg, "soft_weighted", "smote"))

print("\n" + "=" * 96)
print("PART 3 — classifier_tuning_v2: RF/XGB RE-RUN with REAL cross-seed variance")
print("=" * 96)
# show variance is NON-zero here (genuine re-runs, not cached-linear)
gv = defaultdict(list)
for r in V2:
    if r["augmentation_method"] in {"binary_filter", "soft_weighted"}:
        gv[(r["dataset"], r["classifier"], r["augmentation_method"])].append(r["f1_macro"])
stds = [np.std(v, ddof=1) for v in gv.values() if len(v) >= 2]
pct0 = np.mean(np.array(stds) == 0) * 100
print(f"  LLM-method cross-seed std (RF+XGB re-run): mean={np.mean(stds)*100:.4f}pp  "
      f"max={np.max(stds)*100:.4f}pp  %zero-variance={pct0:.0f}%  (n_groups={len(stds)})")
print("  --- soft_weighted vs smote, per classifier (paired dataset×n_shot×seed): ---")
bkv = index(V2, ["dataset", "n_shot", "seed", "classifier"])
for clf in sorted({r["classifier"] for r in V2}):
    sub = {k: m for k, m in bkv.items() if k[3] == clf}
    show(f"{clf}", paired(sub, "soft_weighted", "smote"))
print("  --- pooled (all 4 stochastic tree configs): ---")
show("soft_weighted vs smote (ALL v2)", paired(bkv, "soft_weighted", "smote"))
show("binary_filter vs smote (ALL v2)", paired(bkv, "binary_filter", "smote"))
