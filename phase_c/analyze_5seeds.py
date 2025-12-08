#!/usr/bin/env python3
"""
Multi-Seed Validation Analysis for Phase C v2.1

Analyzes results from 5 seeds (42, 100, 123, 456, 789) and computes:
- Mean/median/std of MID-tier improvements
- 95% confidence interval
- Statistical significance (t-test)
- Per-class improvements across seeds
- Success rate

Expected results:
- MID-tier mean: +1.5% to +2.0%
- 95% CI excludes 0 (all seeds improve)
- Std < 0.5%
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

# MID-tier classes (F1 0.20-0.45)
MID_TIER_CLASSES = ['ENFJ', 'ESFJ', 'ESFP', 'ESTJ', 'ISFJ', 'ISFP', 'ISTJ', 'ESTP']

# Seeds to analyze
SEEDS = [42, 100, 123, 456, 789]

def load_results(seed):
    """Load metrics for a given seed."""
    filepath = Path(f"phaseC_v2.1_seed{seed}_metrics.json")
    if not filepath.exists():
        return None

    with open(filepath) as f:
        return json.load(f)

def calculate_mid_tier_mean(results):
    """Calculate mean MID-tier improvement."""
    if not results or 'improvement' not in results or 'per_class' not in results['improvement']:
        return None

    per_class = results['improvement']['per_class']
    mid_tier_deltas = []

    for cls in MID_TIER_CLASSES:
        if cls in per_class:
            mid_tier_deltas.append(per_class[cls]['delta_pct'])

    if not mid_tier_deltas:
        return None

    return np.mean(mid_tier_deltas)

def main():
    print("=" * 80)
    print("  Phase C v2.1 - Multi-Seed Validation Analysis (5 Seeds)")
    print("=" * 80)
    print()

    # Load all results
    all_results = {}
    missing_seeds = []

    for seed in SEEDS:
        results = load_results(seed)
        if results is None:
            missing_seeds.append(seed)
        else:
            all_results[seed] = results

    print(f"Seeds analyzed: {len(all_results)}/{len(SEEDS)}")
    if missing_seeds:
        print(f"⚠️  Missing seeds: {missing_seeds}")
        print()

    if len(all_results) == 0:
        print("❌ No results found. Run experiments first.")
        return

    print()
    print("-" * 80)

    # Overall macro F1 deltas
    print()
    print("OVERALL MACRO F1 DELTAS")
    print("-" * 80)
    print()

    overall_deltas = []
    for seed in sorted(all_results.keys()):
        results = all_results[seed]
        delta_pct = results['improvement']['f1_delta_pct']
        delta_abs = results['improvement']['f1_delta_abs']
        synthetics = results['synthetic_data']['accepted_count']

        overall_deltas.append(delta_pct)

        print(f"  Seed {seed:3d}: {delta_pct:+.3f}% (abs: {delta_abs:+.6f}, synthetics: {synthetics})")

    print()
    print(f"  Mean:   {np.mean(overall_deltas):+.3f}%")
    print(f"  Median: {np.median(overall_deltas):+.3f}%")
    print(f"  Std:    {np.std(overall_deltas, ddof=1):.3f}%")
    print(f"  Min:    {np.min(overall_deltas):+.3f}%")
    print(f"  Max:    {np.max(overall_deltas):+.3f}%")

    # 95% CI for overall
    if len(overall_deltas) >= 2:
        ci = stats.t.interval(0.95, len(overall_deltas)-1,
                              loc=np.mean(overall_deltas),
                              scale=stats.sem(overall_deltas))
        print(f"  95% CI: [{ci[0]:+.3f}%, {ci[1]:+.3f}%]")

        # t-test vs 0
        t_stat, p_value = stats.ttest_1samp(overall_deltas, 0)
        print(f"  t-test vs 0: t={t_stat:.3f}, p={p_value:.6f}")
        if p_value < 0.05:
            print(f"  ✅ Statistically significant (p < 0.05)")
        else:
            print(f"  ⚠️  Not significant (p >= 0.05)")

    print()
    print("-" * 80)

    # MID-tier analysis
    print()
    print("MID-TIER CLASSES ANALYSIS")
    print("-" * 80)
    print()

    mid_tier_means = []
    for seed in sorted(all_results.keys()):
        mid_mean = calculate_mid_tier_mean(all_results[seed])
        if mid_mean is not None:
            mid_tier_means.append(mid_mean)
            print(f"  Seed {seed:3d}: MID-tier mean = {mid_mean:+.3f}%")

    print()
    print(f"  Mean across seeds:   {np.mean(mid_tier_means):+.3f}%")
    print(f"  Median:              {np.median(mid_tier_means):+.3f}%")
    print(f"  Std:                 {np.std(mid_tier_means, ddof=1):.3f}%")
    print(f"  Min:                 {np.min(mid_tier_means):+.3f}%")
    print(f"  Max:                 {np.max(mid_tier_means):+.3f}%")

    # 95% CI for MID-tier
    if len(mid_tier_means) >= 2:
        ci = stats.t.interval(0.95, len(mid_tier_means)-1,
                              loc=np.mean(mid_tier_means),
                              scale=stats.sem(mid_tier_means))
        print(f"  95% CI:              [{ci[0]:+.3f}%, {ci[1]:+.3f}%]")

        # t-test vs target (+0.10%)
        TARGET = 0.10
        t_stat, p_value = stats.ttest_1samp(mid_tier_means, TARGET)
        print()
        print(f"  t-test vs target (+{TARGET:.2f}%): t={t_stat:.3f}, p={p_value:.6f}")
        if p_value < 0.05 and np.mean(mid_tier_means) > TARGET:
            print(f"  ✅ Significantly better than target (p < 0.05)")
        else:
            print(f"  ⚠️  Not significantly better than target")

    print()
    print("-" * 80)

    # Per-class analysis
    print()
    print("PER-CLASS IMPROVEMENTS (across all seeds)")
    print("-" * 80)
    print()

    class_deltas = {cls: [] for cls in MID_TIER_CLASSES}

    for seed, results in all_results.items():
        per_class = results['improvement'].get('per_class', {})
        for cls in MID_TIER_CLASSES:
            if cls in per_class:
                class_deltas[cls].append(per_class[cls]['delta_pct'])

    print(f"{'Class':<6} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>8} {'Max':>8} {'Success':>8}")
    print("-" * 80)

    for cls in MID_TIER_CLASSES:
        deltas = class_deltas[cls]
        if not deltas:
            continue

        success_rate = sum(1 for d in deltas if d > 0) / len(deltas) * 100

        print(f"{cls:<6} {np.mean(deltas):+8.3f}% {np.median(deltas):+8.3f}% "
              f"{np.std(deltas, ddof=1):7.3f}% {np.min(deltas):+8.3f}% "
              f"{np.max(deltas):+8.3f}% {success_rate:7.0f}%")

    print()
    print("-" * 80)

    # Success rate
    print()
    print("SUCCESS RATE")
    print("-" * 80)
    print()

    overall_improved = sum(1 for d in overall_deltas if d > 0)
    mid_improved = sum(1 for d in mid_tier_means if d > 0)

    print(f"  Overall improved: {overall_improved}/{len(overall_deltas)} seeds ({overall_improved/len(overall_deltas)*100:.0f}%)")
    print(f"  MID-tier improved: {mid_improved}/{len(mid_tier_means)} seeds ({mid_improved/len(mid_tier_means)*100:.0f}%)")

    # Target achievement
    target_achieved = sum(1 for d in mid_tier_means if d >= 0.10)
    print(f"  Target achieved (+0.10% MID-tier): {target_achieved}/{len(mid_tier_means)} seeds ({target_achieved/len(mid_tier_means)*100:.0f}%)")

    print()
    print("-" * 80)

    # Final verdict
    print()
    print("FINAL VERDICT")
    print("-" * 80)
    print()

    mean_mid_tier = np.mean(mid_tier_means)
    ci_lower = ci[0] if len(mid_tier_means) >= 2 else mean_mid_tier

    if mean_mid_tier >= 1.5 and ci_lower > 0:
        print("  ✅ EXCELLENT - v2.1 is robust and exceeds target!")
        print(f"     MID-tier mean: {mean_mid_tier:+.3f}% (target: +0.10%)")
        print(f"     Improvement factor: {mean_mid_tier/0.10:.1f}×")
    elif mean_mid_tier >= 1.0 and ci_lower > 0:
        print("  ✅ GOOD - v2.1 is robust")
        print(f"     MID-tier mean: {mean_mid_tier:+.3f}% (target: +0.10%)")
    elif mean_mid_tier >= 0.10:
        print("  ⚠️  ACCEPTABLE - Target achieved but with variance")
        print(f"     MID-tier mean: {mean_mid_tier:+.3f}% (target: +0.10%)")
    else:
        print("  ❌ FAILED - Target not achieved")
        print(f"     MID-tier mean: {mean_mid_tier:+.3f}% (target: +0.10%)")

    print()
    print("=" * 80)

    # Save summary
    summary = {
        "seeds_analyzed": len(all_results),
        "overall": {
            "mean": float(np.mean(overall_deltas)),
            "median": float(np.median(overall_deltas)),
            "std": float(np.std(overall_deltas, ddof=1)) if len(overall_deltas) > 1 else 0.0,
            "min": float(np.min(overall_deltas)),
            "max": float(np.max(overall_deltas)),
            "ci_95": [float(ci[0]), float(ci[1])] if len(overall_deltas) >= 2 else None
        },
        "mid_tier": {
            "mean": float(np.mean(mid_tier_means)),
            "median": float(np.median(mid_tier_means)),
            "std": float(np.std(mid_tier_means, ddof=1)) if len(mid_tier_means) > 1 else 0.0,
            "min": float(np.min(mid_tier_means)),
            "max": float(np.max(mid_tier_means)),
            "ci_95": [float(ci[0]), float(ci[1])] if len(mid_tier_means) >= 2 else None,
            "target_achieved_count": int(target_achieved),
            "target_achieved_rate": float(target_achieved / len(mid_tier_means)) if mid_tier_means else 0.0
        },
        "per_class": {
            cls: {
                "mean": float(np.mean(deltas)),
                "median": float(np.median(deltas)),
                "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "success_rate": float(sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else 0.0
            }
            for cls, deltas in class_deltas.items() if deltas
        },
        "success_rate": {
            "overall_improved": int(overall_improved),
            "mid_improved": int(mid_improved)
        }
    }

    with open("phaseC_v2.1_5seeds_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("✅ Summary saved to phaseC_v2.1_5seeds_summary.json")
    print()

if __name__ == "__main__":
    main()
