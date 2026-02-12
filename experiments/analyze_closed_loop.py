#!/usr/bin/env python3
"""
Closed-Loop Results Analysis

Analyzes results from exp_closed_loop.py to determine:
1. Whether prompt improvement increased acceptance rates and F1
2. Which failure modes were most common
3. How many iterations were needed to converge
4. Whether closed-loop beats single-pass and SMOTE
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "closed_loop" / "experiment_results.json"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]
    summary = data["summary"]

    print("=" * 70)
    print("CLOSED-LOOP PROMPT IMPROVEMENT — RESULTS ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target per class: {config['target_per_class']}")
    print(f"  Max iterations: {config['max_iterations']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Filters: {[f['name'] for f in config['filters']]}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Total results: {len(results)}")

    datasets = sorted(set(r["dataset"] for r in results))
    filters = sorted(set(r["filter_name"] for r in results))

    # ---- ANALYSIS 1: Overall Performance ----
    print("\n" + "=" * 70)
    print("1. OVERALL PERFORMANCE")
    print("=" * 70)

    cl_vs_smote = [r["delta_vs_smote"] for r in results]
    cl_vs_sp = [r["delta_vs_single_pass"] for r in results]
    print(f"\n  Closed-loop vs SMOTE:       mean={np.mean(cl_vs_smote):+.2f}pp, "
          f"win rate={sum(1 for d in cl_vs_smote if d > 0)/len(cl_vs_smote)*100:.1f}%")
    print(f"  Closed-loop vs single-pass: mean={np.mean(cl_vs_sp):+.2f}pp, "
          f"win rate={sum(1 for d in cl_vs_sp if d > 0)/len(cl_vs_sp)*100:.1f}%")

    # ---- ANALYSIS 2: By Filter ----
    print("\n" + "=" * 70)
    print("2. PERFORMANCE BY FILTER")
    print("=" * 70)
    print(f"\n{'Filter':<15} {'vs SMOTE':>12} {'Win%':>7} {'vs Single':>12} {'Win%':>7}")
    print("-" * 55)

    for filt in filters:
        f_results = [r for r in results if r["filter_name"] == filt]
        vs_smote = [r["delta_vs_smote"] for r in f_results]
        vs_sp = [r["delta_vs_single_pass"] for r in f_results]
        smote_win = sum(1 for d in vs_smote if d > 0) / len(vs_smote) * 100
        sp_win = sum(1 for d in vs_sp if d > 0) / len(vs_sp) * 100
        print(f"{filt:<15} {np.mean(vs_smote):>+11.2f}p {smote_win:>6.1f}% "
              f"{np.mean(vs_sp):>+11.2f}p {sp_win:>6.1f}%")

    # ---- ANALYSIS 3: By Dataset ----
    print("\n" + "=" * 70)
    print("3. PERFORMANCE BY DATASET")
    print("=" * 70)
    print(f"\n{'Dataset':<35} {'vs SMOTE':>12} {'Win%':>7} {'CL F1':>10} {'SP F1':>10}")
    print("-" * 75)

    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        vs_smote = [r["delta_vs_smote"] for r in ds_results]
        cl_f1 = [r["closed_loop_f1"] for r in ds_results]
        sp_f1 = [r["single_pass_f1"] for r in ds_results]
        smote_win = sum(1 for d in vs_smote if d > 0) / len(vs_smote) * 100
        print(f"{ds:<35} {np.mean(vs_smote):>+11.2f}p {smote_win:>6.1f}% "
              f"{np.mean(cl_f1):>10.4f} {np.mean(sp_f1):>10.4f}")

    # ---- ANALYSIS 4: Convergence Analysis ----
    print("\n" + "=" * 70)
    print("4. CONVERGENCE ANALYSIS")
    print("=" * 70)

    convergence_reasons = defaultdict(int)
    iterations_per_class = []
    for r in results:
        for cls, reason in r.get("convergence_reasons", {}).items():
            convergence_reasons[reason] += 1
        for cls, n_iter in r.get("n_iterations_per_class", {}).items():
            iterations_per_class.append(n_iter)

    total_classes = sum(convergence_reasons.values())
    print(f"\n  Convergence reasons (total class-level runs: {total_classes}):")
    for reason, count in sorted(convergence_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<25}: {count:>4} ({count/total_classes*100:.1f}%)")

    if iterations_per_class:
        print(f"\n  Iterations per class:")
        print(f"    Mean: {np.mean(iterations_per_class):.2f}")
        print(f"    Median: {np.median(iterations_per_class):.1f}")
        print(f"    Max: {max(iterations_per_class)}")
        print(f"    Min: {min(iterations_per_class)}")

    # ---- ANALYSIS 5: Failure Mode Analysis ----
    print("\n" + "=" * 70)
    print("5. FAILURE MODE ANALYSIS")
    print("=" * 70)

    all_failures = defaultdict(int)
    failure_by_iteration = defaultdict(lambda: defaultdict(int))
    acceptance_by_iteration = defaultdict(list)

    for r in results:
        for cls, iters in r.get("iteration_metrics", {}).items():
            for it_data in iters:
                it_num = it_data.get("iteration", 0)
                for fail_type, count in it_data.get("rejection_distribution", {}).items():
                    all_failures[fail_type] += count
                    failure_by_iteration[it_num][fail_type] += count
                acc = it_data.get("acceptance_rate", 0)
                acceptance_by_iteration[it_num].append(acc)

    total_rejections = sum(all_failures.values())
    print(f"\n  Overall failure distribution (total rejections: {total_rejections}):")
    for fail_type, count in sorted(all_failures.items(), key=lambda x: -x[1]):
        print(f"    {fail_type:<25}: {count:>5} ({count/total_rejections*100:.1f}%)")

    # ---- ANALYSIS 6: Acceptance Rate Progression ----
    print("\n" + "=" * 70)
    print("6. ACCEPTANCE RATE BY ITERATION")
    print("=" * 70)
    print(f"\n{'Iteration':>10} {'Mean Accept':>14} {'Std':>8} {'N':>6}")
    print("-" * 40)

    for it_num in sorted(acceptance_by_iteration.keys()):
        rates = acceptance_by_iteration[it_num]
        print(f"{it_num:>10} {np.mean(rates)*100:>13.1f}% {np.std(rates)*100:>7.1f}% {len(rates):>6}")

    # Check if acceptance rates improve across iterations
    if len(acceptance_by_iteration) >= 2:
        first_rates = acceptance_by_iteration[0]
        last_it = max(acceptance_by_iteration.keys())
        last_rates = acceptance_by_iteration[last_it]
        if len(first_rates) > 0 and len(last_rates) > 0:
            mean_first = np.mean(first_rates)
            mean_last = np.mean(last_rates)
            delta = (mean_last - mean_first) * 100
            print(f"\n  Acceptance rate change (iteration 0 -> {last_it}): {delta:+.1f}pp")
            if delta > 0:
                print("  >> Prompt improvement INCREASED acceptance rates")
            else:
                print("  >> Prompt improvement did NOT increase acceptance rates")

    # ---- ANALYSIS 7: LLM Call Efficiency ----
    print("\n" + "=" * 70)
    print("7. LLM CALL EFFICIENCY")
    print("=" * 70)

    llm_calls = [r["total_llm_calls"] for r in results]
    print(f"\n  Total LLM calls per experiment:")
    print(f"    Mean: {np.mean(llm_calls):.1f}")
    print(f"    Median: {np.median(llm_calls):.1f}")
    print(f"    Min: {min(llm_calls)}")
    print(f"    Max: {max(llm_calls)}")

    # F1 improvement per LLM call
    efficiency = []
    for r in results:
        if r["total_llm_calls"] > 0:
            eff = r["delta_vs_smote"] / r["total_llm_calls"]
            efficiency.append(eff)
    if efficiency:
        print(f"\n  F1 improvement per LLM call (vs SMOTE):")
        print(f"    Mean: {np.mean(efficiency):.4f}pp/call")
        print(f"    Std:  {np.std(efficiency):.4f}")

    # ---- ANALYSIS 8: Statistical Test ----
    print("\n" + "=" * 70)
    print("8. STATISTICAL TESTS")
    print("=" * 70)

    # Paired test: closed_loop F1 vs smote F1
    cl_f1s = np.array([r["closed_loop_f1"] for r in results])
    smote_f1s = np.array([r["smote_f1"] for r in results])
    sp_f1s = np.array([r["single_pass_f1"] for r in results])

    deltas_vs_smote = cl_f1s - smote_f1s
    deltas_vs_sp = cl_f1s - sp_f1s

    t_smote, p_smote = stats.ttest_rel(cl_f1s, smote_f1s)
    t_sp, p_sp = stats.ttest_rel(cl_f1s, sp_f1s)

    d_smote = np.mean(deltas_vs_smote) / (np.std(deltas_vs_smote, ddof=1) + 1e-10)
    d_sp = np.mean(deltas_vs_sp) / (np.std(deltas_vs_sp, ddof=1) + 1e-10)

    print(f"\n  Closed-loop vs SMOTE:")
    print(f"    Mean delta: {np.mean(deltas_vs_smote)*100:+.2f}pp")
    print(f"    t-statistic: {t_smote:.3f}")
    print(f"    p-value: {p_smote:.6f}")
    print(f"    Cohen's d: {d_smote:.3f}")
    print(f"    Significant: {'YES' if p_smote < 0.05 else 'NO'}")

    print(f"\n  Closed-loop vs single-pass:")
    print(f"    Mean delta: {np.mean(deltas_vs_sp)*100:+.2f}pp")
    print(f"    t-statistic: {t_sp:.3f}")
    print(f"    p-value: {p_sp:.6f}")
    print(f"    Cohen's d: {d_sp:.3f}")
    print(f"    Significant: {'YES' if p_sp < 0.05 else 'NO'}")

    # ---- CONCLUSION ----
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
  Overall Assessment:
    - Closed-loop vs SMOTE: {np.mean(cl_vs_smote):+.2f}pp, {sum(1 for d in cl_vs_smote if d > 0)/len(cl_vs_smote)*100:.0f}% win rate
    - Closed-loop vs single-pass: {np.mean(cl_vs_sp):+.2f}pp, {sum(1 for d in cl_vs_sp if d > 0)/len(cl_vs_sp)*100:.0f}% win rate

  Dominant failure mode: DISTANCE_OUTLIER ({all_failures.get('DISTANCE_OUTLIER', 0)}/{total_rejections} = {all_failures.get('DISTANCE_OUTLIER', 0)/total_rejections*100:.0f}%)
  Convergence: {convergence_reasons.get('TARGET_REACHED', 0)}/{total_classes} classes reached target ({convergence_reasons.get('TARGET_REACHED', 0)/total_classes*100:.0f}%)

  The closed-loop approach shows modest improvement over single-pass
  ({np.mean(cl_vs_sp):+.2f}pp) but limited advantage over SMOTE
  ({np.mean(cl_vs_smote):+.2f}pp with only {sum(1 for d in cl_vs_smote if d > 0)/len(cl_vs_smote)*100:.0f}% win rate).

  Key insight: Simple single-pass filtering with top-N selection is
  competitive with iterative prompt improvement, suggesting that the
  initial LLM generations are already reasonably well-targeted.
""")


if __name__ == "__main__":
    main()
