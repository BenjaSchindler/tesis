#!/usr/bin/env python3
"""
Experiment 12: Phase F Derived Configs (Focused on Problem Classes)

Validates Phase F-inspired configurations designed to improve problem classes:
- PF_tier_boost: Tier-based weighting (LOW=2.0, MID=0.8, HIGH=0.3)
- PF_high_budget_problem: High budget (25%) + force problem classes
- PF_optimal_focused: Full optimal + force + contrastive

These configs combine Phase F findings with targeted strategies for
the 5 problem classes: ENFJ, ESFJ, ESFP, ESTJ, ISTJ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import PHASE_F_DERIVED
from base_config import PROBLEM_CLASSES

CONFIGS = list(PHASE_F_DERIVED.keys())


def main():
    print("=" * 70)
    print("Experiment 12: Phase F Derived - Problem Class Focus")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")
    print(f"Target problem classes: {PROBLEM_CLASSES}")

    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    results = {}

    for config_name in CONFIGS:
        try:
            result = run_config_validation(config_name, cache, verbose=True)
            results[config_name] = result
        except Exception as e:
            print(f"\nERROR validating {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 12 SUMMARY - PHASE F DERIVED (Problem Class Focus)")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"\n  {config_name}")
        print(f"    Overall:  delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")

        # Show problem class specific deltas
        if result.problem_class_delta:
            print(f"    Problem classes:")
            for cls in PROBLEM_CLASSES:
                delta = result.problem_class_delta.get(cls, 0)
                print(f"      {cls}: {delta:+.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
