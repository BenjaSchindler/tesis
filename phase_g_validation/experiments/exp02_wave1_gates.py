#!/usr/bin/env python3
"""
Experiment 02: Wave 1 - Gate Experiments

Validates gate-related configurations:
- W1_low_gate: Very low gate threshold (0.05)
- W1_no_gate: Disable ALL quality gates
- W1_force_problem: Force generation for problem classes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE1_GATES

CONFIGS = list(WAVE1_GATES.keys())


def main():
    print("=" * 70)
    print("Experiment 02: Wave 1 - Gate Experiments")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")

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
    print("EXPERIMENT 02 SUMMARY - WAVE 1 GATES")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
