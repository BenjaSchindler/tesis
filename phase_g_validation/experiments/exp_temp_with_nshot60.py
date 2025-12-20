#!/usr/bin/env python3
"""
Experiment: Temperature Validation with Optimal n_shot=60

Tests temperature interaction with the optimal prompting configuration (60 examples).
Temperatures tested: 0.3, 0.6, 0.9, 1.2, 1.5

Previous findings:
- With n_shot=10: tau=0.9 was optimal (+5.57%)
- Hypothesis: With more context (n_shot=60), optimal temperature may change
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE5B_TEMP_NSHOT60

# All temperature configs
ALL_CONFIGS = list(WAVE5B_TEMP_NSHOT60.keys())


def main():
    print("=" * 70)
    print("Experiment: Temperature with Optimal n_shot=60")
    print("=" * 70)
    print(f"\nConfigs to run: {len(ALL_CONFIGS)}")
    print(f"Temperatures: 0.3, 0.6, 0.9, 1.2, 1.5 (all with n_shot=60)")
    print()

    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    results = {}

    for config_name in ALL_CONFIGS:
        try:
            temp = WAVE5B_TEMP_NSHOT60[config_name]["overrides"]["temperature"]
            print(f"\n{'='*70}")
            print(f"Running {config_name} (temperature={temp})")
            print(f"{'='*70}")

            result = run_config_validation(config_name, cache, verbose=True)
            results[config_name] = result
        except Exception as e:
            print(f"\nERROR validating {config_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEMPERATURE WITH N_SHOT=60 EXPERIMENT SUMMARY")
    print("=" * 70)
    print("\nReference: With n_shot=10, tau=0.9 was optimal (+5.57%)")
    print()

    for config_name, result in sorted(results.items(),
                                       key=lambda x: WAVE5B_TEMP_NSHOT60[x[0]]["overrides"]["temperature"]):
        temp = WAVE5B_TEMP_NSHOT60[config_name]["overrides"]["temperature"]
        sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
        delta_pp = result.delta_mean * 100
        print(f"  tau={temp}: delta={delta_pp:+.2f} pp (rel: {result.delta_pct:+.2f}%) p={result.p_value:.6f} {sig}")

    print("\n" + "=" * 70)
    if results:
        best_config = max(results.items(), key=lambda x: x[1].delta_mean)
        temp_best = WAVE5B_TEMP_NSHOT60[best_config[0]]["overrides"]["temperature"]
        delta_pp_best = best_config[1].delta_mean * 100
        print(f"BEST: tau={temp_best} with delta={delta_pp_best:+.2f} pp")
        print(f"\nComparison with baseline n_shot=60 (tau=default):")
        print(f"  Baseline n_shot=60: +1.67 pp")
        print(f"  Best temperature:   {delta_pp_best:+.2f} pp")
        improvement = delta_pp_best - 1.67
        print(f"  Additional gain:    {improvement:+.2f} pp")


if __name__ == "__main__":
    main()
