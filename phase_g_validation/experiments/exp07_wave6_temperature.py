#!/usr/bin/env python3
"""
Experiment 07: Wave 6 - Temperature Experiments

Validates temperature-related configurations:
- W6_temp_low: Low temperature (0.3) - MATCHES OPTIMAL!
- W6_temp_high: High temperature (0.9)
- W6_temp_extreme: Extreme temperature (1.0)

Note: W6_temp_low uses 0.3, which matches Phase F optimal.
This serves as cross-validation of the Phase F finding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE6_TEMPERATURE

CONFIGS = list(WAVE6_TEMPERATURE.keys())


def main():
    print("=" * 70)
    print("Experiment 07: Wave 6 - Temperature Experiments")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")
    print("\nNote: W6_temp_low (0.3) matches Phase F optimal - cross-validation!")

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

    print("\n" + "=" * 70)
    print("EXPERIMENT 07 SUMMARY - WAVE 6 TEMPERATURE")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        temp = WAVE6_TEMPERATURE[config_name]["overrides"]["temperature"]
        print(f"  {config_name:20} T={temp} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
