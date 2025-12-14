#!/usr/bin/env python3
"""
Experiment 01: Phase F Ensemble Components Validation

Validates the 4 core components of ENS_Top3_G5:
- CMB3_skip: F1-budget scaling
- CF1_conf_band: Confidence band filtering
- V4_ultra: High volume generation
- G5_K25_medium: High K=25 samples

These are run with Phase F optimal base parameters.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import (
    load_data, EmbeddingCache, run_config_validation, print_summary
)
from config_definitions import PHASE_F_COMPONENTS

# Configs to validate
CONFIGS = list(PHASE_F_COMPONENTS.keys())


def main():
    print("=" * 70)
    print("Experiment 01: Phase F Ensemble Components")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")

    # Load data and embeddings
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
    print("EXPERIMENT 01 SUMMARY")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
