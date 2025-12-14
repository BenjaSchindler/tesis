#!/usr/bin/env python3
"""
Experiment 09: Wave 8 - Model Experiments

Validates different LLM model configurations:
- W8_gpt5_reasoning: GPT-5-mini with medium reasoning
- W8_gpt5_high: GPT-5-mini with high reasoning

Note: These use gpt-5-mini instead of gpt-4o-mini.
May require different API access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE8_MODELS

CONFIGS = list(WAVE8_MODELS.keys())


def main():
    print("=" * 70)
    print("Experiment 09: Wave 8 - Model Experiments")
    print("=" * 70)
    print(f"\nConfigs to validate: {CONFIGS}")
    print("\nNote: These configs use gpt-5-mini model")

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
            print("Note: gpt-5-mini may require specific API access")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXPERIMENT 09 SUMMARY - WAVE 8 MODELS")
    print("=" * 70)

    for config_name, result in results.items():
        sig = "*" if result.significant else ""
        print(f"  {config_name:20} delta={result.delta_pct:+.2f}% p={result.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
