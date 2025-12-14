#!/usr/bin/env python3
"""
Quick test script to validate a single config.

Usage:
    python3 test_single_config.py                    # Test W1_low_gate (default)
    python3 test_single_config.py CMB3_skip          # Test specific config
    python3 test_single_config.py --list             # List all configs
"""

import sys
from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import ALL_CONFIGS, list_all_configs


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("Available configs:")
            for name in list_all_configs():
                info = ALL_CONFIGS[name]
                print(f"  {name:25} - {info['description']}")
            return

        config_name = sys.argv[1]
    else:
        config_name = "W1_low_gate"  # Default for testing

    if config_name not in ALL_CONFIGS:
        print(f"ERROR: Unknown config '{config_name}'")
        print(f"Available: {list_all_configs()}")
        return

    print(f"Testing single config: {config_name}")
    print("=" * 70)

    # Load data
    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    # Run validation
    result = run_config_validation(config_name, cache, verbose=True)

    print("\nTest complete!")
    print(f"  Config: {result.config_name}")
    print(f"  Delta: {result.delta_pct:+.2f}%")
    print(f"  p-value: {result.p_value:.6f}")
    print(f"  Significant: {result.significant}")


if __name__ == "__main__":
    main()
