#!/usr/bin/env python3
"""
Experiment: Extended n_shot - Many-Shot Prompting Experiments

Validates extended n_shot configurations from 20 to 200 examples.
Tests: 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200

These extend the original Wave 5 prompting experiments (0, 3, 10 shots)
to explore if more examples in the prompt improve generation quality.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_runner import load_data, EmbeddingCache, run_config_validation
from config_definitions import WAVE5_EXTENDED_NSHOT

# All configs sorted by n_shot value
ALL_CONFIGS = sorted(WAVE5_EXTENDED_NSHOT.keys(),
                     key=lambda x: WAVE5_EXTENDED_NSHOT[x]["overrides"]["n_shot"])

# Already completed configs (from previous run)
COMPLETED = {"W5_shot_20", "W5_shot_50", "W5_shot_100", "W5_shot_200"}

# New configs to run
NEW_CONFIGS = [c for c in ALL_CONFIGS if c not in COMPLETED]


def main():
    parser = argparse.ArgumentParser(description="Extended n_shot experiments")
    parser.add_argument("--all", action="store_true", help="Run all configs (including already completed)")
    parser.add_argument("--configs", nargs="+", help="Specific configs to run")
    args = parser.parse_args()

    if args.configs:
        configs_to_run = args.configs
    elif args.all:
        configs_to_run = ALL_CONFIGS
    else:
        configs_to_run = NEW_CONFIGS

    print("=" * 70)
    print("Experiment: Extended n_shot - Many-Shot Prompting")
    print("=" * 70)
    print(f"\nConfigs to run: {len(configs_to_run)}")
    n_shots = [WAVE5_EXTENDED_NSHOT[c]["overrides"]["n_shot"] for c in configs_to_run]
    print(f"n_shot values: {sorted(n_shots)}")
    print()

    texts, labels = load_data()
    cache = EmbeddingCache()
    cache.load_or_compute(texts, labels)

    results = {}

    for config_name in configs_to_run:
        try:
            n_shot = WAVE5_EXTENDED_NSHOT[config_name]["overrides"]["n_shot"]
            print(f"\n{'='*70}")
            print(f"Running {config_name} (n_shot={n_shot})")
            print(f"{'='*70}")

            result = run_config_validation(config_name, cache, verbose=True)
            results[config_name] = result
        except Exception as e:
            print(f"\nERROR validating {config_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXTENDED N_SHOT EXPERIMENT SUMMARY")
    print("=" * 70)
    print("\nReference: n_shot=0: +0.37 pp, n_shot=3: +1.09 pp, n_shot=10: +1.22 pp")
    print("Previous:  n_shot=20: +1.27 pp, n_shot=50: +1.20 pp, n_shot=100: +1.36 pp, n_shot=200: +1.12 pp")
    print()

    for config_name, result in sorted(results.items(), key=lambda x: WAVE5_EXTENDED_NSHOT[x[0]]["overrides"]["n_shot"]):
        n_shot = WAVE5_EXTENDED_NSHOT[config_name]["overrides"]["n_shot"]
        sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
        delta_pp = result.delta_mean * 100  # Convert to percentage points
        print(f"  n_shot={n_shot:3d}: delta={delta_pp:+.2f} pp (rel: {result.delta_pct:+.2f}%) p={result.p_value:.6f} {sig}")

    print("\n" + "=" * 70)
    print("Key insights:")
    if results:
        best_config = max(results.items(), key=lambda x: x[1].delta_mean)
        n_shot_best = WAVE5_EXTENDED_NSHOT[best_config[0]]["overrides"]["n_shot"]
        delta_pp_best = best_config[1].delta_mean * 100
        print(f"  Best n_shot value (this run): {n_shot_best} with delta={delta_pp_best:+.2f} pp")


if __name__ == "__main__":
    main()
