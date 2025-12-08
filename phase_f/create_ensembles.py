#!/usr/bin/env python3
"""
Create Ensemble Synthetic Datasets for Phase F

This script combines synthetic data from multiple configurations
to create ensemble datasets for evaluation.

Usage:
    python3 create_ensembles.py

Output:
    results/ENS_*_s42_synth.csv files
"""

import pandas as pd
import os

BASE_DIR = "results"
SEED = 42

def load_synth(config: str, seed: int = SEED) -> pd.DataFrame:
    """Load synthetic data for a configuration."""
    path = f"{BASE_DIR}/{config}_s{seed}_synth.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded {config}: {len(df)} samples")
        return df
    else:
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()

def create_ensemble(name: str, configs: list, seed: int = SEED) -> None:
    """Create an ensemble by combining synthetic data from multiple configs."""
    print(f"\nCreating {name}:")
    print(f"  Components: {' + '.join(configs)}")

    # Load all components
    dfs = [load_synth(cfg, seed) for cfg in configs]
    dfs = [df for df in dfs if len(df) > 0]

    if not dfs:
        print(f"  ERROR: No data to combine")
        return

    # Combine
    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates if any (based on text)
    text_col = 'text' if 'text' in combined.columns else 'posts'
    initial_len = len(combined)
    combined = combined.drop_duplicates(subset=[text_col])

    if len(combined) < initial_len:
        print(f"  Removed {initial_len - len(combined)} duplicates")

    # Save
    out_path = f"{BASE_DIR}/{name}_s{seed}_synth.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Total samples: {len(combined)}")

def main():
    print("=" * 60)
    print("Phase F: Ensemble Creation Script")
    print("=" * 60)

    # Define ensembles
    ensembles = {
        # Two-config ensembles
        "ENS_CMB3_V2": ["CMB3_skip", "V2_high_vol"],
        "ENS_CMB3_CF1": ["CMB3_skip", "CF1_conf_band"],
        "ENS_CMB3_G5": ["CMB3_skip", "G5_K25_medium"],

        # Three-config ensemble
        "ENS_Top3": ["CMB3_skip", "CF1_conf_band", "V4_ultra"],

        # Four-config ensemble (best performer)
        "ENS_Top3_G5": ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium"],
    }

    # Create each ensemble
    for name, configs in ensembles.items():
        create_ensemble(name, configs)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Summary table
    print(f"\n{'Ensemble':<20} {'Samples':>10}")
    print("-" * 32)
    for name in ensembles:
        path = f"{BASE_DIR}/{name}_s{SEED}_synth.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"{name:<20} {len(df):>10}")

    print("\nTo evaluate, run:")
    print("  python3 kfold_evaluator.py --config ENS_Top3_G5 --seed 42 --k 5 --repeated 3")

if __name__ == "__main__":
    main()
