#!/usr/bin/env python3
"""
Create and evaluate cross-phase ensembles.
Combines synthetics from Phase F and Phase G configs.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib

# Paths
PHASE_F_RESULTS = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_f/results")
PHASE_G_RESULTS = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g/results")
OUTPUT_DIR = PHASE_G_RESULTS / "ensembles_v2"

# Ensemble definitions
ENSEMBLES = {
    "ENS_W9_EXP7": {
        "components": [
            ("W9_contrastive", PHASE_G_RESULTS),
            ("EXP7_hybrid_best", PHASE_F_RESULTS),
        ],
        "description": "Best Phase G individual + Best Phase F individual"
    },
    "ENS_TopG3": {
        "components": [
            ("W9_contrastive", PHASE_G_RESULTS),
            ("W1_low_gate", PHASE_G_RESULTS),
            ("CF1_conf_band", PHASE_G_RESULTS),
        ],
        "description": "Top 3 Phase G configs"
    },
    "ENS_ENTJ_Protect": {
        "components": [
            ("W1_force_problem", PHASE_G_RESULTS),
            ("W3_no_dedup", PHASE_G_RESULTS),
            ("EXP8_intj_protect", PHASE_F_RESULTS),
        ],
        "description": "Configs that protect ENTJ"
    },
    "ENS_HighVol_Safe": {
        "components": [
            ("W2_ultra_vol", PHASE_G_RESULTS),
            ("W1_force_problem", PHASE_G_RESULTS),
        ],
        "description": "High ISFJ boost + ENTJ protection"
    },
    "ENS_MegaMix": {
        "components": [
            ("W9_contrastive", PHASE_G_RESULTS),
            ("W1_low_gate", PHASE_G_RESULTS),
            ("G5_K25_medium", PHASE_G_RESULTS),
            ("EXP7_hybrid_best", PHASE_F_RESULTS),
        ],
        "description": "4-way ensemble: best of each strategy"
    },
    "ENS_TopG5_Extended": {
        "components": [
            ("ENS_Top3_G5", PHASE_G_RESULTS),
            ("W9_contrastive", PHASE_G_RESULTS),
            ("W1_low_gate", PHASE_G_RESULTS),
        ],
        "description": "Extend ENS_Top3_G5 with new Phase G winners"
    },
}


def load_synth_csv(config_name: str, base_path: Path, seed: int = 42) -> pd.DataFrame:
    """Load synthetic CSV file."""
    synth_file = base_path / f"{config_name}_s{seed}_synth.csv"
    if not synth_file.exists():
        raise FileNotFoundError(f"Synth file not found: {synth_file}")

    df = pd.read_csv(synth_file)
    df['source_config'] = config_name
    return df


def compute_text_hash(text: str) -> str:
    """Compute hash of text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def create_ensemble(
    ensemble_name: str,
    components: List[Tuple[str, Path]],
    seed: int = 42,
    dedup: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create ensemble by combining synthetics from multiple configs.

    Args:
        ensemble_name: Name for the ensemble
        components: List of (config_name, base_path) tuples
        seed: Random seed used
        dedup: Whether to deduplicate based on text hash

    Returns:
        Combined DataFrame and metadata dict
    """
    dfs = []
    component_counts = {}

    for config_name, base_path in components:
        try:
            df = load_synth_csv(config_name, base_path, seed)
            component_counts[config_name] = len(df)
            dfs.append(df)
            print(f"  Loaded {config_name}: {len(df)} synthetics")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

    if not dfs:
        raise ValueError(f"No synthetic files found for ensemble {ensemble_name}")

    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    total_before_dedup = len(combined)

    # Deduplicate based on text content
    if dedup and 'text' in combined.columns:
        combined['_hash'] = combined['text'].apply(compute_text_hash)
        combined = combined.drop_duplicates(subset=['_hash'])
        combined = combined.drop(columns=['_hash'])

    total_after_dedup = len(combined)
    duplicates_removed = total_before_dedup - total_after_dedup

    # Class distribution after dedup
    class_dist = combined['label'].value_counts().to_dict()

    metadata = {
        'ensemble_name': ensemble_name,
        'components': [c[0] for c in components],
        'component_counts': component_counts,
        'total_before_dedup': total_before_dedup,
        'total_after_dedup': total_after_dedup,
        'duplicates_removed': duplicates_removed,
        'class_distribution': class_dist,
        'seed': seed,
    }

    return combined, metadata


def save_ensemble(
    ensemble_name: str,
    df: pd.DataFrame,
    metadata: Dict,
    output_dir: Path
):
    """Save ensemble CSV and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV (only text and label columns for compatibility)
    csv_cols = ['text', 'label']
    if 'source_config' in df.columns:
        csv_cols.append('source_config')

    csv_path = output_dir / f"{ensemble_name}_s{metadata['seed']}_synth.csv"
    df[csv_cols].to_csv(csv_path, index=False)

    # Save metadata
    meta_path = output_dir / f"{ensemble_name}_s{metadata['seed']}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {csv_path.name} ({len(df)} synthetics)")
    return csv_path


def main():
    print("=" * 70)
    print("Creating Cross-Phase Ensembles")
    print("=" * 70)

    results = []

    for ens_name, ens_config in ENSEMBLES.items():
        print(f"\n{ens_name}:")
        print(f"  {ens_config['description']}")

        try:
            df, metadata = create_ensemble(
                ens_name,
                ens_config['components'],
                seed=42,
                dedup=True
            )

            csv_path = save_ensemble(ens_name, df, metadata, OUTPUT_DIR)

            results.append({
                'name': ens_name,
                'total': metadata['total_after_dedup'],
                'dedup': metadata['duplicates_removed'],
                'components': len(ens_config['components']),
                'path': str(csv_path),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Ensemble':<25} {'Synth':>6} {'Dedup':>6} {'Components':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<25} {r['total']:>6} {r['dedup']:>6} {r['components']:>10}")

    print(f"\nEnsembles saved to: {OUTPUT_DIR}")
    print("\nNext step: Run K-fold evaluation on these ensembles")
    print("  python3 kfold_multimodel.py --synth-dir ensembles_v2 --k 5 --repeats 3")


if __name__ == "__main__":
    main()
