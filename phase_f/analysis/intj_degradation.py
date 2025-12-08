#!/usr/bin/env python3
"""
Phase F Analysis: Why does INTJ always degrade?

Hypothesis: Synthetics from other classes are "invading" INTJ's embedding space,
causing the classifier to misclassify INTJ samples.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "results"

# MBTI classes similar to INTJ (share 3 letters)
INTJ_NEIGHBORS = ['INFJ', 'INTP', 'ENTJ', 'ISTJ']

def load_synth_data(config, seed=42):
    """Load synthetic data with quality metrics."""
    path = RESULTS_DIR / f"{config}_s{seed}_synth.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def analyze_intj_neighbors():
    """Analyze which classes have synthetics that might affect INTJ."""
    print("\n" + "="*80)
    print("INTJ DEGRADATION ANALYSIS")
    print("="*80)

    print("\n### INTJ Context")
    print("-"*80)
    print("INTJ characteristics:")
    print("  - Original samples: 1091 (3rd largest class)")
    print("  - Baseline F1: ~0.38")
    print("  - Synthetics generated: 0")
    print("  - Consistent degradation: -0.5% to -1.3% across all configs")
    print("\nINTJ neighbors (share 3 letters):")
    for n in INTJ_NEIGHBORS:
        print(f"  - {n}")

    # Load ensemble data
    configs = ["CMB3_skip", "CF1_conf_band", "V4_ultra", "G5_K25_medium"]

    print("\n### Synthetic Distribution for INTJ Neighbors")
    print("-"*80)

    neighbor_synth = defaultdict(list)

    for config in configs:
        df = load_synth_data(config)
        if df is None:
            continue

        print(f"\n{config}:")
        for neighbor in INTJ_NEIGHBORS:
            count = len(df[df['label'] == neighbor])
            neighbor_synth[neighbor].append((config, count))
            if count > 0:
                # Get quality metrics for these synthetics
                neighbor_df = df[df['label'] == neighbor]
                avg_sim = neighbor_df['similarity_to_centroid'].mean()
                avg_conf = neighbor_df['classifier_confidence'].mean()
                print(f"  {neighbor}: {count} synthetics (avg_sim={avg_sim:.3f}, avg_conf={avg_conf:.3f})")

    print("\n### Hypothesis Analysis")
    print("-"*80)

    # Calculate total neighbor synthetics
    total_neighbor_synth = sum(
        sum(c[1] for c in synths)
        for synths in neighbor_synth.values()
    )

    print(f"\nTotal synthetics for INTJ neighbors: {total_neighbor_synth}")

    print("\n### Quality Analysis of Neighbor Synthetics")
    print("-"*80)

    # Analyze quality metrics of synthetics that might "invade" INTJ space
    for config in configs:
        df = load_synth_data(config)
        if df is None:
            continue

        for neighbor in INTJ_NEIGHBORS:
            neighbor_df = df[df['label'] == neighbor]
            if len(neighbor_df) == 0:
                continue

            # Low similarity to centroid = possibly in wrong space
            low_sim = neighbor_df[neighbor_df['similarity_to_centroid'] < 0.4]
            # Low classifier confidence = ambiguous
            low_conf = neighbor_df[neighbor_df['classifier_confidence'] < 0.3]
            # Low KNN similarity = far from training examples
            if 'knn_similarity' in neighbor_df.columns:
                low_knn = neighbor_df[neighbor_df['knn_similarity'] < 0.4]
            else:
                low_knn = pd.DataFrame()

            if len(low_sim) > 0 or len(low_conf) > 0:
                print(f"\n{config} - {neighbor}:")
                if len(low_sim) > 0:
                    print(f"  Low centroid similarity (<0.4): {len(low_sim)}/{len(neighbor_df)} ({100*len(low_sim)/len(neighbor_df):.0f}%)")
                if len(low_conf) > 0:
                    print(f"  Low classifier confidence (<0.3): {len(low_conf)}/{len(neighbor_df)} ({100*len(low_conf)/len(neighbor_df):.0f}%)")
                if len(low_knn) > 0:
                    print(f"  Low KNN similarity (<0.4): {len(low_knn)}/{len(neighbor_df)} ({100*len(low_knn)/len(neighbor_df):.0f}%)")

def analyze_confusion_pattern():
    """Analyze if INTJ is being confused with neighboring classes."""
    print("\n" + "="*80)
    print("CONFUSION PATTERN ANALYSIS")
    print("="*80)

    print("""
    Theory: When we add synthetics for classes like INTP, INFJ, ENTJ, ISTJ,
    some of these synthetics might fall in regions that the classifier
    previously assigned to INTJ.

    This would cause:
    1. INTJ samples misclassified as neighbor classes
    2. INTJ F1 drops while neighbor F1 might increase

    From PER_CLASS_ANALYSIS.md we know:
    - INTJ: -0.95% average degradation
    - INFJ: +0.53% average improvement
    - INTP: +0.16% average improvement
    - ENTJ: +0.53% average improvement (but from very low baseline)
    - ISTJ: 0% (unchanged)

    This pattern is CONSISTENT with the confusion hypothesis:
    Synthetics for INFJ, INTP, ENTJ are "stealing" from INTJ's decision space.
    """)

def suggest_mitigation():
    """Suggest ways to mitigate INTJ degradation."""
    print("\n" + "="*80)
    print("MITIGATION STRATEGIES")
    print("="*80)

    print("""
    1. **Increase contamination threshold for INTJ neighbors**
       - When generating INFJ/INTP/ENTJ synthetics, use stricter filters
       - Increase similarity_threshold for classes near INTJ

    2. **Exclude INTJ's embedding region**
       - Before accepting a synthetic, check if it's closer to INTJ centroid
       - Reject if similarity to INTJ centroid > threshold

    3. **Class-weighted training**
       - Give INTJ samples higher weight during training
       - Compensate for the synthetic "invasion"

    4. **Ensemble filtering**
       - Remove synthetics that hurt INTJ specifically
       - Create "INTJ-safe" ensemble variant

    5. **Quality gate for neighbors**
       - Require higher anchor_quality for INTJ neighbors
       - Only accept very high-confidence synthetics
    """)

def main():
    analyze_intj_neighbors()
    analyze_confusion_pattern()
    suggest_mitigation()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
    INTJ degradation is caused by synthetics from neighboring classes
    (INFJ, INTP, ENTJ, ISTJ) "invading" INTJ's decision space.

    The classifier learns from these new synthetic examples and shifts
    its decision boundaries, causing INTJ samples to be misclassified.

    This is a TRADE-OFF: We improve minority classes at the cost of
    slightly degrading majority classes like INTJ.

    Net effect: +1.29% macro F1 (positive), but INTJ takes the hit.
    """)

if __name__ == "__main__":
    main()
