"""
Phase E: Auto-parameter adjustment based on dataset characteristics.

Analyzes the dataset and suggests/applies parameter adjustments for optimal results.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DatasetProfile:
    """Profile of dataset characteristics."""
    total_samples: int
    n_classes: int
    min_class_size: int
    max_class_size: int
    median_class_size: int
    imbalance_ratio: float  # max/min class size
    avg_quality: Optional[float] = None
    avg_purity: Optional[float] = None


@dataclass
class AutoParams:
    """Auto-adjusted parameters."""
    anchor_quality_threshold: float
    cap_class_ratio: float
    similarity_threshold: float
    min_classifier_confidence: float
    max_clusters: Optional[int] = None  # None = don't override
    prompts_per_cluster: Optional[int] = None  # None = don't override
    min_cluster_samples: int = 5  # Purity-aware: min samples per cluster
    reason: str = ""


def analyze_dataset(labels: np.ndarray) -> DatasetProfile:
    """Analyze dataset characteristics."""
    unique, counts = np.unique(labels, return_counts=True)
    
    return DatasetProfile(
        total_samples=len(labels),
        n_classes=len(unique),
        min_class_size=int(counts.min()),
        max_class_size=int(counts.max()),
        median_class_size=int(np.median(counts)),
        imbalance_ratio=float(counts.max() / counts.min()) if counts.min() > 0 else float('inf'),
    )


def compute_auto_params(
    profile: DatasetProfile,
    quality_scores: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> AutoParams:
    """
    Compute auto-adjusted parameters based on dataset profile.
    
    Logic:
    - Small datasets (< 10K): Lower thresholds to allow more generation
    - Medium datasets (10K-50K): Standard thresholds
    - Large datasets (> 50K): Can be more selective
    
    - Low avg quality (< 0.25): Much lower threshold to allow any generation
    - Medium avg quality (0.25-0.35): Slightly lower threshold
    - High avg quality (> 0.35): Standard threshold
    """
    reasons = []
    
    # === DATASET SIZE ADJUSTMENTS ===
    # NOTE: We no longer override max_clusters/prompts_per_cluster here
    # The elbow method in runner handles this adaptively
    # We only adjust quality thresholds based on dataset size

    if profile.total_samples < 10000:
        # Small dataset - be more permissive
        base_quality_th = 0.15
        base_cap_ratio = 0.20
        base_sim_th = 0.85
        base_conf = 0.05
        min_cluster_samples = 5  # Allow smaller clusters for small datasets
        reasons.append(f"Small dataset ({profile.total_samples:,} samples) → permissive thresholds")
    elif profile.total_samples < 50000:
        # Medium dataset
        base_quality_th = 0.25
        base_cap_ratio = 0.15
        base_sim_th = 0.88
        base_conf = 0.08
        min_cluster_samples = 8
        reasons.append(f"Medium dataset ({profile.total_samples:,} samples) → balanced thresholds")
    else:
        # Large dataset - can be selective
        base_quality_th = 0.30
        base_cap_ratio = 0.10
        base_sim_th = 0.90
        base_conf = 0.10
        min_cluster_samples = 10
        reasons.append(f"Large dataset ({profile.total_samples:,} samples) → selective thresholds")
    
    # === CLASS SIZE ADJUSTMENTS ===
    if profile.min_class_size < 50:
        # Very small minority classes - need more permissive generation
        base_quality_th *= 0.7
        base_cap_ratio *= 1.5
        reasons.append(f"Tiny minority class ({profile.min_class_size} samples) → boosted generation")
    elif profile.min_class_size < 200:
        base_quality_th *= 0.85
        base_cap_ratio *= 1.2
        reasons.append(f"Small minority class ({profile.min_class_size} samples) → slightly boosted")
    
    # === IMBALANCE ADJUSTMENTS ===
    if profile.imbalance_ratio > 50:
        base_cap_ratio *= 1.3
        reasons.append(f"High imbalance ({profile.imbalance_ratio:.1f}x) → increased cap ratio")
    
    # === QUALITY-BASED ADJUSTMENTS ===
    if quality_scores:
        avg_quality = np.mean(list(quality_scores.values()))
        profile.avg_quality = avg_quality
        
        if avg_quality < 0.20:
            # Very low quality - accept almost anything
            base_quality_th = min(base_quality_th, avg_quality * 1.2)
            reasons.append(f"Very low avg quality ({avg_quality:.3f}) → minimal threshold")
        elif avg_quality < 0.30:
            # Low quality - lower threshold to percentile
            base_quality_th = min(base_quality_th, avg_quality * 0.9)
            reasons.append(f"Low avg quality ({avg_quality:.3f}) → adjusted threshold")
    
    # Ensure minimums
    anchor_quality_threshold = max(0.10, min(0.35, base_quality_th))
    cap_class_ratio = max(0.05, min(0.30, base_cap_ratio))
    similarity_threshold = max(0.80, min(0.95, base_sim_th))
    min_classifier_confidence = max(0.01, min(0.15, base_conf))
    
    if verbose:
        print("\n" + "="*60)
        print("📊 AUTO-PARAMETER ADJUSTMENT")
        print("="*60)
        print(f"\nDataset Profile:")
        print(f"  Total samples: {profile.total_samples:,}")
        print(f"  Classes: {profile.n_classes}")
        print(f"  Class sizes: {profile.min_class_size} - {profile.max_class_size} (median: {profile.median_class_size})")
        print(f"  Imbalance ratio: {profile.imbalance_ratio:.1f}x")
        if profile.avg_quality:
            print(f"  Avg quality: {profile.avg_quality:.3f}")

        print(f"\nAdjustments applied:")
        for r in reasons:
            print(f"  • {r}")

        print(f"\nAuto-parameters:")
        print(f"  --anchor-quality-threshold {anchor_quality_threshold:.2f}")
        print(f"  --cap-class-ratio {cap_class_ratio:.2f}")
        print(f"  --similarity-threshold {similarity_threshold:.2f}")
        print(f"  --min-classifier-confidence {min_classifier_confidence:.2f}")
        print(f"  --min-cluster-samples {min_cluster_samples}")
        print(f"  (max-clusters/prompts-per-cluster: use CLI args, elbow method adapts)")
        print("="*60 + "\n")

    return AutoParams(
        anchor_quality_threshold=anchor_quality_threshold,
        cap_class_ratio=cap_class_ratio,
        similarity_threshold=similarity_threshold,
        min_classifier_confidence=min_classifier_confidence,
        max_clusters=None,  # Don't override - let CLI args or elbow handle it
        prompts_per_cluster=None,  # Don't override
        min_cluster_samples=min_cluster_samples,
        reason=" | ".join(reasons),
    )


def apply_auto_params(args, auto_params: AutoParams) -> None:
    """Apply auto-parameters to args namespace."""
    args.anchor_quality_threshold = auto_params.anchor_quality_threshold
    args.cap_class_ratio = auto_params.cap_class_ratio
    args.similarity_threshold = auto_params.similarity_threshold
    args.min_classifier_confidence = auto_params.min_classifier_confidence

    # Only override cluster settings if explicitly set (not None)
    if auto_params.max_clusters is not None:
        args.max_clusters = auto_params.max_clusters
    if auto_params.prompts_per_cluster is not None:
        args.prompts_per_cluster = auto_params.prompts_per_cluster

    # Store min_cluster_samples for purity-aware clustering
    args.min_cluster_samples = auto_params.min_cluster_samples


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Simulate small dataset
    labels = np.array(['A']*50 + ['B']*200 + ['C']*500 + ['D']*1000)
    profile = analyze_dataset(labels)
    
    # Simulate quality scores
    quality_scores = {'A': 0.15, 'B': 0.18, 'C': 0.22, 'D': 0.25}
    
    auto_params = compute_auto_params(profile, quality_scores)
    print(f"\nFinal params: {auto_params}")
