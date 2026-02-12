#!/usr/bin/env python3
"""
Geometric Filter for LLM-Generated Sample Quality Control

Implements LOF (Local Outlier Factor) based filtering to detect
and remove synthetic samples that are out-of-distribution.

Key Finding from Geometric Analysis:
- LOF score has r=0.923 correlation with F1 improvement (p=0.0011)
- Synthetics with LOF > 0 are well-integrated in the distribution
- Synthetics with LOF < 0 are likely outliers that hurt performance

Usage:
    from core.geometric_filter import LOFFilter

    lof_filter = LOFFilter(n_neighbors=20, threshold=0.0)
    filtered_emb, mask, scores = lof_filter.filter(
        synthetic_embeddings=llm_embeddings,
        real_embeddings=original_embeddings,
        real_labels=labels,
        target_class="INTJ"
    )
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.neighbors import LocalOutlierFactor


class LOFFilter:
    """
    Local Outlier Factor based filter for synthetic samples.

    LOF measures the local density deviation of a point with respect
    to its neighbors. Points with substantially lower density than
    their neighbors are considered outliers.

    Attributes:
        n_neighbors: Number of neighbors for LOF calculation
        threshold: LOF score threshold (keep samples with score > threshold)
        contamination: Expected proportion of outliers in training data
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        threshold: float = 0.0,
        contamination: float = 0.1
    ):
        """
        Initialize LOF filter.

        Args:
            n_neighbors: Number of neighbors for LOF (default 20)
            threshold: Score threshold for filtering (default 0.0)
                       Samples with LOF score > threshold are kept
                       LOF scores > 0 indicate inliers
                       LOF scores < 0 indicate outliers
            contamination: Expected outlier proportion in original data
        """
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.contamination = contamination

    def filter(
        self,
        synthetic_embeddings: np.ndarray,
        real_embeddings: np.ndarray,
        real_labels: np.ndarray,
        target_class: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter synthetic samples using LOF.

        Args:
            synthetic_embeddings: (N, D) synthetic sample embeddings
            real_embeddings: (M, D) all real embeddings
            real_labels: (M,) labels for real embeddings
            target_class: Target class for filtering

        Returns:
            filtered_embeddings: (K, D) filtered synthetic embeddings
            keep_mask: (N,) boolean mask of kept samples
            lof_scores: (N,) LOF scores for all synthetic samples
        """
        if len(synthetic_embeddings) == 0:
            return np.array([]).reshape(0, real_embeddings.shape[1]), \
                   np.array([], dtype=bool), np.array([])

        # Get real samples for target class
        class_mask = real_labels == target_class
        class_embeddings = real_embeddings[class_mask]

        if len(class_embeddings) < self.n_neighbors + 1:
            # Not enough real samples for LOF, return all
            return synthetic_embeddings, \
                   np.ones(len(synthetic_embeddings), dtype=bool), \
                   np.zeros(len(synthetic_embeddings))

        # Train LOF on real class samples
        n_neighbors = min(self.n_neighbors, len(class_embeddings) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination=self.contamination
        )
        lof.fit(class_embeddings)

        # Get LOF scores for synthetic samples
        # decision_function returns negative values for outliers
        lof_scores = lof.decision_function(synthetic_embeddings)

        # Create keep mask based on threshold
        keep_mask = lof_scores > self.threshold

        # Filter embeddings
        filtered_embeddings = synthetic_embeddings[keep_mask]

        return filtered_embeddings, keep_mask, lof_scores

    def filter_by_class(
        self,
        synthetic_embeddings: np.ndarray,
        synthetic_labels: np.ndarray,
        real_embeddings: np.ndarray,
        real_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict]]:
        """
        Filter synthetic samples for all classes.

        Args:
            synthetic_embeddings: (N, D) all synthetic embeddings
            synthetic_labels: (N,) labels for synthetic samples
            real_embeddings: (M, D) all real embeddings
            real_labels: (M,) labels for real embeddings

        Returns:
            filtered_embeddings: Combined filtered embeddings
            filtered_labels: Labels for filtered embeddings
            stats: Per-class filtering statistics
        """
        unique_classes = np.unique(synthetic_labels)

        all_filtered_embeddings = []
        all_filtered_labels = []
        stats = {}

        for target_class in unique_classes:
            # Get synthetic samples for this class
            synth_mask = synthetic_labels == target_class
            synth_emb = synthetic_embeddings[synth_mask]

            if len(synth_emb) == 0:
                stats[target_class] = {
                    'n_original': 0,
                    'n_filtered': 0,
                    'pct_kept': 0,
                    'mean_lof_score': np.nan
                }
                continue

            # Filter
            filtered_emb, keep_mask, lof_scores = self.filter(
                synth_emb, real_embeddings, real_labels, target_class
            )

            # Record stats
            stats[target_class] = {
                'n_original': len(synth_emb),
                'n_filtered': len(filtered_emb),
                'pct_kept': 100 * len(filtered_emb) / len(synth_emb),
                'mean_lof_score': float(lof_scores.mean()),
                'std_lof_score': float(lof_scores.std()),
                'min_lof_score': float(lof_scores.min()),
                'max_lof_score': float(lof_scores.max()),
                'pct_outliers': 100 * (~keep_mask).sum() / len(keep_mask)
            }

            if len(filtered_emb) > 0:
                all_filtered_embeddings.append(filtered_emb)
                all_filtered_labels.extend([target_class] * len(filtered_emb))

        # Combine
        if all_filtered_embeddings:
            filtered_embeddings = np.vstack(all_filtered_embeddings)
            filtered_labels = np.array(all_filtered_labels)
        else:
            filtered_embeddings = np.array([]).reshape(0, real_embeddings.shape[1])
            filtered_labels = np.array([])

        return filtered_embeddings, filtered_labels, stats


class CombinedGeometricFilter:
    """
    Combined filter using LOF + cosine similarity.

    This filter applies multiple geometric criteria:
    1. LOF score (must be > threshold for inlier status)
    2. Cosine similarity to centroid (must be > sim_threshold)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        lof_threshold: float = 0.0,
        sim_threshold: float = 0.5,
        contamination: float = 0.1
    ):
        """
        Initialize combined filter.

        Args:
            n_neighbors: Number of neighbors for LOF
            lof_threshold: LOF score threshold (default 0.0)
            sim_threshold: Cosine similarity threshold (default 0.5)
            contamination: Expected outlier proportion
        """
        self.lof_filter = LOFFilter(n_neighbors, lof_threshold, contamination)
        self.sim_threshold = sim_threshold

    def filter(
        self,
        synthetic_embeddings: np.ndarray,
        real_embeddings: np.ndarray,
        real_labels: np.ndarray,
        target_class: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Filter using combined criteria.

        Returns samples that pass BOTH:
        - LOF score > lof_threshold
        - Cosine similarity to centroid > sim_threshold
        """
        if len(synthetic_embeddings) == 0:
            return np.array([]).reshape(0, real_embeddings.shape[1]), \
                   np.array([], dtype=bool), {}

        # Get class embeddings
        class_mask = real_labels == target_class
        class_embeddings = real_embeddings[class_mask]

        if len(class_embeddings) == 0:
            return synthetic_embeddings, \
                   np.ones(len(synthetic_embeddings), dtype=bool), {}

        # Calculate centroid
        centroid = class_embeddings.mean(axis=0, keepdims=True)

        # Calculate cosine similarity to centroid
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(synthetic_embeddings, centroid).flatten()

        # Apply LOF filter
        _, lof_mask, lof_scores = self.lof_filter.filter(
            synthetic_embeddings, real_embeddings, real_labels, target_class
        )

        # Apply similarity filter
        sim_mask = similarities > self.sim_threshold

        # Combine masks (must pass both)
        combined_mask = lof_mask & sim_mask

        filtered_embeddings = synthetic_embeddings[combined_mask]

        stats = {
            'n_original': len(synthetic_embeddings),
            'n_pass_lof': lof_mask.sum(),
            'n_pass_sim': sim_mask.sum(),
            'n_pass_both': combined_mask.sum(),
            'pct_kept': 100 * combined_mask.sum() / len(synthetic_embeddings),
            'mean_lof_score': float(lof_scores.mean()),
            'mean_similarity': float(similarities.mean())
        }

        return filtered_embeddings, combined_mask, stats


def test_lof_filter():
    """Test LOF filter with synthetic data."""
    np.random.seed(42)

    # Create mock data
    n_real = 100
    n_synthetic = 30
    n_dims = 768

    # Real embeddings for one class
    real_center = np.random.randn(n_dims)
    real_center = real_center / np.linalg.norm(real_center)

    real_embeddings = real_center + 0.1 * np.random.randn(n_real, n_dims)
    real_labels = np.array(["INTJ"] * n_real)

    # Synthetic embeddings: some good (near center), some bad (far from center)
    good_synthetic = real_center + 0.1 * np.random.randn(20, n_dims)  # Near
    bad_synthetic = 2 * np.random.randn(10, n_dims)  # Far

    synthetic_embeddings = np.vstack([good_synthetic, bad_synthetic])
    synthetic_labels = np.array(["INTJ"] * 30)

    print("=" * 60)
    print("LOF FILTER TEST")
    print("=" * 60)

    # Test with different thresholds
    for threshold in [-1.0, -0.5, 0.0, 0.5]:
        print(f"\nThreshold: {threshold}")

        lof_filter = LOFFilter(n_neighbors=10, threshold=threshold)
        filtered, mask, scores = lof_filter.filter(
            synthetic_embeddings, real_embeddings, real_labels, "INTJ"
        )

        n_good_kept = mask[:20].sum()
        n_bad_kept = mask[20:].sum()

        print(f"  Total: {len(synthetic_embeddings)} -> {len(filtered)}")
        print(f"  Good samples kept: {n_good_kept}/20")
        print(f"  Bad samples kept: {n_bad_kept}/10")
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test filter_by_class
    print("\n" + "=" * 60)
    print("FILTER BY CLASS TEST")
    print("=" * 60)

    lof_filter = LOFFilter(n_neighbors=10, threshold=0.0)
    filtered_emb, filtered_labels, stats = lof_filter.filter_by_class(
        synthetic_embeddings, synthetic_labels,
        real_embeddings, real_labels
    )

    print(f"\nFiltered: {len(filtered_emb)} samples")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    test_lof_filter()
