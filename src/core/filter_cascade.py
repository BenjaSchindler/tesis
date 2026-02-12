#!/usr/bin/env python3
"""
Filter Cascade for LLM-Generated Sample Quality Control

Implements a 4-level filter cascade to rank and select the best
LLM-generated samples based on embedding quality metrics.

Filter Levels:
1. length/distance - Euclidean distance from anchor (closer = better)
2. similarity - Cosine similarity to anchor
3. knn - K-NN purity (inverse avg distance to k nearest same-class neighbors)
4. confidence - Distance to class centroid

Quality Score: Geometric mean of active filter scores
Selection: Top-N by ranking (no hard threshold)

Usage:
    from core.filter_cascade import FilterCascade

    cascade = FilterCascade(filter_level=4)
    filtered_emb, avg_quality = cascade.filter_samples(
        candidates=llm_embeddings,
        real_embeddings=cache.embeddings,
        real_labels=cache.labels,
        target_class="INTJ",
        target_count=100
    )
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import cdist


class FilterCascade:
    """
    Multi-level filter cascade for LLM sample quality control.

    Attributes:
        filter_level: Number of filters to apply (1-4)
        k_neighbors: K for K-NN purity calculation
    """

    FILTER_NAMES = ["distance", "similarity", "knn", "confidence"]

    def __init__(
        self,
        filter_level: int = 4,
        k_neighbors: int = 10
    ):
        """
        Initialize filter cascade.

        Args:
            filter_level: Number of filters (1-4)
                1 = distance only
                2 = distance + similarity
                3 = distance + similarity + knn
                4 = full cascade (all 4 filters)
            k_neighbors: K for K-NN purity calculation
        """
        self.filter_level = min(max(filter_level, 0), 4)
        self.k_neighbors = k_neighbors

    def compute_quality_scores(
        self,
        candidates: np.ndarray,
        anchor_emb: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute composite quality score for each candidate.

        Args:
            candidates: (N, D) LLM-generated embeddings
            anchor_emb: (D,) or (1, D) class anchor embedding
            all_embeddings: (M, D) all real embeddings
            all_labels: (M,) labels for real embeddings
            target_class: Target MBTI class

        Returns:
            (composite_scores, individual_scores_dict)
        """
        if len(candidates) == 0:
            return np.array([]), {}

        n_candidates = len(candidates)

        # Ensure anchor is 2D
        if anchor_emb.ndim == 1:
            anchor_emb = anchor_emb.reshape(1, -1)

        scores_dict = {}
        active_filters = self.FILTER_NAMES[:self.filter_level]

        # Filter 1: Distance from anchor (closer = better)
        if "distance" in active_filters:
            dists_to_anchor = np.linalg.norm(candidates - anchor_emb, axis=1)
            max_dist = np.max(dists_to_anchor) + 1e-6
            scores_dict["distance"] = 1 - (dists_to_anchor / max_dist)

        # Filter 2: Cosine similarity to anchor
        if "similarity" in active_filters:
            similarities = 1 - cdist(candidates, anchor_emb, metric='cosine').flatten()
            scores_dict["similarity"] = np.clip(similarities, 0, 1)

        # Filter 3: K-NN purity
        if "knn" in active_filters:
            class_mask = all_labels == target_class
            class_embs = all_embeddings[class_mask]

            knn_scores = np.zeros(n_candidates)
            if len(class_embs) > 0:
                k = min(self.k_neighbors, len(class_embs))
                for i, cand in enumerate(candidates):
                    dists = np.linalg.norm(class_embs - cand, axis=1)
                    nearest_dists = np.sort(dists)[:k]
                    # Inverse of average distance (closer = better)
                    knn_scores[i] = 1.0 / (1.0 + nearest_dists.mean())
            # Normalize to 0-1
            if knn_scores.max() > 0:
                knn_scores = knn_scores / knn_scores.max()
            scores_dict["knn"] = knn_scores

        # Filter 4: Confidence (distance to class centroid)
        if "confidence" in active_filters:
            class_mask = all_labels == target_class
            class_embs = all_embeddings[class_mask]

            if len(class_embs) > 0:
                centroid = class_embs.mean(axis=0)
                dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
                max_dist = np.max(dists_to_centroid) + 1e-6
                scores_dict["confidence"] = 1 - (dists_to_centroid / max_dist)
            else:
                scores_dict["confidence"] = np.ones(n_candidates) * 0.5

        # Combine scores: geometric mean
        if not scores_dict:
            return np.ones(n_candidates), {}

        combined = np.ones(n_candidates)
        for score_array in scores_dict.values():
            combined *= score_array

        # Take nth root where n = number of filters
        n_filters = len(scores_dict)
        combined = np.power(combined, 1.0 / n_filters)

        return combined, scores_dict

    def select_top_by_ranking(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        target: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Select top candidates by quality score ranking.

        Args:
            candidates: (N, D) candidate embeddings
            scores: (N,) quality scores
            target: Number to select

        Returns:
            (selected_embeddings, selected_indices, avg_quality)
        """
        if len(candidates) == 0:
            return np.array([]).reshape(0, candidates.shape[1] if len(candidates) > 0 else 768), \
                   np.array([]), 0.0

        # Select top-N by score
        n_select = min(target, len(candidates))
        top_idx = np.argsort(scores)[-n_select:]

        selected = candidates[top_idx]
        avg_quality = scores[top_idx].mean()

        return selected, top_idx, avg_quality

    def filter_samples(
        self,
        candidates: np.ndarray,
        real_embeddings: np.ndarray,
        real_labels: np.ndarray,
        target_class: str,
        target_count: int,
        anchor_emb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Filter LLM-generated samples using the cascade.

        Args:
            candidates: (N, D) LLM-generated embeddings
            real_embeddings: (M, D) real data embeddings
            real_labels: (M,) labels for real data
            target_class: Target MBTI class
            target_count: Number of samples to select
            anchor_emb: Optional anchor (defaults to class centroid)

        Returns:
            (filtered_embeddings, avg_quality, details)
        """
        if len(candidates) == 0:
            return np.array([]).reshape(0, real_embeddings.shape[1]), 0.0, {}

        # Use class centroid as anchor if not provided
        if anchor_emb is None:
            class_mask = real_labels == target_class
            class_embs = real_embeddings[class_mask]
            if len(class_embs) > 0:
                anchor_emb = class_embs.mean(axis=0)
            else:
                anchor_emb = real_embeddings.mean(axis=0)

        # Compute quality scores
        scores, score_components = self.compute_quality_scores(
            candidates, anchor_emb, real_embeddings, real_labels, target_class
        )

        # Select top by ranking
        filtered, indices, avg_quality = self.select_top_by_ranking(
            candidates, scores, target_count
        )

        details = {
            "n_candidates": len(candidates),
            "n_selected": len(filtered),
            "avg_quality": avg_quality,
            "filter_level": self.filter_level,
            "score_components": {k: float(v[indices].mean()) for k, v in score_components.items()} if len(indices) > 0 else {},
            "quality_distribution": {
                "min": float(scores.min()) if len(scores) > 0 else 0,
                "max": float(scores.max()) if len(scores) > 0 else 0,
                "mean": float(scores.mean()) if len(scores) > 0 else 0,
                "selected_min": float(scores[indices].min()) if len(indices) > 0 else 0,
            }
        }

        return filtered, avg_quality, details


def test_filter_cascade():
    """Test filter cascade with synthetic data."""
    np.random.seed(42)

    # Create mock data
    n_real = 100
    n_candidates = 50
    n_dims = 768

    # Real embeddings (two classes)
    real_embeddings = np.random.randn(n_real, n_dims)
    real_embeddings = real_embeddings / np.linalg.norm(real_embeddings, axis=1, keepdims=True)
    real_labels = np.array(["INTJ"] * 50 + ["ENFP"] * 50)

    # Candidate embeddings (some close to INTJ, some far)
    intj_center = real_embeddings[:50].mean(axis=0)

    # Good candidates (close to INTJ center)
    good_candidates = intj_center + 0.1 * np.random.randn(25, n_dims)
    good_candidates = good_candidates / np.linalg.norm(good_candidates, axis=1, keepdims=True)

    # Bad candidates (far from INTJ center)
    bad_candidates = np.random.randn(25, n_dims)
    bad_candidates = bad_candidates / np.linalg.norm(bad_candidates, axis=1, keepdims=True)

    candidates = np.vstack([good_candidates, bad_candidates])

    print("=" * 60)
    print("FILTER CASCADE TEST")
    print("=" * 60)

    # Test different filter levels
    for level in [0, 1, 2, 3, 4]:
        cascade = FilterCascade(filter_level=level)
        filtered, avg_quality, details = cascade.filter_samples(
            candidates=candidates,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class="INTJ",
            target_count=20
        )

        print(f"\nFilter Level {level}:")
        print(f"  Candidates: {details['n_candidates']}")
        print(f"  Selected: {details['n_selected']}")
        print(f"  Avg Quality: {avg_quality:.3f}")
        print(f"  Score Components: {details['score_components']}")
        print(f"  Quality Range: [{details['quality_distribution']['min']:.3f}, {details['quality_distribution']['max']:.3f}]")


if __name__ == "__main__":
    test_filter_cascade()
