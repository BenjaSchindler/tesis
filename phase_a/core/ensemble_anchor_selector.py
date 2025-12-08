"""
Phase 2: Ensemble Anchor Selection

Combines multiple anchor selection strategies to improve cluster purity and reduce
contamination in synthetic data generation.

Author: Benja
Date: 2025-10-30
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist


class EnsembleAnchorSelector:
    """
    Ensemble anchor selection combining multiple strategies:
    1. Medoid (robust, stable)
    2. Quality-gated (high-purity clusters only)
    3. Diverse (max-min distance for coverage)
    """

    def __init__(
        self,
        min_purity: float = 0.60,
        min_quality_score: float = 0.50,
        diversity_weight: float = 0.3,
        quality_weight: float = 0.3,
        hardness_weight: float = 0.2,
        stability_weight: float = 0.2,
        dedup_threshold: float = 0.95
    ):
        """
        Initialize ensemble anchor selector.

        Args:
            min_purity: Minimum purity for quality-gated anchors
            min_quality_score: Minimum composite quality score
            diversity_weight: Weight for diversity in ensemble scoring
            quality_weight: Weight for quality in ensemble scoring
            hardness_weight: Weight for hardness (boundary proximity) in ensemble scoring
            stability_weight: Weight for stability in ensemble scoring
            dedup_threshold: Similarity threshold for deduplication
        """
        self.min_purity = min_purity
        self.min_quality_score = min_quality_score
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.hardness_weight = hardness_weight
        self.stability_weight = stability_weight
        self.dedup_threshold = dedup_threshold

    def select_ensemble_anchors(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        target_class: str,
        k_clusters: int,
        cluster_assignments: Optional[np.ndarray] = None
    ) -> Tuple[List[int], Dict[str, any]]:
        """
        Select anchors using ensemble strategy.

        Args:
            embeddings: (N, D) array of embeddings
            labels: (N,) array of class labels
            target_class: Target class to select anchors for
            k_clusters: Number of clusters/anchors to select
            cluster_assignments: Optional pre-computed cluster assignments

        Returns:
            anchor_indices: List of selected anchor indices
            metrics: Dictionary of ensemble metrics
        """
        # Filter to target class
        mask = labels == target_class
        class_embeddings = embeddings[mask]
        class_indices = np.where(mask)[0]

        if len(class_embeddings) < k_clusters:
            print(f"⚠️  Warning: Only {len(class_embeddings)} samples for {k_clusters} clusters")
            return list(class_indices), {"warning": "insufficient_samples"}

        # Strategy 1: Medoid anchors (stable, robust)
        medoid_anchors = self._select_medoid_anchors(
            class_embeddings, k_clusters, cluster_assignments
        )

        # Strategy 2: Quality-gated anchors (high-purity only)
        quality_anchors = self._select_quality_gated_anchors(
            class_embeddings, embeddings, labels, target_class, k_clusters
        )

        # Strategy 3: Diverse anchors (max-min distance)
        diverse_anchors = self._select_diverse_anchors(
            class_embeddings, k_clusters
        )

        # Strategy 4: Hard anchors (boundary points)
        hard_anchors = self._select_hard_anchors(
            class_embeddings, embeddings, labels, target_class, k_clusters
        )

        # Combine all anchor candidates
        all_anchor_indices = []
        all_anchor_indices.extend([(idx, "medoid") for idx in medoid_anchors])
        all_anchor_indices.extend([(idx, "quality") for idx in quality_anchors])
        all_anchor_indices.extend([(idx, "diverse") for idx in diverse_anchors])
        all_anchor_indices.extend([(idx, "hard") for idx in hard_anchors])

        # Deduplicate by similarity
        unique_anchors = self._deduplicate_anchors(
            class_embeddings, all_anchor_indices
        )

        # Rank by ensemble score
        ranked_anchors = self._rank_by_ensemble_score(
            class_embeddings, embeddings, labels, target_class, unique_anchors
        )

        # Select top k_clusters anchors
        final_anchors = ranked_anchors[:k_clusters]

        # Map back to original indices
        final_anchor_indices = [class_indices[idx] for idx, _ in final_anchors]

        # Calculate metrics
        metrics = self._calculate_ensemble_metrics(
            class_embeddings, embeddings, labels, target_class,
            [idx for idx, _ in final_anchors], medoid_anchors, quality_anchors, diverse_anchors, hard_anchors
        )

        return final_anchor_indices, metrics

    def _select_medoid_anchors(
        self,
        embeddings: np.ndarray,
        k_clusters: int,
        cluster_assignments: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select medoid (most central point) for each cluster.

        Args:
            embeddings: (N, D) embeddings
            k_clusters: Number of clusters
            cluster_assignments: Optional cluster assignments

        Returns:
            List of medoid indices
        """
        from sklearn.cluster import KMeans

        # Cluster if not provided
        if cluster_assignments is None:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(embeddings)

        medoid_indices = []
        for cluster_id in range(k_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = embeddings[cluster_mask]
            cluster_point_indices = np.where(cluster_mask)[0]

            if len(cluster_points) == 0:
                continue

            # Find medoid: point with minimum sum of distances to all others in cluster
            distances = pairwise_distances(cluster_points, metric='cosine')
            medoid_idx_in_cluster = np.argmin(distances.sum(axis=1))
            medoid_idx = cluster_point_indices[medoid_idx_in_cluster]

            medoid_indices.append(medoid_idx)

        return medoid_indices

    def _select_quality_gated_anchors(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        k_clusters: int
    ) -> List[int]:
        """
        Select anchors from high-purity regions only.

        Args:
            class_embeddings: Embeddings of target class
            all_embeddings: All embeddings
            all_labels: All labels
            target_class: Target class
            k_clusters: Number of clusters

        Returns:
            List of quality-gated anchor indices
        """
        from sklearn.cluster import KMeans

        # Cluster the class
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(class_embeddings)

        quality_anchors = []
        for cluster_id in range(k_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = class_embeddings[cluster_mask]
            cluster_point_indices = np.where(cluster_mask)[0]

            if len(cluster_points) == 0:
                continue

            # Calculate purity for this cluster
            # Use cluster centroid to find neighbors in full dataset
            centroid = cluster_points.mean(axis=0, keepdims=True)
            distances = cdist(centroid, all_embeddings, metric='cosine')[0]
            k_neighbors = min(15, len(all_embeddings))
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_labels = all_labels[nearest_indices]

            purity = (nearest_labels == target_class).mean()

            # Only select if purity is high
            if purity >= self.min_purity:
                # Select most central point in high-purity cluster
                cluster_distances = pairwise_distances(cluster_points, metric='cosine')
                centroid_idx_in_cluster = np.argmin(cluster_distances.sum(axis=1))
                centroid_idx = cluster_point_indices[centroid_idx_in_cluster]
                quality_anchors.append(centroid_idx)

        return quality_anchors

    def _select_diverse_anchors(
        self,
        embeddings: np.ndarray,
        k_clusters: int
    ) -> List[int]:
        """
        Select diverse anchors using max-min distance (farthest-first traversal).

        Args:
            embeddings: (N, D) embeddings
            k_clusters: Number of anchors to select

        Returns:
            List of diverse anchor indices
        """
        diverse_anchors = []
        remaining_indices = set(range(len(embeddings)))

        # Start with random point
        first_anchor = np.random.choice(list(remaining_indices))
        diverse_anchors.append(first_anchor)
        remaining_indices.remove(first_anchor)

        # Iteratively select farthest point from existing anchors
        for _ in range(k_clusters - 1):
            if not remaining_indices:
                break

            # Calculate minimum distance to existing anchors
            anchor_embeddings = embeddings[diverse_anchors]
            remaining_embeddings = embeddings[list(remaining_indices)]

            distances = cdist(remaining_embeddings, anchor_embeddings, metric='cosine')
            min_distances = distances.min(axis=1)

            # Select point with maximum minimum distance
            farthest_idx_in_remaining = np.argmax(min_distances)
            farthest_idx = list(remaining_indices)[farthest_idx_in_remaining]

            diverse_anchors.append(farthest_idx)
            remaining_indices.remove(farthest_idx)

        return diverse_anchors

    def _select_hard_anchors(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        k_clusters: int
    ) -> List[int]:
        """
        Select "hard" anchors: points closest to the decision boundary (nearest enemies).
        These samples are critical for defining the class boundary.

        Args:
            class_embeddings: (N, D) embeddings of target class
            all_embeddings: All embeddings
            all_labels: All labels
            target_class: Target class
            k_clusters: Number of anchors to select

        Returns:
            List of hard anchor indices
        """
        # Identify non-target samples
        non_target_mask = all_labels != target_class
        non_target_embeddings = all_embeddings[non_target_mask]

        if len(non_target_embeddings) == 0:
            return []

        # Calculate distance to nearest enemy for each class sample
        # We use a subset of enemies for speed if dataset is large
        if len(non_target_embeddings) > 5000:
            indices = np.random.choice(len(non_target_embeddings), 5000, replace=False)
            enemy_subset = non_target_embeddings[indices]
        else:
            enemy_subset = non_target_embeddings

        distances = cdist(class_embeddings, enemy_subset, metric='cosine')
        min_dist_to_enemy = distances.min(axis=1)

        # Select samples with SMALLEST distance to enemy (closest to boundary)
        # We want the "hardest" samples
        hardest_indices = np.argsort(min_dist_to_enemy)[:k_clusters]

        return list(hardest_indices)

    def _deduplicate_anchors(
        self,
        embeddings: np.ndarray,
        anchors: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """
        Remove duplicate anchors based on similarity threshold.

        Args:
            embeddings: (N, D) embeddings
            anchors: List of (index, strategy) tuples

        Returns:
            Deduplicated list of anchors
        """
        if len(anchors) == 0:
            return []

        unique_anchors = [anchors[0]]
        unique_embeddings = [embeddings[anchors[0][0]]]

        for anchor_idx, strategy in anchors[1:]:
            anchor_emb = embeddings[anchor_idx]

            # Check similarity to existing unique anchors
            similarities = 1 - cdist([anchor_emb], unique_embeddings, metric='cosine')[0]

            # Keep if not too similar to any existing anchor
            if (similarities < self.dedup_threshold).all():
                unique_anchors.append((anchor_idx, strategy))
                unique_embeddings.append(anchor_emb)

        return unique_anchors

    def _rank_by_ensemble_score(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        anchors: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """
        Rank anchors by composite ensemble score.

        Score = diversity_weight × diversity_score
              + quality_weight × quality_score
              + hardness_weight × hardness_score
              + stability_weight × stability_score

        Args:
            class_embeddings: Embeddings of target class
            all_embeddings: All embeddings
            all_labels: All labels
            target_class: Target class
            anchors: List of (index, strategy) tuples

        Returns:
            Ranked list of anchors
        """
        scored_anchors = []

        # Pre-calculate boundary distances for hardness scoring
        non_target_mask = all_labels != target_class
        non_target_embeddings = all_embeddings[non_target_mask]
        if len(non_target_embeddings) > 2000:
             indices = np.random.choice(len(non_target_embeddings), 2000, replace=False)
             enemy_subset = non_target_embeddings[indices]
        else:
             enemy_subset = non_target_embeddings

        for anchor_idx, strategy in anchors:
            anchor_emb = class_embeddings[anchor_idx: anchor_idx + 1]

            # 1. Diversity score: average distance to other anchors
            other_anchors = [class_embeddings[idx] for idx, _ in anchors if idx != anchor_idx]
            if len(other_anchors) > 0:
                distances = cdist(anchor_emb, other_anchors, metric='cosine')[0]
                diversity_score = distances.mean()
            else:
                diversity_score = 1.0

            # 2. Quality score: purity of neighborhood
            distances_to_all = cdist(anchor_emb, all_embeddings, metric='cosine')[0]
            k_neighbors = min(15, len(all_embeddings))
            nearest_indices = np.argsort(distances_to_all)[:k_neighbors]
            nearest_labels = all_labels[nearest_indices]
            purity = (nearest_labels == target_class).mean()
            quality_score = purity

            # 3. Hardness score: proximity to boundary (1 - min_dist)
            # Closer to enemy = Higher hardness score
            if len(enemy_subset) > 0:
                dists_to_enemy = cdist(anchor_emb, enemy_subset, metric='cosine')[0]
                min_dist = dists_to_enemy.min()
                # Normalize: assume dist usually 0.2-1.0.
                # We want small dist -> high score.
                hardness_score = np.exp(-2.0 * min_dist) # 0.0 -> 1.0, 0.5 -> 0.36
            else:
                hardness_score = 0.0

            # 4. Stability score: inverse of distance variance to cluster members
            distances_to_class = cdist(anchor_emb, class_embeddings, metric='cosine')[0]
            stability_score = 1.0 / (1.0 + distances_to_class.std())

            # Composite score
            ensemble_score = (
                self.diversity_weight * diversity_score +
                self.quality_weight * quality_score +
                self.hardness_weight * hardness_score +
                self.stability_weight * stability_score
            )

            scored_anchors.append((anchor_idx, strategy, ensemble_score))

        # Sort by score descending
        scored_anchors.sort(key=lambda x: x[2], reverse=True)

        return [(idx, strategy) for idx, strategy, _ in scored_anchors]

    def _calculate_ensemble_metrics(
        self,
        class_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        all_labels: np.ndarray,
        target_class: str,
        final_anchors: List[int],
        medoid_anchors: List[int],
        quality_anchors: List[int],
        diverse_anchors: List[int],
        hard_anchors: List[int]
    ) -> Dict[str, any]:
        """
        Calculate metrics for ensemble anchor selection.

        Returns:
            Dictionary of metrics
        """
        # Count strategy contributions
        strategy_counts = {
            "medoid": len(medoid_anchors),
            "quality": len(quality_anchors),
            "diverse": len(diverse_anchors),
            "hard": len(hard_anchors),
            "final": len(final_anchors)
        }

        # Calculate average purity of selected anchors
        purities = []
        for anchor_idx in final_anchors:
            anchor_emb = class_embeddings[anchor_idx: anchor_idx + 1]
            distances = cdist(anchor_emb, all_embeddings, metric='cosine')[0]
            k_neighbors = min(15, len(all_embeddings))
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_labels = all_labels[nearest_indices]
            purity = (nearest_labels == target_class).mean()
            purities.append(purity)

        # Calculate diversity (average pairwise distance)
        if len(final_anchors) > 1:
            anchor_embeddings = class_embeddings[final_anchors]
            pairwise_dists = pairwise_distances(anchor_embeddings, metric='cosine')
            # Upper triangle (exclude diagonal)
            upper_triangle = np.triu(pairwise_dists, k=1)
            diversity = upper_triangle[upper_triangle > 0].mean()
        else:
            diversity = 0.0

        return {
            "strategy_counts": strategy_counts,
            "avg_purity": np.mean(purities),
            "min_purity": np.min(purities),
            "max_purity": np.max(purities),
            "diversity": diversity,
            "n_final_anchors": len(final_anchors)
        }


def test_ensemble_anchor_selector():
    """Test ensemble anchor selector with synthetic data."""
    np.random.seed(42)

    # Create synthetic data
    n_samples = 1000
    n_dims = 384
    n_classes = 4

    # Generate embeddings with some structure
    embeddings = np.random.randn(n_samples, n_dims)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create labels
    labels = np.random.choice([f"class_{i}" for i in range(n_classes)], size=n_samples)

    # Test ensemble selector
    selector = EnsembleAnchorSelector(
        min_purity=0.60,
        diversity_weight=0.3,
        quality_weight=0.3,
        hardness_weight=0.2,
        stability_weight=0.2
    )

    target_class = "class_0"
    k_clusters = 12

    anchor_indices, metrics = selector.select_ensemble_anchors(
        embeddings, labels, target_class, k_clusters
    )

    print("🎯 Ensemble Anchor Selection Test")
    print(f"Target Class: {target_class}")
    print(f"K Clusters: {k_clusters}")
    print(f"\n📊 Results:")
    print(f"Selected Anchors: {len(anchor_indices)}")
    print(f"Strategy Counts: {metrics['strategy_counts']}")
    print(f"Average Purity: {metrics['avg_purity']:.3f}")
    print(f"Min Purity: {metrics['min_purity']:.3f}")
    print(f"Max Purity: {metrics['max_purity']:.3f}")
    print(f"Diversity: {metrics['diversity']:.3f}")


if __name__ == "__main__":
    test_ensemble_anchor_selector()
