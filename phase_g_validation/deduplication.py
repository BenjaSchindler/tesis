#!/usr/bin/env python3
"""
Deduplication Utilities for Ensemble Synthetic Data

Removes redundant synthetic samples across ensemble components using:
- Cosine similarity thresholding
- Class-wise deduplication
- Clustering-based deduplication
"""
import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict


def deduplicate_by_similarity(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    threshold: float = 0.95,
    method: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Remove duplicate synthetic samples based on embedding similarity.

    Args:
        X_synth: Synthetic embeddings (N x 768)
        y_synth: Synthetic labels (N,)
        threshold: Similarity threshold (default 0.95). Samples above this are considered duplicates.
        method: 'cosine' (cosine similarity) or 'euclidean' (Euclidean distance)

    Returns:
        X_dedup: Deduplicated embeddings
        y_dedup: Deduplicated labels
        stats: Deduplication statistics
    """
    if len(X_synth) == 0:
        return X_synth, y_synth, {"removed": 0, "kept": 0}

    n = len(X_synth)

    if method == 'cosine':
        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(X_synth)
    elif method == 'euclidean':
        # Compute pairwise Euclidean distance, normalize to [0,1]
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(X_synth)
        max_dist = np.max(dist_matrix)
        sim_matrix = 1 - (dist_matrix / max_dist) if max_dist > 0 else np.ones_like(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Mark samples to keep (greedy: keep first occurrence)
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue

        # Find duplicates of sample i (samples j > i with sim > threshold)
        duplicates = np.where((sim_matrix[i, i+1:] > threshold) & keep_mask[i+1:])[0] + (i + 1)

        # Remove duplicates
        keep_mask[duplicates] = False

    X_dedup = X_synth[keep_mask]
    y_dedup = y_synth[keep_mask]

    stats = {
        "original": n,
        "kept": int(np.sum(keep_mask)),
        "removed": int(n - np.sum(keep_mask)),
        "removal_rate": float((n - np.sum(keep_mask)) / n) if n > 0 else 0.0
    }

    return X_dedup, y_dedup, stats


def deduplicate_classwise(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    threshold: float = 0.95,
    method: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Deduplicate within each class separately.

    This ensures we don't remove samples from different classes
    even if they have similar embeddings.

    Args:
        X_synth: Synthetic embeddings (N x 768)
        y_synth: Synthetic labels (N,)
        threshold: Similarity threshold
        method: 'cosine' or 'euclidean'

    Returns:
        X_dedup: Deduplicated embeddings
        y_dedup: Deduplicated labels
        stats: Per-class deduplication statistics
    """
    if len(X_synth) == 0:
        return X_synth, y_synth, {}

    unique_classes = np.unique(y_synth)
    X_dedup_list = []
    y_dedup_list = []
    stats = {}

    for cls in unique_classes:
        # Get samples for this class
        mask = (y_synth == cls)
        X_cls = X_synth[mask]
        y_cls = y_synth[mask]

        # Deduplicate within class
        X_cls_dedup, y_cls_dedup, cls_stats = deduplicate_by_similarity(
            X_cls, y_cls, threshold=threshold, method=method
        )

        X_dedup_list.append(X_cls_dedup)
        y_dedup_list.append(y_cls_dedup)

        stats[cls] = cls_stats

    # Combine all classes
    X_dedup = np.vstack(X_dedup_list) if X_dedup_list else np.array([]).reshape(0, X_synth.shape[1])
    y_dedup = np.concatenate(y_dedup_list) if y_dedup_list else np.array([])

    # Add overall stats
    stats["overall"] = {
        "original": len(X_synth),
        "kept": len(X_dedup),
        "removed": len(X_synth) - len(X_dedup),
        "removal_rate": (len(X_synth) - len(X_dedup)) / len(X_synth) if len(X_synth) > 0 else 0.0
    }

    return X_dedup, y_dedup, stats


def deduplicate_by_clustering(
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    n_clusters: int = None,
    cluster_method: str = 'kmeans',
    keep_strategy: str = 'centroid'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Deduplicate by clustering and keeping representative samples.

    Args:
        X_synth: Synthetic embeddings (N x 768)
        y_synth: Synthetic labels (N,)
        n_clusters: Number of clusters (default: N/10). If None, auto-determined.
        cluster_method: 'kmeans' or 'agglomerative'
        keep_strategy: 'centroid' (keep closest to cluster center) or
                       'first' (keep first sample in cluster)

    Returns:
        X_dedup: Deduplicated embeddings (cluster representatives)
        y_dedup: Deduplicated labels
        stats: Clustering statistics
    """
    if len(X_synth) == 0:
        return X_synth, y_synth, {"removed": 0, "kept": 0}

    n = len(X_synth)

    # Auto-determine n_clusters if not specified
    if n_clusters is None:
        n_clusters = max(1, n // 10)  # 10% of samples

    n_clusters = min(n_clusters, n)  # Can't have more clusters than samples

    # Perform clustering
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif cluster_method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}")

    cluster_labels = clusterer.fit_predict(X_synth)

    # Select representatives from each cluster
    keep_indices = []

    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        if keep_strategy == 'centroid':
            # Keep sample closest to cluster centroid
            cluster_samples = X_synth[cluster_mask]
            centroid = np.mean(cluster_samples, axis=0)

            # Find closest sample to centroid
            distances = np.linalg.norm(cluster_samples - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            keep_indices.append(closest_idx)

        elif keep_strategy == 'first':
            # Keep first sample in cluster
            keep_indices.append(cluster_indices[0])

        else:
            raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

    keep_indices = sorted(keep_indices)

    X_dedup = X_synth[keep_indices]
    y_dedup = y_synth[keep_indices]

    stats = {
        "original": n,
        "n_clusters": n_clusters,
        "kept": len(keep_indices),
        "removed": n - len(keep_indices),
        "removal_rate": (n - len(keep_indices)) / n if n > 0 else 0.0,
        "avg_cluster_size": n / n_clusters if n_clusters > 0 else 0
    }

    return X_dedup, y_dedup, stats


def deduplicate_cross_component(
    component_data: List[Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.95,
    method: str = 'cosine'
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], dict]:
    """
    Deduplicate across multiple component datasets.

    Removes samples from later components that are duplicates of earlier components.
    This preserves component ordering priority (earlier components have priority).

    Args:
        component_data: List of (X, y) tuples for each component
        threshold: Similarity threshold
        method: 'cosine' or 'euclidean'

    Returns:
        deduped_components: List of deduplicated (X, y) tuples
        stats: Deduplication statistics
    """
    if not component_data:
        return [], {}

    # Stack all data with component IDs
    all_X = []
    all_y = []
    component_ids = []

    for comp_id, (X, y) in enumerate(component_data):
        all_X.append(X)
        all_y.append(y)
        component_ids.extend([comp_id] * len(X))

    if not all_X:
        return component_data, {}

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    component_ids = np.array(component_ids)

    # Compute similarity matrix
    if method == 'cosine':
        sim_matrix = cosine_similarity(all_X)
    elif method == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(all_X)
        max_dist = np.max(dist_matrix)
        sim_matrix = 1 - (dist_matrix / max_dist) if max_dist > 0 else np.ones_like(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Mark samples to keep (priority to earlier components)
    keep_mask = np.ones(len(all_X), dtype=bool)

    for i in range(len(all_X)):
        if not keep_mask[i]:
            continue

        # Find duplicates in LATER samples (j > i)
        for j in range(i + 1, len(all_X)):
            if not keep_mask[j]:
                continue

            # Only remove if:
            # 1. Similarity > threshold
            # 2. Different components (component_ids[i] != component_ids[j])
            if sim_matrix[i, j] > threshold and component_ids[i] != component_ids[j]:
                keep_mask[j] = False

    # Split back into components
    deduped_components = []
    stats = {"components": []}

    for comp_id in range(len(component_data)):
        comp_mask = (component_ids == comp_id) & keep_mask
        X_comp = all_X[comp_mask]
        y_comp = all_y[comp_mask]

        deduped_components.append((X_comp, y_comp))

        original_size = len(component_data[comp_id][0])
        kept_size = len(X_comp)

        stats["components"].append({
            "component_id": comp_id,
            "original": original_size,
            "kept": kept_size,
            "removed": original_size - kept_size,
            "removal_rate": (original_size - kept_size) / original_size if original_size > 0 else 0.0
        })

    stats["overall"] = {
        "original": len(all_X),
        "kept": int(np.sum(keep_mask)),
        "removed": int(len(all_X) - np.sum(keep_mask)),
        "removal_rate": (len(all_X) - np.sum(keep_mask)) / len(all_X) if len(all_X) > 0 else 0.0
    }

    return deduped_components, stats


def print_dedup_stats(stats: dict, title: str = "Deduplication Statistics"):
    """Print deduplication statistics."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}\n")

    if "overall" in stats:
        print("Overall:")
        for key, value in stats["overall"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if "rate" in key else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if "components" in stats:
        print("\nPer-Component:")
        for comp_stat in stats["components"]:
            print(f"  Component {comp_stat['component_id']}:")
            print(f"    Original: {comp_stat['original']}")
            print(f"    Kept: {comp_stat['kept']}")
            print(f"    Removed: {comp_stat['removed']} ({comp_stat['removal_rate']:.2%})")

    # Per-class stats (if available)
    if any(isinstance(k, str) and k not in ["overall", "components"] for k in stats.keys()):
        print("\nPer-Class:")
        for cls, cls_stats in stats.items():
            if cls in ["overall", "components"]:
                continue
            print(f"  {cls}:")
            print(f"    Original: {cls_stats['original']}")
            print(f"    Kept: {cls_stats['kept']}")
            print(f"    Removed: {cls_stats['removed']} ({cls_stats['removal_rate']:.2%})")


# ============================================================================
# Demo / Test Functions
# ============================================================================

def demo_deduplication():
    """Demo deduplication methods on synthetic data."""
    print("Generating synthetic test data...")

    # Create synthetic data with known duplicates
    np.random.seed(42)

    # 100 unique samples
    X_unique = np.random.randn(100, 768)
    y_unique = np.random.randint(0, 16, 100)

    # Add 50 near-duplicates (noise added to existing samples)
    duplicate_indices = np.random.choice(100, 50, replace=True)
    X_duplicates = X_unique[duplicate_indices] + np.random.randn(50, 768) * 0.01
    y_duplicates = y_unique[duplicate_indices]

    # Combine
    X_synth = np.vstack([X_unique, X_duplicates])
    y_synth = np.concatenate([y_unique, y_duplicates])

    print(f"Created {len(X_synth)} samples ({len(X_unique)} unique + {len(X_duplicates)} near-duplicates)")

    # Test 1: Similarity-based deduplication
    print("\n\nTest 1: Similarity-based deduplication (threshold=0.95)")
    X_dedup1, y_dedup1, stats1 = deduplicate_by_similarity(X_synth, y_synth, threshold=0.95)
    print_dedup_stats(stats1, "Similarity-Based (0.95)")

    # Test 2: Similarity-based deduplication (more aggressive)
    print("\n\nTest 2: Similarity-based deduplication (threshold=0.90)")
    X_dedup2, y_dedup2, stats2 = deduplicate_by_similarity(X_synth, y_synth, threshold=0.90)
    print_dedup_stats(stats2, "Similarity-Based (0.90)")

    # Test 3: Class-wise deduplication
    print("\n\nTest 3: Class-wise deduplication (threshold=0.95)")
    X_dedup3, y_dedup3, stats3 = deduplicate_classwise(X_synth, y_synth, threshold=0.95)
    print_dedup_stats(stats3, "Class-Wise Deduplication")

    # Test 4: Clustering-based deduplication
    print("\n\nTest 4: Clustering-based deduplication (n_clusters=20)")
    X_dedup4, y_dedup4, stats4 = deduplicate_by_clustering(
        X_synth, y_synth, n_clusters=20, keep_strategy='centroid'
    )
    print_dedup_stats(stats4, "Clustering-Based (20 clusters)")

    # Test 5: Cross-component deduplication
    print("\n\nTest 5: Cross-component deduplication")
    # Simulate 3 components with overlapping data
    comp1_X = X_synth[:50]
    comp1_y = y_synth[:50]
    comp2_X = X_synth[25:75]  # 25 overlapping with comp1
    comp2_y = y_synth[25:75]
    comp3_X = X_synth[60:110]  # 15 overlapping with comp2
    comp3_y = y_synth[60:110]

    component_data = [(comp1_X, comp1_y), (comp2_X, comp2_y), (comp3_X, comp3_y)]
    deduped_components, stats5 = deduplicate_cross_component(
        component_data, threshold=0.95
    )
    print_dedup_stats(stats5, "Cross-Component Deduplication")


if __name__ == "__main__":
    demo_deduplication()
