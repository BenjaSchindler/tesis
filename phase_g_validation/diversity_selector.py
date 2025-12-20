#!/usr/bin/env python3
"""
Diversity-Based Config Selection for Ensemble Building

Selects configs that maximize diversity in per-class performance,
ensuring complementary strengths across different MBTI classes.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.spatial.distance import euclidean, cosine


# MBTI classes in canonical order
MBTI_CLASSES = [
    "ENFJ", "ENFP", "ENTJ", "ENTP",
    "ESFJ", "ESFP", "ESTJ", "ESTP",
    "INFJ", "INFP", "INTJ", "INTP",
    "ISFJ", "ISFP", "ISTJ", "ISTP"
]

RESULTS_FILE = Path("/home/benja/Desktop/Tesis/SMOTE-LLM/phase_g_validation/results/FULL_SUMMARY.json")


def load_all_configs() -> List[Dict]:
    """Load all configurations from FULL_SUMMARY.json."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    # Flatten all categories into single list
    all_configs = []
    for category, configs in data.items():
        for cfg in configs:
            cfg["category"] = category
            all_configs.append(cfg)

    return all_configs


def get_per_class_vector(config: Dict) -> np.ndarray:
    """Extract per-class delta vector (16D) from config."""
    per_class = config.get("per_class_delta", {})
    vector = np.array([per_class.get(cls, 0.0) for cls in MBTI_CLASSES])
    return vector


def compute_pairwise_distances(configs: List[Dict], metric='euclidean') -> np.ndarray:
    """
    Compute pairwise distances between configs in per-class space.

    Args:
        configs: List of config dictionaries
        metric: 'euclidean' or 'cosine'

    Returns:
        Distance matrix (N x N)
    """
    n = len(configs)
    distances = np.zeros((n, n))

    vectors = [get_per_class_vector(cfg) for cfg in configs]

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                dist = euclidean(vectors[i], vectors[j])
            elif metric == 'cosine':
                dist = cosine(vectors[i], vectors[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def select_diverse_configs(
    configs: List[Dict],
    k: int = 5,
    metric: str = 'euclidean',
    seed_config: str = None,
    exclude_configs: List[str] = None
) -> List[Dict]:
    """
    Greedy diversity maximization: select k configs with maximum pairwise distance.

    Strategy:
    1. Start with best overall config (or specified seed)
    2. Iteratively add config that maximizes minimum distance to selected set

    Args:
        configs: All available configs
        k: Number of configs to select
        metric: Distance metric ('euclidean' or 'cosine')
        seed_config: Optional initial config name (default: best overall)
        exclude_configs: Optional list of config names to exclude

    Returns:
        List of k selected configs
    """
    if exclude_configs is None:
        exclude_configs = []

    # Filter out excluded configs
    available = [cfg for cfg in configs if cfg["config"] not in exclude_configs]

    if len(available) < k:
        raise ValueError(f"Not enough configs: {len(available)} available, {k} requested")

    # Compute distance matrix
    distances = compute_pairwise_distances(available, metric=metric)

    # Initialize selected set
    if seed_config:
        # Find seed config
        seed_idx = next((i for i, cfg in enumerate(available) if cfg["config"] == seed_config), None)
        if seed_idx is None:
            raise ValueError(f"Seed config not found: {seed_config}")
    else:
        # Use best overall config (highest delta_pct)
        seed_idx = max(range(len(available)), key=lambda i: available[i]["delta_pct"])

    selected_indices = [seed_idx]
    remaining_indices = list(set(range(len(available))) - {seed_idx})

    # Greedy selection
    while len(selected_indices) < k:
        best_idx = None
        best_min_dist = -1

        for candidate_idx in remaining_indices:
            # Compute minimum distance to already selected configs
            min_dist = min(distances[candidate_idx, sel_idx] for sel_idx in selected_indices)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate_idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    selected_configs = [available[i] for i in selected_indices]

    return selected_configs


def select_class_balanced_configs(
    configs: List[Dict],
    k: int = 5,
    target_classes: List[str] = None
) -> List[Dict]:
    """
    Select configs ensuring each target class has at least one config improving it.

    Args:
        configs: All available configs
        k: Number of configs to select
        target_classes: Classes to ensure coverage (default: all 16)

    Returns:
        List of k configs with balanced class coverage
    """
    if target_classes is None:
        target_classes = MBTI_CLASSES

    selected = []
    covered_classes = set()

    # Sort configs by overall performance
    sorted_configs = sorted(configs, key=lambda c: c["delta_pct"], reverse=True)

    # Phase 1: Ensure at least one config per target class
    for cls in target_classes:
        if cls in covered_classes:
            continue

        # Find best config for this class
        best_for_class = max(
            (cfg for cfg in sorted_configs if cfg not in selected),
            key=lambda cfg: cfg.get("per_class_delta", {}).get(cls, -999),
            default=None
        )

        if best_for_class and best_for_class.get("per_class_delta", {}).get(cls, 0) > 0:
            selected.append(best_for_class)
            # Mark which classes this config improves
            for c in MBTI_CLASSES:
                if best_for_class.get("per_class_delta", {}).get(c, 0) > 0:
                    covered_classes.add(c)

        if len(selected) >= k:
            break

    # Phase 2: Fill remaining slots with best overall configs
    while len(selected) < k:
        best_remaining = next(
            (cfg for cfg in sorted_configs if cfg not in selected),
            None
        )
        if best_remaining is None:
            break
        selected.append(best_remaining)

    return selected[:k]


def select_strategy_diverse_configs(
    configs: List[Dict],
    strategy_types: Dict[str, List[str]]
) -> List[Dict]:
    """
    Select best config from each strategy type.

    Args:
        configs: All available configs
        strategy_types: Dict mapping strategy name to list of config names
            e.g., {"prompting": ["W5_many_shot_10", "W5_few_shot_3"], ...}

    Returns:
        List of selected configs (one per strategy)
    """
    selected = []

    for strategy_name, config_names in strategy_types.items():
        # Find configs matching this strategy
        strategy_configs = [cfg for cfg in configs if cfg["config"] in config_names]

        if not strategy_configs:
            continue

        # Select best from this strategy
        best = max(strategy_configs, key=lambda c: c["delta_pct"])
        selected.append(best)

    return selected


def analyze_diversity(configs: List[Dict], metric: str = 'euclidean') -> Dict:
    """
    Analyze diversity metrics for a set of configs.

    Returns:
        Dict with diversity statistics
    """
    if len(configs) < 2:
        return {"error": "Need at least 2 configs"}

    distances = compute_pairwise_distances(configs, metric=metric)

    # Get upper triangle (excluding diagonal)
    n = len(configs)
    pairwise_dists = [distances[i, j] for i in range(n) for j in range(i + 1, n)]

    return {
        "num_configs": n,
        "min_pairwise_distance": float(np.min(pairwise_dists)),
        "max_pairwise_distance": float(np.max(pairwise_dists)),
        "mean_pairwise_distance": float(np.mean(pairwise_dists)),
        "median_pairwise_distance": float(np.median(pairwise_dists)),
        "std_pairwise_distance": float(np.std(pairwise_dists)),
    }


def print_diversity_report(configs: List[Dict], title: str = "Diversity Report"):
    """Print diversity analysis report."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}\n")

    print(f"Configs: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg['config']:25} delta={cfg['delta_pct']:+.2f}%  p={cfg['p_value']:.6f}")

    print("\nPer-Class Performance Matrix:")
    print(f"{'Config':25}", end="")
    for cls in MBTI_CLASSES:
        print(f"{cls:>6}", end="")
    print()

    for cfg in configs:
        print(f"{cfg['config']:25}", end="")
        for cls in MBTI_CLASSES:
            delta = cfg.get("per_class_delta", {}).get(cls, 0.0)
            print(f"{delta:+6.3f}", end="")
        print()

    diversity_stats = analyze_diversity(configs)
    print(f"\nDiversity Metrics:")
    for key, value in diversity_stats.items():
        if key == "num_configs":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")


# ============================================================================
# Main: Test/Demo Functions
# ============================================================================

def demo_diversity_selection():
    """Demo: Select top 5 diverse configs."""
    print("Loading configs...")
    all_configs = load_all_configs()
    print(f"Loaded {len(all_configs)} configs\n")

    # Test 1: Top 5 diverse
    print("Test 1: Selecting 5 most diverse configs...")
    diverse_5 = select_diverse_configs(all_configs, k=5, metric='euclidean')
    print_diversity_report(diverse_5, "Top 5 Diverse Configs (Euclidean)")

    # Test 2: Top 7 diverse
    print("\n\nTest 2: Selecting 7 most diverse configs...")
    diverse_7 = select_diverse_configs(all_configs, k=7, metric='euclidean')
    print_diversity_report(diverse_7, "Top 7 Diverse Configs (Euclidean)")

    # Test 3: Class-balanced selection
    print("\n\nTest 3: Class-balanced selection (5 configs)...")
    balanced_5 = select_class_balanced_configs(all_configs, k=5)
    print_diversity_report(balanced_5, "Class-Balanced 5 Configs")

    # Test 4: Strategy-diverse selection
    print("\n\nTest 4: Strategy-diverse selection...")
    strategy_types = {
        "prompting": ["W5_many_shot_10", "W5_few_shot_3", "W5_zero_shot"],
        "temperature": ["W6_temp_high", "W6_temp_low", "W6_temp_extreme"],
        "volume": ["V4_ultra", "W2_ultra_vol", "W2_mega_vol"],
        "filtering": ["W3_permissive_filter", "W7_yolo"],
        "budget": ["CMB3_skip"],
    }
    strategy_diverse = select_strategy_diverse_configs(all_configs, strategy_types)
    print_diversity_report(strategy_diverse, "Strategy-Diverse Configs")


if __name__ == "__main__":
    demo_diversity_selection()
