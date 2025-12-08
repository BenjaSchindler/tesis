"""
Phase 2: Contamination-Aware Filtering

Applies dynamic filtering thresholds based on cluster contamination risk to improve
synthetic data quality while maintaining acceptance rates.

Author: Benja
Date: 2025-10-30
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist


class ContaminationAwareFilter:
    """
    Dynamic filtering system that adjusts thresholds based on cluster quality.

    High-contamination clusters (low purity) get stricter filters.
    Low-contamination clusters (high purity) get more lenient filters.
    """

    def __init__(
        self,
        # Base thresholds (Phase 1 defaults)
        base_min_similarity: float = 0.35,
        base_min_confidence: float = 0.60,
        base_similarity_to_anchor: float = 0.50,
        # Risk thresholds
        high_risk_purity_threshold: float = 0.30,
        medium_risk_purity_threshold: float = 0.50,
        low_risk_purity_threshold: float = 0.70,
        # Risk multipliers
        high_risk_similarity_mult: float = 1.43,  # 0.35 → 0.50
        high_risk_confidence_mult: float = 1.33,  # 0.60 → 0.80
        medium_risk_similarity_mult: float = 1.14,  # 0.35 → 0.40
        medium_risk_confidence_mult: float = 1.17,  # 0.60 → 0.70
        # Additional features
        enable_contamination_penalty: bool = True,
        contamination_penalty_weight: float = 0.2
    ):
        """
        Initialize contamination-aware filter.

        Args:
            base_min_similarity: Base KNN similarity threshold
            base_min_confidence: Base classifier confidence threshold
            base_similarity_to_anchor: Base anchor similarity threshold
            high_risk_purity_threshold: Purity below which cluster is high risk
            medium_risk_purity_threshold: Purity below which cluster is medium risk
            low_risk_purity_threshold: Purity above which cluster is low risk
            high_risk_similarity_mult: Multiplier for similarity threshold in high risk
            high_risk_confidence_mult: Multiplier for confidence threshold in high risk
            medium_risk_similarity_mult: Multiplier for similarity threshold in medium risk
            medium_risk_confidence_mult: Multiplier for confidence threshold in medium risk
            enable_contamination_penalty: Whether to apply contamination penalty to scores
            contamination_penalty_weight: Weight of contamination penalty
        """
        self.base_min_similarity = base_min_similarity
        self.base_min_confidence = base_min_confidence
        self.base_similarity_to_anchor = base_similarity_to_anchor

        self.high_risk_purity_threshold = high_risk_purity_threshold
        self.medium_risk_purity_threshold = medium_risk_purity_threshold
        self.low_risk_purity_threshold = low_risk_purity_threshold

        self.high_risk_similarity_mult = high_risk_similarity_mult
        self.high_risk_confidence_mult = high_risk_confidence_mult
        self.medium_risk_similarity_mult = medium_risk_similarity_mult
        self.medium_risk_confidence_mult = medium_risk_confidence_mult

        self.enable_contamination_penalty = enable_contamination_penalty
        self.contamination_penalty_weight = contamination_penalty_weight

    def get_cluster_thresholds(
        self,
        cluster_purity: float,
        cluster_size: int,
        cluster_cohesion: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate dynamic thresholds based on cluster quality.

        Args:
            cluster_purity: Purity of the cluster (0-1)
            cluster_size: Number of samples in cluster
            cluster_cohesion: Optional cohesion score (0-1)

        Returns:
            Dictionary of threshold values
        """
        # Determine risk level
        if cluster_purity < self.high_risk_purity_threshold:
            risk_level = "high"
            similarity_mult = self.high_risk_similarity_mult
            confidence_mult = self.high_risk_confidence_mult
        elif cluster_purity < self.medium_risk_purity_threshold:
            risk_level = "medium"
            similarity_mult = self.medium_risk_similarity_mult
            confidence_mult = self.medium_risk_confidence_mult
        elif cluster_purity >= self.low_risk_purity_threshold:
            risk_level = "low"
            similarity_mult = 0.85  # More lenient (0.35 → 0.30)
            confidence_mult = 0.92  # More lenient (0.60 → 0.55)
        else:
            risk_level = "normal"
            similarity_mult = 1.0
            confidence_mult = 1.0

        # Apply multipliers
        min_similarity = min(0.95, self.base_min_similarity * similarity_mult)
        min_confidence = min(0.95, self.base_min_confidence * confidence_mult)
        similarity_to_anchor = min(0.95, self.base_similarity_to_anchor * similarity_mult)

        # Adjust for very small clusters (less reliable)
        if cluster_size < 30:
            min_similarity = min(0.95, min_similarity * 1.1)
            min_confidence = min(0.95, min_confidence * 1.05)

        return {
            "risk_level": risk_level,
            "min_similarity": min_similarity,
            "min_confidence": min_confidence,
            "similarity_to_anchor": similarity_to_anchor,
            "cluster_purity": cluster_purity,
            "cluster_size": cluster_size
        }

    def calculate_contamination_score(
        self,
        cluster_purity: float,
        synthetic_ratio: float
    ) -> float:
        """
        Calculate contamination score using Proportional Contamination Theory.

        Contamination = Ratio × (1 - Purity)

        Args:
            cluster_purity: Purity of the cluster (0-1)
            synthetic_ratio: synthetics / reals ratio (0-1)

        Returns:
            Contamination score (0-1)
        """
        contamination = synthetic_ratio * (1.0 - cluster_purity)
        return contamination

    def filter_synthetic(
        self,
        synthetic_embedding: np.ndarray,
        anchor_embedding: np.ndarray,
        real_embeddings: np.ndarray,
        cluster_metrics: Dict[str, float],
        classifier_confidence: float,
        k_neighbors: int = 15
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Filter a synthetic example using contamination-aware thresholds.

        Args:
            synthetic_embedding: (D,) embedding of synthetic sample
            anchor_embedding: (D,) embedding of cluster anchor
            real_embeddings: (N, D) embeddings of real samples in cluster
            cluster_metrics: Dictionary with purity, size, cohesion
            classifier_confidence: Confidence from classifier (0-1)
            k_neighbors: Number of neighbors for KNN filter

        Returns:
            (accept, details) tuple
        """
        # Get dynamic thresholds
        thresholds = self.get_cluster_thresholds(
            cluster_purity=cluster_metrics['purity'],
            cluster_size=cluster_metrics['size'],
            cluster_cohesion=cluster_metrics.get('cohesion')
        )

        # Calculate similarities
        sim_to_anchor = 1.0 - cdist(
            [synthetic_embedding], [anchor_embedding], metric='cosine'
        )[0, 0]

        # KNN similarity
        distances_to_reals = cdist(
            [synthetic_embedding], real_embeddings, metric='cosine'
        )[0]
        k = min(k_neighbors, len(real_embeddings))
        nearest_distances = np.sort(distances_to_reals)[:k]
        avg_knn_similarity = 1.0 - nearest_distances.mean()

        # Apply filters
        filters_passed = {
            "anchor_similarity": sim_to_anchor >= thresholds['similarity_to_anchor'],
            "knn_similarity": avg_knn_similarity >= thresholds['min_similarity'],
            "classifier_confidence": classifier_confidence >= thresholds['min_confidence']
        }

        # Overall decision
        accept = all(filters_passed.values())

        # Apply contamination penalty if enabled
        final_score = classifier_confidence
        if self.enable_contamination_penalty and not accept:
            contamination = self.calculate_contamination_score(
                cluster_purity=cluster_metrics['purity'],
                synthetic_ratio=cluster_metrics.get('synthetic_ratio', 0.05)
            )
            penalty = self.contamination_penalty_weight * contamination
            final_score = classifier_confidence * (1.0 - penalty)

        # Detailed results
        details = {
            "accept": accept,
            "risk_level": thresholds['risk_level'],
            "thresholds": thresholds,
            "scores": {
                "sim_to_anchor": sim_to_anchor,
                "knn_similarity": avg_knn_similarity,
                "classifier_confidence": classifier_confidence,
                "final_score": final_score
            },
            "filters_passed": filters_passed,
            "reason": self._get_rejection_reason(filters_passed) if not accept else "accepted"
        }

        return accept, details

    def _get_rejection_reason(self, filters_passed: Dict[str, bool]) -> str:
        """Get human-readable rejection reason."""
        failed_filters = [name for name, passed in filters_passed.items() if not passed]
        if len(failed_filters) == 0:
            return "none"
        elif len(failed_filters) == 1:
            return f"failed_{failed_filters[0]}"
        else:
            return f"failed_multiple ({', '.join(failed_filters)})"

    def get_cluster_budget_adjustment(
        self,
        cluster_purity: float,
        cluster_size: int,
        base_budget: int
    ) -> Tuple[int, str]:
        """
        Adjust cluster budget based on contamination risk.

        Args:
            cluster_purity: Purity of cluster (0-1)
            cluster_size: Number of samples in cluster
            base_budget: Base budget from dynamic calculator

        Returns:
            (adjusted_budget, reason) tuple
        """
        # High risk: reduce budget further
        if cluster_purity < self.high_risk_purity_threshold:
            adjusted = max(5, int(base_budget * 0.5))
            reason = f"high_contamination_risk (purity={cluster_purity:.3f})"

        # Medium risk: slight reduction
        elif cluster_purity < self.medium_risk_purity_threshold:
            adjusted = max(8, int(base_budget * 0.7))
            reason = f"medium_contamination_risk (purity={cluster_purity:.3f})"

        # Low risk: can increase budget
        elif cluster_purity >= self.low_risk_purity_threshold:
            adjusted = int(base_budget * 1.2)
            reason = f"low_contamination_risk (purity={cluster_purity:.3f})"

        # Normal risk: no adjustment
        else:
            adjusted = base_budget
            reason = "normal_risk"

        # Ensure minimum budget
        adjusted = max(5, adjusted)

        return adjusted, reason


def test_contamination_aware_filter():
    """Test contamination-aware filter with synthetic data."""
    np.random.seed(42)

    # Create synthetic data
    n_reals = 100
    n_dims = 384

    # Real embeddings
    real_embeddings = np.random.randn(n_reals, n_dims)
    real_embeddings = real_embeddings / np.linalg.norm(real_embeddings, axis=1, keepdims=True)

    # Anchor
    anchor_embedding = real_embeddings[0]

    # Synthetic (similar to anchor)
    synthetic_embedding = anchor_embedding + 0.1 * np.random.randn(n_dims)
    synthetic_embedding = synthetic_embedding / np.linalg.norm(synthetic_embedding)

    # Test filter
    filter_system = ContaminationAwareFilter(
        base_min_similarity=0.35,
        base_min_confidence=0.60,
        high_risk_purity_threshold=0.30,
        medium_risk_purity_threshold=0.50
    )

    # Test with different purity levels
    test_cases = [
        {"purity": 0.10, "size": 50, "label": "High Risk (purity=0.10)"},
        {"purity": 0.40, "size": 75, "label": "Medium Risk (purity=0.40)"},
        {"purity": 0.70, "size": 100, "label": "Low Risk (purity=0.70)"}
    ]

    print("🔬 Contamination-Aware Filter Test\n")

    for test_case in test_cases:
        cluster_metrics = {
            "purity": test_case["purity"],
            "size": test_case["size"],
            "cohesion": 0.5,
            "synthetic_ratio": 0.05
        }

        accept, details = filter_system.filter_synthetic(
            synthetic_embedding=synthetic_embedding,
            anchor_embedding=anchor_embedding,
            real_embeddings=real_embeddings,
            cluster_metrics=cluster_metrics,
            classifier_confidence=0.65,
            k_neighbors=15
        )

        print(f"📊 {test_case['label']}")
        print(f"   Risk Level: {details['risk_level']}")
        print(f"   Decision: {'✅ ACCEPT' if accept else '❌ REJECT'}")
        print(f"   Thresholds:")
        print(f"      - Min Similarity: {details['thresholds']['min_similarity']:.3f}")
        print(f"      - Min Confidence: {details['thresholds']['min_confidence']:.3f}")
        print(f"      - Anchor Similarity: {details['thresholds']['similarity_to_anchor']:.3f}")
        print(f"   Scores:")
        print(f"      - Similarity to Anchor: {details['scores']['sim_to_anchor']:.3f}")
        print(f"      - KNN Similarity: {details['scores']['knn_similarity']:.3f}")
        print(f"      - Classifier Confidence: {details['scores']['classifier_confidence']:.3f}")
        print(f"   Filters Passed: {details['filters_passed']}")
        if not accept:
            print(f"   Rejection Reason: {details['reason']}")
        print()

    # Test budget adjustments
    print("💰 Budget Adjustment Tests\n")
    base_budget = 100
    for test_case in test_cases:
        adjusted, reason = filter_system.get_cluster_budget_adjustment(
            cluster_purity=test_case["purity"],
            cluster_size=test_case["size"],
            base_budget=base_budget
        )
        print(f"📊 {test_case['label']}")
        print(f"   Base Budget: {base_budget}")
        print(f"   Adjusted Budget: {adjusted}")
        print(f"   Adjustment Factor: {adjusted/base_budget:.2f}×")
        print(f"   Reason: {reason}")
        print()


if __name__ == "__main__":
    test_contamination_aware_filter()
