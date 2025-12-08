#!/usr/bin/env python3
"""
Quality Gate Predictor System

Predicts whether synthetic generation will help or hurt,
based on class characteristics and anchor quality metrics.

The system always generates synthetics anyway to validate predictions,
creating a confusion matrix to evaluate the predictor.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ClassMetrics:
    """Metrics used for prediction."""
    n_samples: int
    n_clusters: int
    baseline_f1: float
    anchor_cohesion: float
    anchor_purity: float
    anchor_separation: float
    anchor_quality_score: float


class QualityGatePredictor:
    """
    Predicts whether to generate synthetics based on class characteristics.

    Uses heuristic rules derived from empirical observations:
    - Very small classes (<30 samples): High risk
    - Low baseline F1 (<0.1): May need more data
    - Poor anchor quality (<0.5): Risky generation
    - Too many clusters: Fragmented class
    """

    def __init__(self):
        # Thresholds derived from empirical data
        self.min_samples_safe = 30
        self.min_baseline_f1 = 0.05
        self.min_anchor_quality = 0.40  # Phase 1: P25 for MBTI (was 0.5)
        self.max_clusters_ratio = 0.3  # clusters / samples

    def predict(self, metrics: ClassMetrics) -> Tuple[str, float, Dict[str, str]]:
        """
        Predict whether to generate synthetics.

        Returns:
            decision: "generate" or "skip"
            confidence: 0.0 to 1.0
            reasoning: Dict of reasons for the decision
        """
        reasons = {}
        score = 0.5  # Start neutral

        # Rule 1: Sample size
        if metrics.n_samples < self.min_samples_safe:
            score -= 0.2
            reasons["samples"] = f"⚠️  Only {metrics.n_samples} samples (risky)"
        else:
            score += 0.1
            reasons["samples"] = f"✅ {metrics.n_samples} samples (safe)"

        # Rule 2: Baseline performance
        if metrics.baseline_f1 < self.min_baseline_f1:
            score += 0.15  # Very weak baseline might benefit
            reasons["baseline"] = f"💡 Very low F1 ({metrics.baseline_f1:.3f}), may benefit from data"
        elif metrics.baseline_f1 < 0.2:
            score += 0.1
            reasons["baseline"] = f"⚠️  Low F1 ({metrics.baseline_f1:.3f}), some risk"
        else:
            score += 0.05
            reasons["baseline"] = f"✅ Decent F1 ({metrics.baseline_f1:.3f})"

        # Rule 3: Anchor quality
        if metrics.anchor_quality_score < self.min_anchor_quality:
            score -= 0.3
            reasons["anchor"] = f"🔴 Poor anchor quality ({metrics.anchor_quality_score:.3f})"
        elif metrics.anchor_quality_score < 0.7:
            score -= 0.1
            reasons["anchor"] = f"⚠️  Mediocre anchor quality ({metrics.anchor_quality_score:.3f})"
        else:
            score += 0.2
            reasons["anchor"] = f"✅ Good anchor quality ({metrics.anchor_quality_score:.3f})"

        # Rule 4: Cluster fragmentation
        cluster_ratio = metrics.n_clusters / metrics.n_samples
        if cluster_ratio > self.max_clusters_ratio:
            score -= 0.15
            reasons["clusters"] = f"⚠️  High fragmentation ({metrics.n_clusters} clusters / {metrics.n_samples} samples)"
        else:
            score += 0.1
            reasons["clusters"] = f"✅ Good clustering ({metrics.n_clusters} clusters)"

        # Rule 5: Cohesion
        if metrics.anchor_cohesion < 0.5:
            score -= 0.1
            reasons["cohesion"] = f"⚠️  Low cohesion ({metrics.anchor_cohesion:.3f})"
        else:
            score += 0.05
            reasons["cohesion"] = f"✅ Good cohesion ({metrics.anchor_cohesion:.3f})"

        # Rule 6: Purity
        if metrics.anchor_purity < 0.35:  # Phase 1: P25 for MBTI (was 0.6)
            score -= 0.15
            reasons["purity"] = f"🔴 Low purity ({metrics.anchor_purity:.3f})"
        else:
            score += 0.1
            reasons["purity"] = f"✅ Good purity ({metrics.anchor_purity:.3f})"

        # Normalize score to [0, 1]
        confidence = max(0.0, min(1.0, score))

        # Make decision
        decision = "generate" if confidence >= 0.5 else "skip"

        return decision, confidence, reasons

    def validate_prediction(
        self,
        decision: str,
        actual_improvement: float
    ) -> Dict[str, str]:
        """
        Validate prediction against actual results.

        Returns classification:
        - TP: Predicted generate, actually helped
        - TN: Predicted skip, actually didn't help
        - FP: Predicted generate, actually hurt
        - FN: Predicted skip, but would have helped
        """
        improved = actual_improvement > 0

        if decision == "generate" and improved:
            return {
                "classification": "TP",
                "result": "✅ Correct! Predicted generate, actually helped",
                "accuracy": "Correct"
            }
        elif decision == "skip" and not improved:
            return {
                "classification": "TN",
                "result": "✅ Correct! Predicted skip, actually didn't help",
                "accuracy": "Correct"
            }
        elif decision == "generate" and not improved:
            return {
                "classification": "FP",
                "result": "❌ Wrong! Predicted generate, but hurt performance",
                "accuracy": "Incorrect"
            }
        else:  # skip and improved
            return {
                "classification": "FN",
                "result": "❌ Wrong! Predicted skip, but would have helped",
                "accuracy": "Incorrect"
            }


def example_usage():
    """Example of how to use the predictor."""

    predictor = QualityGatePredictor()

    # Example 1: Good case (ISTJ with 281 samples)
    metrics_good = ClassMetrics(
        n_samples=281,
        n_clusters=6,
        baseline_f1=0.024,
        anchor_cohesion=0.65,
        anchor_purity=0.75,
        anchor_separation=0.60,
        anchor_quality_score=0.67
    )

    decision, confidence, reasons = predictor.predict(metrics_good)

    print("=" * 80)
    print("EXAMPLE 1: ISTJ (Good Case)")
    print("=" * 80)
    print(f"Decision: {decision}")
    print(f"Confidence: {confidence:.2f}")
    print("\nReasons:")
    for key, reason in reasons.items():
        print(f"  {reason}")

    # Simulate actual result
    actual_improvement = +6.34  # From our data
    validation = predictor.validate_prediction(decision, actual_improvement)
    print(f"\nValidation: {validation['result']}")
    print(f"Classification: {validation['classification']}")

    # Example 2: Bad case (ESFJ with 51 samples)
    metrics_bad = ClassMetrics(
        n_samples=51,
        n_clusters=4,
        baseline_f1=0.125,
        anchor_cohesion=0.45,
        anchor_purity=0.55,
        anchor_separation=0.50,
        anchor_quality_score=0.50
    )

    decision, confidence, reasons = predictor.predict(metrics_bad)

    print("\n" + "=" * 80)
    print("EXAMPLE 2: ESFJ (Risky Case)")
    print("=" * 80)
    print(f"Decision: {decision}")
    print(f"Confidence: {confidence:.2f}")
    print("\nReasons:")
    for key, reason in reasons.items():
        print(f"  {reason}")

    # Simulate actual result
    actual_improvement = -2.5  # Hypothetical degradation
    validation = predictor.validate_prediction(decision, actual_improvement)
    print(f"\nValidation: {validation['result']}")
    print(f"Classification: {validation['classification']}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX AFTER MANY TESTS:")
    print("=" * 80)
    print("""
                      Actual Result
                  Helped  |  Hurt/Neutral
    Predicted  -------------------------
    Generate   |   TP    |     FP
    Skip       |   FN    |     TN

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """)


if __name__ == "__main__":
    example_usage()
