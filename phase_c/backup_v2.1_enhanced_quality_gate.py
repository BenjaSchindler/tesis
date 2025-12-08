"""
Phase 2: Enhanced Quality Gate Predictor

Improved quality gate with:
1. Probabilistic decision making (vs hard threshold)
2. Purity-based budget reduction
3. F1-based budget scaling
4. Lower threshold (0.35 vs 0.40)

Author: Benja
Date: 2025-10-30

BACKUP: Phase C v2.1 successful version (2025-01-15)
This version fixed the probabilistic gate non-determinism bug by seeding RNG.
Result: +1.72% MID-tier improvement (17× better than target!)
"""

import numpy as np
from typing import Dict, Tuple, Optional


class EnhancedQualityGate:
    """
    Enhanced quality gate predictor with Phase 2 improvements.

    Key Changes from Phase 1:
    - Threshold lowered: 0.40 → 0.35
    - Probabilistic decisions (not just hard threshold)
    - Purity gating: extra reduction if purity <0.30
    - F1 scaling: adjust budgets based on baseline F1
    """

    def __init__(
        self,
        # Phase 2 thresholds (lowered from Phase 1)
        min_anchor_quality: float = 0.35,  # Was 0.40 in Phase 1
        min_anchor_purity: float = 0.30,   # Was 0.35 in Phase 1
        # F1 gating
        f1_skip_threshold: float = 0.60,   # Was 0.65 in Phase 1
        f1_high_threshold: float = 0.45,
        f1_low_threshold: float = 0.15,
        # Budget multipliers
        purity_low_threshold: float = 0.30,
        purity_low_multiplier: float = 0.3,
        f1_high_multiplier: float = 0.5,
        f1_low_multiplier: float = 1.5,
        # Decision mode
        decision_mode: str = "probabilistic",  # "probabilistic" or "threshold"
        # Base budget target
        target_ratio: float = 0.08,  # 8% synthetic/real ratio
        # Phase C v2.1: Seed for deterministic probabilistic decisions
        seed: Optional[int] = None
    ):
        """
        Initialize enhanced quality gate.

        Args:
            min_anchor_quality: Minimum quality score to generate
            min_anchor_purity: Minimum purity to avoid heavy reduction
            f1_skip_threshold: Skip generation if F1 above this
            f1_high_threshold: F1 above which to reduce budget
            f1_low_threshold: F1 below which to increase budget
            purity_low_threshold: Purity below which to reduce budget
            purity_low_multiplier: Multiplier when purity is low
            f1_high_multiplier: Multiplier when F1 is high
            f1_low_multiplier: Multiplier when F1 is low
            decision_mode: "probabilistic" or "threshold"
            target_ratio: Target synthetic/real ratio (default 8%)
        """
        self.min_anchor_quality = min_anchor_quality
        self.min_anchor_purity = min_anchor_purity
        self.f1_skip_threshold = f1_skip_threshold
        self.f1_high_threshold = f1_high_threshold
        self.f1_low_threshold = f1_low_threshold
        self.purity_low_threshold = purity_low_threshold
        self.purity_low_multiplier = purity_low_multiplier
        self.f1_high_multiplier = f1_high_multiplier
        self.f1_low_multiplier = f1_low_multiplier
        self.decision_mode = decision_mode
        self.target_ratio = target_ratio

        # Phase C v2.1: Initialize RNG for deterministic probabilistic decisions
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def predict(
        self,
        n_samples: int,
        baseline_f1: float,
        quality_score: float,
        purity: float,
        cohesion: Optional[float] = None,
        n_clusters: int = 12
    ) -> Dict[str, any]:
        """
        Predict whether to generate synthetics and calculate budget.

        Args:
            n_samples: Number of training samples for this class
            baseline_f1: Baseline F1 score before augmentation
            quality_score: Composite anchor quality score
            purity: Anchor purity score
            cohesion: Optional anchor cohesion score
            n_clusters: Number of clusters

        Returns:
            Dictionary with decision, confidence, budget, and metrics
        """
        # Check F1 skip threshold first
        if baseline_f1 >= self.f1_skip_threshold:
            return {
                "decision": "skip",
                "reason": f"f1_too_high ({baseline_f1:.3f} >= {self.f1_skip_threshold})",
                "confidence": 0.0,
                "budget": 0,
                "metrics": {
                    "n_samples": n_samples,
                    "baseline_f1": baseline_f1,
                    "quality_score": quality_score,
                    "purity": purity
                }
            }

        # Calculate confidence based on quality
        confidence = self._calculate_confidence(quality_score, purity, cohesion)

        # Make decision
        if self.decision_mode == "probabilistic":
            decision, reason = self._probabilistic_decision(confidence)
        else:
            decision, reason = self._threshold_decision(confidence)

        # Calculate dynamic budget
        budget = self._calculate_enhanced_budget(
            n_samples=n_samples,
            quality_score=quality_score,
            purity=purity,
            baseline_f1=baseline_f1
        )

        return {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "budget": budget,
            "budget_per_cluster": max(1, budget // n_clusters),
            "metrics": {
                "n_samples": n_samples,
                "baseline_f1": baseline_f1,
                "quality_score": quality_score,
                "purity": purity,
                "cohesion": cohesion,
                "n_clusters": n_clusters
            }
        }

    def _calculate_confidence(
        self,
        quality_score: float,
        purity: float,
        cohesion: Optional[float]
    ) -> float:
        """
        Calculate confidence that generation will help.

        Phase 2 enhancement: Incorporate purity more heavily.
        """
        # Base confidence from quality (Phase 1 logic)
        if quality_score < 0.25:
            quality_conf = 0.10
        elif quality_score < 0.30:
            quality_conf = 0.20
        elif quality_score < 0.35:
            quality_conf = 0.30
        elif quality_score < 0.40:
            quality_conf = 0.40
        elif quality_score < 0.45:
            quality_conf = 0.50
        elif quality_score < 0.50:
            quality_conf = 0.60
        else:
            quality_conf = 0.75

        # Purity penalty (Phase 2 enhancement)
        if purity < 0.10:
            purity_conf = 0.05  # Disaster zone
        elif purity < 0.20:
            purity_conf = 0.15
        elif purity < 0.30:
            purity_conf = 0.30
        elif purity < 0.40:
            purity_conf = 0.50
        elif purity < 0.50:
            purity_conf = 0.70
        else:
            purity_conf = 0.90  # High purity, confident

        # Cohesion bonus (if available)
        cohesion_conf = 0.5  # Neutral if not provided
        if cohesion is not None:
            if cohesion > 0.60:
                cohesion_conf = 0.70
            elif cohesion > 0.50:
                cohesion_conf = 0.60
            elif cohesion < 0.30:
                cohesion_conf = 0.30

        # Weighted combination
        # Purity is most important (learned from ISTJ disaster)
        confidence = (
            0.3 * quality_conf +
            0.5 * purity_conf +
            0.2 * cohesion_conf
        )

        return np.clip(confidence, 0.0, 1.0)

    def _probabilistic_decision(self, confidence: float) -> Tuple[str, str]:
        """
        Make probabilistic decision based on confidence.

        Instead of hard threshold, use confidence as probability.
        This catches borderline cases (like ISTJ at 0.40).

        Phase C v2.1: Uses seeded RNG if provided for deterministic results.
        """
        # Random draw based on confidence
        if self.rng is not None:
            # Phase C v2.1: Deterministic (seeded) random draw
            should_generate = self.rng.random() < confidence
        else:
            # Fallback to numpy global RNG (non-deterministic)
            should_generate = np.random.random() < confidence

        if should_generate:
            decision = "generate"
            reason = f"probabilistic_accept (confidence={confidence:.2f})"
        else:
            decision = "skip"
            reason = f"probabilistic_reject (confidence={confidence:.2f})"

        return decision, reason

    def _threshold_decision(self, confidence: float) -> Tuple[str, str]:
        """
        Make threshold-based decision (Phase 1 style, but with lower threshold).
        """
        if confidence >= self.min_anchor_quality:
            decision = "generate"
            reason = f"quality_sufficient (confidence={confidence:.2f} >= {self.min_anchor_quality})"
        else:
            decision = "skip"
            reason = f"quality_insufficient (confidence={confidence:.2f} < {self.min_anchor_quality})"

        return decision, reason

    def _calculate_enhanced_budget(
        self,
        n_samples: int,
        quality_score: float,
        purity: float,
        baseline_f1: float
    ) -> int:
        """
        Calculate enhanced budget with Phase 2 improvements.

        Improvements over Phase 1:
        1. Purity-based reduction
        2. F1-based scaling
        3. Combined multiplicative effects
        """
        # Base budget (8% target)
        base_budget = int(n_samples * self.target_ratio)

        # 1. Quality multiplier (Phase 1 logic)
        if quality_score < 0.35:
            quality_mult = 0.1
        elif quality_score < 0.40:
            quality_mult = 0.3
        elif quality_score < 0.50:
            quality_mult = 0.7
        else:
            quality_mult = 1.0

        # 2. Purity multiplier (Phase 2 NEW)
        if purity < self.purity_low_threshold:
            purity_mult = self.purity_low_multiplier  # Extra 70% reduction
        else:
            purity_mult = 1.0

        # 3. F1 multiplier (Phase 2 NEW)
        if baseline_f1 > self.f1_high_threshold:
            f1_mult = self.f1_high_multiplier  # High performers need less
        elif baseline_f1 < self.f1_low_threshold:
            f1_mult = self.f1_low_multiplier  # Low performers can use more
        else:
            f1_mult = 1.0

        # Combine multipliers
        total_mult = quality_mult * purity_mult * f1_mult

        # Final budget
        budget = max(10, int(base_budget * total_mult))

        return budget

    def calculate_expected_contamination(
        self,
        budget: int,
        n_samples: int,
        purity: float
    ) -> float:
        """
        Calculate expected contamination using Proportional Contamination Theory.

        Contamination = (Synthetics / Reals) × (1 - Purity)
        """
        if n_samples == 0:
            return 0.0

        ratio = budget / n_samples
        contamination = ratio * (1.0 - purity)

        return contamination


def test_enhanced_quality_gate():
    """Test enhanced quality gate with Phase 1 cases."""
    gate = EnhancedQualityGate(
        min_anchor_quality=0.35,
        decision_mode="probabilistic"
    )

    print("🚪 Enhanced Quality Gate Test (Phase 2)\n")

    # Test cases from Phase 1 complete validation
    test_cases = [
        {
            "class": "ISTJ",
            "n_samples": 994,
            "baseline_f1": 0.0759,
            "quality_score": 0.256,
            "purity": 0.025,
            "cohesion": 0.611
        },
        {
            "class": "INFJ",
            "n_samples": 11970,
            "baseline_f1": 0.477,
            "quality_score": 0.349,
            "purity": 0.244,
            "cohesion": 0.626
        },
        {
            "class": "ENFP",
            "n_samples": 4934,
            "baseline_f1": 0.378,
            "quality_score": 0.316,
            "purity": 0.50,  # Estimated
            "cohesion": 0.55  # Estimated
        },
        {
            "class": "INFP",
            "n_samples": 9707,
            "baseline_f1": 0.500,
            "quality_score": 0.341,
            "purity": 0.50,  # Estimated
            "cohesion": 0.60  # Estimated
        }
    ]

    for test in test_cases:
        result = gate.predict(
            n_samples=test["n_samples"],
            baseline_f1=test["baseline_f1"],
            quality_score=test["quality_score"],
            purity=test["purity"],
            cohesion=test["cohesion"]
        )

        contamination = gate.calculate_expected_contamination(
            budget=result["budget"],
            n_samples=test["n_samples"],
            purity=test["purity"]
        )

        print(f"📊 {test['class']}")
        print(f"   Samples: {test['n_samples']}")
        print(f"   Baseline F1: {test['baseline_f1']:.3f}")
        print(f"   Quality: {test['quality_score']:.3f}")
        print(f"   Purity: {test['purity']:.3f}")
        print(f"   Cohesion: {test['cohesion']:.3f}")
        print(f"   → Decision: {result['decision'].upper()}")
        print(f"   → Confidence: {result['confidence']:.3f}")
        print(f"   → Budget: {result['budget']}")
        print(f"   → Expected Contamination: {contamination*100:.2f}%")
        print(f"   → Reason: {result['reason']}")
        print()

    # Test probabilistic behavior (run ISTJ 10 times)
    print("🎲 Probabilistic Decision Test (ISTJ, 10 runs)\n")
    istj_case = test_cases[0]
    decisions = {"generate": 0, "skip": 0}

    for i in range(10):
        result = gate.predict(
            n_samples=istj_case["n_samples"],
            baseline_f1=istj_case["baseline_f1"],
            quality_score=istj_case["quality_score"],
            purity=istj_case["purity"],
            cohesion=istj_case["cohesion"]
        )
        decisions[result["decision"]] += 1

    print(f"Generate: {decisions['generate']}/10 ({decisions['generate']*10}%)")
    print(f"Skip: {decisions['skip']}/10 ({decisions['skip']*10}%)")
    print(f"Expected (confidence={istj_case['quality_score']}): ~{gate._calculate_confidence(istj_case['quality_score'], istj_case['purity'], istj_case['cohesion'])*100:.0f}% generate")


if __name__ == "__main__":
    test_enhanced_quality_gate()
