#!/usr/bin/env python3
"""
Rejection Analyzer for Closed-Loop Regeneration

Diagnoses WHY geometric filters reject synthetic samples by classifying
each rejection into one of 5 geometric failure categories:

1. CROSS_CLASS       - Sample is closer to another class's centroid
2. DISTANCE_OUTLIER  - Too far from target class centroid
3. DENSITY_OUTLIER   - Low local density (LOF-based)
4. DIRECTION_OUTLIER - Wrong angular orientation (low cosine similarity)
5. GENERIC_COLLAPSE  - Batch-level: accepted pool lacks diversity

Usage:
    from core.rejection_analyzer import RejectionAnalyzer

    analyzer = RejectionAnalyzer(real_embeddings, real_labels)
    diagnosis = analyzer.analyze_batch(
        candidate_embeddings, accepted_mask, target_class="spam"
    )
    print(diagnosis.dominant_failure)  # e.g. "DISTANCE_OUTLIER"
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import cdist


# Failure categories
CROSS_CLASS = "CROSS_CLASS"
DISTANCE_OUTLIER = "DISTANCE_OUTLIER"
DENSITY_OUTLIER = "DENSITY_OUTLIER"
DIRECTION_OUTLIER = "DIRECTION_OUTLIER"
GENERIC_COLLAPSE = "GENERIC_COLLAPSE"


@dataclass
class RejectionDiagnosis:
    """Diagnosis for a single rejected sample."""
    sample_idx: int
    primary_reason: str
    severity: float  # 0.0 (borderline) to 1.0 (extreme)
    confused_class: Optional[str] = None  # Only for CROSS_CLASS
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class BatchDiagnosis:
    """Aggregate diagnosis for a batch of candidates."""
    n_candidates: int
    n_accepted: int
    n_rejected: int
    acceptance_rate: float
    rejection_distribution: Dict[str, int]
    dominant_failure: str
    diversity_ratio: float  # accepted vs real diversity (1.0 = same)
    mean_severity: float
    confused_classes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class RejectionAnalyzer:
    """
    Analyze why synthetic samples were rejected by geometric filters.

    Precomputes per-class statistics from real embeddings so that
    diagnosis is fast during the iteration loop.
    """

    def __init__(
        self,
        real_embeddings: np.ndarray,
        real_labels: np.ndarray,
        lof_n_neighbors: int = 10
    ):
        self.real_embeddings = real_embeddings
        self.real_labels = real_labels
        self.classes = list(np.unique(real_labels))

        # Precompute per-class statistics
        self.centroids: Dict[str, np.ndarray] = {}
        self.distance_95th: Dict[str, float] = {}
        self.cosine_sim_5th: Dict[str, float] = {}
        self.real_diversity: Dict[str, float] = {}
        self.lof_models: Dict[str, Optional[LocalOutlierFactor]] = {}

        for cls in self.classes:
            mask = real_labels == cls
            class_embs = real_embeddings[mask]

            # Centroid
            centroid = class_embs.mean(axis=0)
            self.centroids[cls] = centroid

            # Distance distribution (for DISTANCE_OUTLIER threshold)
            dists = np.linalg.norm(class_embs - centroid, axis=1)
            self.distance_95th[cls] = float(np.percentile(dists, 95))

            # Cosine similarity distribution (for DIRECTION_OUTLIER threshold)
            centroid_2d = centroid.reshape(1, -1)
            cosine_sims = 1 - cdist(class_embs, centroid_2d, metric='cosine').flatten()
            self.cosine_sim_5th[cls] = float(np.percentile(cosine_sims, 5))

            # Diversity: mean pairwise distance among real samples
            if len(class_embs) > 1:
                # Subsample for efficiency if class is large
                n_sample = min(len(class_embs), 50)
                idx = np.random.choice(len(class_embs), n_sample, replace=False)
                pairwise = cdist(class_embs[idx], class_embs[idx], metric='euclidean')
                np.fill_diagonal(pairwise, np.nan)
                self.real_diversity[cls] = float(np.nanmean(pairwise))
            else:
                self.real_diversity[cls] = 1.0

            # LOF model for density analysis
            if len(class_embs) > lof_n_neighbors + 1:
                k = min(lof_n_neighbors, len(class_embs) - 1)
                lof = LocalOutlierFactor(
                    n_neighbors=k, novelty=True, contamination=0.1
                )
                lof.fit(class_embs)
                self.lof_models[cls] = lof
            else:
                self.lof_models[cls] = None

    def diagnose_sample(
        self,
        sample_embedding: np.ndarray,
        target_class: str
    ) -> RejectionDiagnosis:
        """Diagnose a single rejected sample."""
        sample = sample_embedding.reshape(1, -1)
        centroid = self.centroids[target_class].reshape(1, -1)

        # Compute all metrics
        dist_to_target = float(np.linalg.norm(sample - centroid))

        cosine_sim = float(
            1 - cdist(sample, centroid, metric='cosine').flatten()[0]
        )

        # LOF score
        lof_model = self.lof_models.get(target_class)
        lof_score = float(lof_model.decision_function(sample)[0]) if lof_model else 0.0

        # Distance to all class centroids (for cross-class check)
        class_distances = {}
        for cls in self.classes:
            c = self.centroids[cls].reshape(1, -1)
            class_distances[cls] = float(np.linalg.norm(sample - c))
        nearest_class = min(class_distances, key=class_distances.get)

        details = {
            "dist_to_centroid": dist_to_target,
            "cosine_sim_to_centroid": cosine_sim,
            "lof_score": lof_score,
            "nearest_class": nearest_class,
            "dist_to_nearest": class_distances[nearest_class],
            "dist_95th_threshold": self.distance_95th[target_class],
            "cosine_5th_threshold": self.cosine_sim_5th[target_class],
        }

        # Priority classification
        # 1. Cross-class contamination
        if nearest_class != target_class:
            severity = 1.0 - (class_distances[target_class] /
                              (class_distances[nearest_class] + 1e-8))
            severity = max(0.0, min(1.0, severity))
            return RejectionDiagnosis(
                sample_idx=-1, primary_reason=CROSS_CLASS,
                severity=severity, confused_class=nearest_class,
                details=details
            )

        # 2. Distance outlier
        if dist_to_target > self.distance_95th[target_class]:
            severity = min(1.0, (dist_to_target - self.distance_95th[target_class]) /
                           (self.distance_95th[target_class] + 1e-8))
            return RejectionDiagnosis(
                sample_idx=-1, primary_reason=DISTANCE_OUTLIER,
                severity=severity, details=details
            )

        # 3. Density outlier
        if lof_score < -0.5:
            severity = min(1.0, abs(lof_score + 0.5) / 1.5)
            return RejectionDiagnosis(
                sample_idx=-1, primary_reason=DENSITY_OUTLIER,
                severity=severity, details=details
            )

        # 4. Direction outlier
        if cosine_sim < self.cosine_sim_5th[target_class]:
            severity = min(1.0, (self.cosine_sim_5th[target_class] - cosine_sim) /
                           (self.cosine_sim_5th[target_class] + 1e-8))
            return RejectionDiagnosis(
                sample_idx=-1, primary_reason=DIRECTION_OUTLIER,
                severity=severity, details=details
            )

        # Default: distance outlier (borderline case)
        return RejectionDiagnosis(
            sample_idx=-1, primary_reason=DISTANCE_OUTLIER,
            severity=0.1, details=details
        )

    def analyze_batch(
        self,
        candidate_embeddings: np.ndarray,
        accepted_mask: np.ndarray,
        target_class: str
    ) -> BatchDiagnosis:
        """
        Analyze a batch of candidates (accepted + rejected).

        Args:
            candidate_embeddings: (N, D) all candidates
            accepted_mask: (N,) boolean mask of accepted samples
            target_class: the target class label

        Returns:
            BatchDiagnosis with aggregate statistics
        """
        n_candidates = len(candidate_embeddings)
        n_accepted = int(accepted_mask.sum())
        n_rejected = n_candidates - n_accepted

        if n_rejected == 0:
            return BatchDiagnosis(
                n_candidates=n_candidates,
                n_accepted=n_accepted,
                n_rejected=0,
                acceptance_rate=1.0,
                rejection_distribution={},
                dominant_failure="NONE",
                diversity_ratio=self._compute_diversity_ratio(
                    candidate_embeddings[accepted_mask], target_class
                ) if n_accepted > 1 else 1.0,
                mean_severity=0.0
            )

        # Diagnose each rejected sample
        rejected_idx = np.where(~accepted_mask)[0]
        diagnoses: List[RejectionDiagnosis] = []
        for idx in rejected_idx:
            diag = self.diagnose_sample(candidate_embeddings[idx], target_class)
            diag.sample_idx = int(idx)
            diagnoses.append(diag)

        # Aggregate rejection distribution
        rejection_dist: Dict[str, int] = {}
        confused_classes: Dict[str, int] = {}
        severities = []
        for d in diagnoses:
            rejection_dist[d.primary_reason] = rejection_dist.get(d.primary_reason, 0) + 1
            severities.append(d.severity)
            if d.confused_class:
                confused_classes[d.confused_class] = confused_classes.get(d.confused_class, 0) + 1

        dominant = max(rejection_dist, key=rejection_dist.get)

        # Compute diversity ratio for accepted samples
        if n_accepted > 1:
            diversity_ratio = self._compute_diversity_ratio(
                candidate_embeddings[accepted_mask], target_class
            )
        else:
            diversity_ratio = 0.0

        return BatchDiagnosis(
            n_candidates=n_candidates,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            acceptance_rate=n_accepted / n_candidates if n_candidates > 0 else 0.0,
            rejection_distribution=rejection_dist,
            dominant_failure=dominant,
            diversity_ratio=diversity_ratio,
            mean_severity=float(np.mean(severities)),
            confused_classes=confused_classes
        )

    def _compute_diversity_ratio(
        self,
        accepted_embeddings: np.ndarray,
        target_class: str
    ) -> float:
        """Compare diversity of accepted pool vs real data."""
        if len(accepted_embeddings) < 2:
            return 0.0

        n_sample = min(len(accepted_embeddings), 50)
        idx = np.random.choice(len(accepted_embeddings), n_sample, replace=False)
        pairwise = cdist(accepted_embeddings[idx], accepted_embeddings[idx], metric='euclidean')
        np.fill_diagonal(pairwise, np.nan)
        accepted_div = float(np.nanmean(pairwise))

        real_div = self.real_diversity.get(target_class, 1.0)
        return accepted_div / (real_div + 1e-8)
