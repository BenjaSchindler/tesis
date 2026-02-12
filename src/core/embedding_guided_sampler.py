#!/usr/bin/env python3
"""
Embedding-Guided Sampler

Accepts only LLM samples that fill gaps in the embedding space.
Uses coverage-based selection to maximize distribution improvement.

Key idea: Generate many LLM samples, but only keep those that
land in underrepresented regions of the embedding space.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageScore:
    """Score for how much a sample improves coverage."""
    sample_idx: int
    coverage_gain: float
    nearest_real_distance: float
    density_score: float
    quality_score: float
    combined_score: float


class EmbeddingGuidedSampler:
    """
    Select LLM samples that fill gaps in the embedding space.

    This addresses the problem that LLM samples tend to cluster
    near class centroids. By selecting only samples that fill gaps,
    we get better distribution coverage similar to SMOTE.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        coverage_weight: float = 0.6,
        quality_weight: float = 0.4,
        min_distance_threshold: float = 0.1
    ):
        """
        Initialize the sampler.

        Args:
            k_neighbors: Number of neighbors for density estimation
            coverage_weight: Weight for coverage gain score
            quality_weight: Weight for quality (centroid proximity) score
            min_distance_threshold: Minimum distance from existing samples
        """
        self.k_neighbors = k_neighbors
        self.coverage_weight = coverage_weight
        self.quality_weight = quality_weight
        self.min_distance_threshold = min_distance_threshold

    def compute_coverage_gain(
        self,
        candidate_emb: np.ndarray,
        real_embeddings: np.ndarray,
        already_selected: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute how much a candidate improves coverage.

        Coverage gain is high when:
        - Candidate is far from nearest real sample (fills gap)
        - Candidate is in a low-density region

        Args:
            candidate_emb: Candidate embedding (1, dim)
            real_embeddings: Real sample embeddings (n, dim)
            already_selected: Previously selected candidates

        Returns:
            Coverage gain score (higher = better)
        """
        if len(candidate_emb.shape) == 1:
            candidate_emb = candidate_emb.reshape(1, -1)

        # Distance to nearest real sample
        distances_to_real = euclidean_distances(candidate_emb, real_embeddings)[0]
        min_distance_real = np.min(distances_to_real)

        # If too close to existing, low gain
        if min_distance_real < self.min_distance_threshold:
            return 0.0

        # Distance to already selected (if any)
        if already_selected is not None and len(already_selected) > 0:
            distances_to_selected = euclidean_distances(candidate_emb, already_selected)[0]
            min_distance_selected = np.min(distances_to_selected)

            # Penalize if too close to already selected
            if min_distance_selected < self.min_distance_threshold:
                return 0.0

        # Density score: inverse of average distance to k nearest
        k = min(self.k_neighbors, len(real_embeddings))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(real_embeddings)
        knn_distances, _ = nn.kneighbors(candidate_emb)
        avg_knn_distance = np.mean(knn_distances)

        # Higher avg distance = lower density = higher coverage gain
        coverage_gain = avg_knn_distance * min_distance_real

        return float(coverage_gain)

    def compute_quality_score(
        self,
        candidate_emb: np.ndarray,
        class_centroid: np.ndarray,
        class_embeddings: np.ndarray
    ) -> float:
        """
        Compute quality score based on proximity to class distribution.

        Quality is high when:
        - Candidate is reasonably close to class centroid
        - Candidate is within the class boundary

        Args:
            candidate_emb: Candidate embedding
            class_centroid: Centroid of class embeddings
            class_embeddings: All class embeddings

        Returns:
            Quality score (higher = better)
        """
        if len(candidate_emb.shape) == 1:
            candidate_emb = candidate_emb.reshape(1, -1)

        # Distance to centroid
        dist_to_centroid = euclidean_distances(candidate_emb, class_centroid.reshape(1, -1))[0, 0]

        # Max distance in class (to normalize)
        real_dists_to_centroid = euclidean_distances(class_embeddings, class_centroid.reshape(1, -1)).flatten()
        max_dist = np.max(real_dists_to_centroid)

        # Quality score: closer to centroid = higher quality
        # But cap at max class distance (samples outside class boundary get 0)
        if dist_to_centroid > max_dist * 1.5:  # Allow some margin
            return 0.0

        quality = 1.0 - (dist_to_centroid / (max_dist * 1.5))
        return float(max(0, quality))

    def select_samples(
        self,
        candidate_embeddings: np.ndarray,
        candidate_texts: List[str],
        real_embeddings: np.ndarray,
        target_n: int,
        class_label: Optional[str] = None
    ) -> Tuple[List[str], np.ndarray, List[CoverageScore]]:
        """
        Select best samples from candidates using coverage + quality scoring.

        Args:
            candidate_embeddings: All candidate embeddings (n_candidates, dim)
            candidate_texts: Corresponding texts
            real_embeddings: Real class embeddings
            target_n: Number of samples to select
            class_label: Class label (for logging)

        Returns:
            Tuple of (selected_texts, selected_embeddings, scores)
        """
        if len(candidate_embeddings) <= target_n:
            # Return all if not enough candidates
            scores = [CoverageScore(i, 0, 0, 0, 0, 0) for i in range(len(candidate_embeddings))]
            return candidate_texts, candidate_embeddings, scores

        # Compute class statistics
        class_centroid = real_embeddings.mean(axis=0)

        # Score all candidates
        scores = []
        for i, emb in enumerate(candidate_embeddings):
            coverage = self.compute_coverage_gain(emb, real_embeddings)
            quality = self.compute_quality_score(emb, class_centroid, real_embeddings)

            # Nearest real distance
            distances_to_real = euclidean_distances(emb.reshape(1, -1), real_embeddings)[0]
            nearest_dist = np.min(distances_to_real)

            # Density (inverse of avg knn distance)
            k = min(self.k_neighbors, len(real_embeddings))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(real_embeddings)
            knn_dists, _ = nn.kneighbors(emb.reshape(1, -1))
            density = 1.0 / (1.0 + np.mean(knn_dists))

            # Combined score
            combined = (self.coverage_weight * coverage +
                       self.quality_weight * quality)

            scores.append(CoverageScore(
                sample_idx=i,
                coverage_gain=coverage,
                nearest_real_distance=nearest_dist,
                density_score=density,
                quality_score=quality,
                combined_score=combined
            ))

        # Sort by combined score (descending)
        scores.sort(key=lambda x: x.combined_score, reverse=True)

        # Greedy selection with diversity constraint
        selected_indices = []
        selected_embeddings = []

        for score in scores:
            if len(selected_indices) >= target_n:
                break

            idx = score.sample_idx
            emb = candidate_embeddings[idx]

            # Check diversity constraint against already selected
            if len(selected_embeddings) > 0:
                selected_array = np.array(selected_embeddings)
                min_dist_to_selected = np.min(
                    euclidean_distances(emb.reshape(1, -1), selected_array)
                )

                # Skip if too similar to already selected
                if min_dist_to_selected < self.min_distance_threshold:
                    continue

            selected_indices.append(idx)
            selected_embeddings.append(emb)

        # Get corresponding texts
        selected_texts = [candidate_texts[i] for i in selected_indices]
        selected_embs = np.array(selected_embeddings) if selected_embeddings else np.array([])
        selected_scores = [s for s in scores if s.sample_idx in selected_indices]

        logger.info(f"Selected {len(selected_texts)}/{target_n} samples for {class_label}")
        if selected_scores:
            avg_coverage = np.mean([s.coverage_gain for s in selected_scores])
            avg_quality = np.mean([s.quality_score for s in selected_scores])
            logger.info(f"  Avg coverage gain: {avg_coverage:.4f}, Avg quality: {avg_quality:.4f}")

        return selected_texts, selected_embs, selected_scores


def embedding_guided_selection(
    candidate_texts: List[str],
    candidate_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    target_n: int,
    coverage_weight: float = 0.6,
    quality_weight: float = 0.4
) -> Tuple[List[str], np.ndarray]:
    """
    Convenience function for embedding-guided selection.

    Args:
        candidate_texts: Generated texts
        candidate_embeddings: Embeddings of generated texts
        real_embeddings: Real class embeddings
        target_n: Number to select
        coverage_weight: Weight for coverage score
        quality_weight: Weight for quality score

    Returns:
        Tuple of (selected_texts, selected_embeddings)
    """
    sampler = EmbeddingGuidedSampler(
        coverage_weight=coverage_weight,
        quality_weight=quality_weight
    )

    texts, embs, _ = sampler.select_samples(
        candidate_embeddings,
        candidate_texts,
        real_embeddings,
        target_n
    )

    return texts, embs


class HybridEmbeddingGuidedGenerator:
    """
    Complete pipeline: Generate many LLM samples, select best, then SMOTE.
    """

    def __init__(
        self,
        llm_provider: str = "google",
        llm_model: str = "gemini-3-flash-preview",
        oversample_factor: float = 3.0,
        coverage_weight: float = 0.6,
        quality_weight: float = 0.4
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.oversample_factor = oversample_factor
        self.sampler = EmbeddingGuidedSampler(
            coverage_weight=coverage_weight,
            quality_weight=quality_weight
        )

    def generate_and_select(
        self,
        llm_generator,  # AsyncLLMGenerator or similar
        prompts: List[str],
        real_embeddings: np.ndarray,
        embed_fn: callable,
        target_n: int
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate many samples, embed them, select best.

        Args:
            llm_generator: Generator to produce texts
            prompts: Prompts to generate from
            real_embeddings: Real class embeddings
            embed_fn: Function to embed texts
            target_n: Number of samples to keep

        Returns:
            Tuple of (selected_texts, selected_embeddings)
        """
        # Generate candidates (synchronous wrapper for async generator)
        import asyncio

        async def generate_all():
            all_texts = []
            for prompt in prompts:
                texts = await llm_generator.generate_batch([prompt])
                all_texts.extend(texts)
            return all_texts

        # Run async generation
        try:
            loop = asyncio.get_event_loop()
            candidate_texts = loop.run_until_complete(generate_all())
        except RuntimeError:
            # No event loop, create new one
            candidate_texts = asyncio.run(generate_all())

        if not candidate_texts:
            return [], np.array([])

        # Embed candidates
        candidate_embeddings = embed_fn(candidate_texts)

        # Select best using embedding guidance
        return self.sampler.select_samples(
            candidate_embeddings,
            candidate_texts,
            real_embeddings,
            target_n
        )[:2]  # Return only texts and embeddings
