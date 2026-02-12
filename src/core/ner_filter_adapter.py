#!/usr/bin/env python3
"""
NER Filter Adapter

Maps NER sentence data to the geometric filter interface.
Existing filters expect: (synthetic_embeddings, real_embeddings, real_labels, target_class)

For NER:
- Embeddings = sentence-level embeddings (768-dim, same all-mpnet-base-v2)
- Labels = "dominant entity type" per sentence (most frequent B- tag type)
- Target class = entity type being augmented (e.g., "PER")

This adapter does NOT modify any filter code. It only prepares the inputs.

Usage:
    from core.ner_filter_adapter import assign_dominant_entity_types, apply_ner_filter

    labels = assign_dominant_entity_types(sentences)
    filtered_sents, filtered_embs = apply_ner_filter(
        filter_obj, filter_type, synthetic_sents, synthetic_embs,
        real_embs, real_labels, "PER", target_n=50
    )
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

from core.geometric_filter import LOFFilter, CombinedGeometricFilter
from core.filter_cascade import FilterCascade
from core.embedding_guided_sampler import EmbeddingGuidedSampler


def assign_dominant_entity_types(sentences: List[Dict]) -> np.ndarray:
    """Assign a dominant entity type label to each sentence.

    The dominant type is the most frequent entity type (by B- tag count).
    Sentences with no entities get label "O".
    Ties are broken alphabetically for determinism.

    Args:
        sentences: List of {"tokens": [...], "ner_tags": [...]}

    Returns:
        np.ndarray of string labels, one per sentence
    """
    labels = []
    for sent in sentences:
        type_counts = Counter()
        for tag in sent["ner_tags"]:
            if tag.startswith("B-"):
                type_counts[tag[2:]] += 1

        if not type_counts:
            labels.append("O")
        else:
            # Most common, break ties alphabetically
            max_count = max(type_counts.values())
            top_types = sorted(t for t, c in type_counts.items() if c == max_count)
            labels.append(top_types[0])

    return np.array(labels)


def apply_ner_filter(
    filter_obj,
    filter_type: str,
    candidate_sentences: List[Dict],
    candidate_embeddings: np.ndarray,
    candidate_texts: List[str],
    real_embeddings: np.ndarray,
    real_labels: np.ndarray,
    target_entity_type: str,
    target_n: int
) -> Tuple[List[Dict], np.ndarray, List[str], Dict]:
    """Apply a geometric filter to candidate NER sentences.

    Uses the same filter interface as classification experiments.
    The "class" is the target entity type.

    Args:
        filter_obj: Filter instance (LOFFilter, FilterCascade, etc.) or None
        filter_type: Filter type string ("lof", "cascade", "combined", "none")
        candidate_sentences: Parsed NER sentences to filter
        candidate_embeddings: (N, 768) embeddings of candidates
        candidate_texts: Plain text versions of candidates
        real_embeddings: (M, 768) real training sentence embeddings
        real_labels: (M,) dominant entity type labels for real sentences
        target_entity_type: Entity type being augmented (used as target_class)
        target_n: Maximum number of samples to select

    Returns:
        (filtered_sentences, filtered_embeddings, filtered_texts, stats)
    """
    if len(candidate_embeddings) == 0:
        return [], np.array([]).reshape(0, 768), [], {"n_candidates": 0, "n_selected": 0}

    # No filter — return all (truncated to target_n)
    if filter_obj is None or filter_type == "none":
        n = min(len(candidate_sentences), target_n)
        return (
            candidate_sentences[:n],
            candidate_embeddings[:n],
            candidate_texts[:n],
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": n,
                "method": "none",
                "pct_accepted": 100.0
            }
        )

    # LOF Filter
    if filter_type == "lof":
        filtered_emb, mask, scores = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_entity_type
        )
        passed_indices = np.where(mask)[0]
        # Truncate to target_n
        if len(passed_indices) > target_n:
            # Keep top-scoring
            top = np.argsort(scores[passed_indices])[-target_n:]
            passed_indices = passed_indices[top]
            filtered_emb = candidate_embeddings[passed_indices]

        return (
            [candidate_sentences[i] for i in passed_indices],
            filtered_emb,
            [candidate_texts[i] for i in passed_indices],
            {
                "n_candidates": len(candidate_embeddings),
                "n_passed_filter": int(mask.sum()),
                "n_selected": len(passed_indices),
                "pct_accepted": 100 * mask.sum() / len(mask),
                "mean_lof_score": float(scores.mean()) if len(scores) > 0 else 0.0,
            }
        )

    # Combined Geometric Filter
    if filter_type == "combined":
        filtered_emb, mask, stats = filter_obj.filter(
            candidate_embeddings, real_embeddings, real_labels, target_entity_type
        )
        passed_indices = np.where(mask)[0]
        if len(passed_indices) > target_n:
            passed_indices = passed_indices[:target_n]
            filtered_emb = candidate_embeddings[passed_indices]

        return (
            [candidate_sentences[i] for i in passed_indices],
            filtered_emb,
            [candidate_texts[i] for i in passed_indices],
            stats
        )

    # Filter Cascade
    if filter_type == "cascade":
        filtered_emb, avg_quality, details = filter_obj.filter_samples(
            candidates=candidate_embeddings,
            real_embeddings=real_embeddings,
            real_labels=real_labels,
            target_class=target_entity_type,
            target_count=target_n
        )

        # Get indices of selected samples
        class_mask = real_labels == target_entity_type
        class_embs = real_embeddings[class_mask]
        anchor = class_embs.mean(axis=0) if len(class_embs) > 0 else real_embeddings.mean(axis=0)

        scores, _ = filter_obj.compute_quality_scores(
            candidate_embeddings, anchor, real_embeddings, real_labels, target_entity_type
        )
        top_idx = np.argsort(scores)[-len(filtered_emb):]

        return (
            [candidate_sentences[i] for i in top_idx],
            filtered_emb,
            [candidate_texts[i] for i in top_idx],
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(filtered_emb),
                "avg_quality": avg_quality,
                "pct_accepted": 100 * len(filtered_emb) / max(1, len(candidate_embeddings)),
            }
        )

    # Embedding Guided Sampler
    if filter_type == "embedding_guided":
        class_mask = real_labels == target_entity_type
        class_embs = real_embeddings[class_mask]

        selected_texts, selected_embs, scores = filter_obj.select_samples(
            candidate_embeddings,
            candidate_texts,
            class_embs,
            target_n,
            class_label=target_entity_type
        )

        # Map back to sentences by matching texts
        text_to_sent = {t: s for t, s in zip(candidate_texts, candidate_sentences)}
        selected_sents = [text_to_sent.get(t, candidate_sentences[0]) for t in selected_texts]

        return (
            selected_sents,
            selected_embs if len(selected_embs) > 0 else np.array([]).reshape(0, candidate_embeddings.shape[1]),
            selected_texts,
            {
                "n_candidates": len(candidate_embeddings),
                "n_selected": len(selected_texts),
                "pct_accepted": 100 * len(selected_texts) / max(1, len(candidate_embeddings)),
            }
        )

    raise ValueError(f"Unknown filter type: {filter_type}")
