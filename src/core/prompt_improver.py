#!/usr/bin/env python3
"""
Prompt Improver for Closed-Loop Regeneration

Translates geometric rejection diagnostics into concrete prompt modifications.
Each failure mode maps to a specific, actionable prompt change:

- DISTANCE_OUTLIER  -> Swap examples for centroid-nearest (prototypical) samples
- DENSITY_OUTLIER   -> Use examples from the cluster with best acceptance rate
- DIRECTION_OUTLIER -> Add topic keywords extracted via TF-IDF
- CROSS_CLASS       -> Add negative examples from the confused class
- GENERIC_COLLAPSE  -> Increase temperature + add diversity instruction

Usage:
    from core.prompt_improver import PromptImprover

    improver = PromptImprover(real_texts, real_labels, real_embeddings, "sms_spam")
    new_prompt, new_temp = improver.improve_prompt(
        current_prompt, batch_diagnosis, "spam", iteration=1, temperature=0.8
    )
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from core.rejection_analyzer import (
    BatchDiagnosis, CROSS_CLASS, DISTANCE_OUTLIER,
    DENSITY_OUTLIER, DIRECTION_OUTLIER, GENERIC_COLLAPSE
)


@dataclass
class PromptModification:
    """A single modification to apply to a prompt."""
    modification_type: str  # "inject_section", "replace_examples", "adjust_param"
    content: str            # Text to inject or description
    parameter_changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class PromptImprover:
    """
    Translate geometric rejection diagnostics into prompt modifications.

    Precomputes TF-IDF keywords and cluster info from real data so that
    prompt improvements are fast during the iteration loop.
    """

    def __init__(
        self,
        real_texts: List[str],
        real_labels: List[str],
        real_embeddings: np.ndarray,
        dataset_name: str,
        n_clusters: int = 3
    ):
        self.real_texts = real_texts
        self.real_labels = list(real_labels)
        self.real_embeddings = real_embeddings
        self.dataset_name = dataset_name
        self.classes = list(set(real_labels))

        # Per-class data
        self.class_texts: Dict[str, List[str]] = {}
        self.class_embeddings: Dict[str, np.ndarray] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        self.centroid_distances: Dict[str, np.ndarray] = {}

        for cls in self.classes:
            mask = np.array(self.real_labels) == cls
            self.class_texts[cls] = [t for t, m in zip(real_texts, mask) if m]
            self.class_embeddings[cls] = real_embeddings[mask]
            self.centroids[cls] = real_embeddings[mask].mean(axis=0)
            self.centroid_distances[cls] = np.linalg.norm(
                real_embeddings[mask] - self.centroids[cls], axis=1
            )

        # TF-IDF for keyword extraction
        self.tfidf = TfidfVectorizer(
            max_features=500, stop_words='english', min_df=1
        )
        self.tfidf.fit(real_texts)
        self.feature_names = self.tfidf.get_feature_names_out()

        # Per-class top keywords
        self.class_keywords: Dict[str, List[str]] = {}
        for cls in self.classes:
            self.class_keywords[cls] = self._extract_class_keywords(cls, top_n=15)

        # Clustering for density-based example selection
        self.class_clusters: Dict[str, Optional[KMeans]] = {}
        self.class_cluster_labels: Dict[str, Optional[np.ndarray]] = {}
        for cls in self.classes:
            embs = self.class_embeddings[cls]
            if len(embs) >= n_clusters * 2:
                k = min(n_clusters, len(embs) // 2)
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(embs)
                self.class_clusters[cls] = km
                self.class_cluster_labels[cls] = km.labels_
            else:
                self.class_clusters[cls] = None
                self.class_cluster_labels[cls] = None

    def _extract_class_keywords(self, target_class: str, top_n: int = 15) -> List[str]:
        """Extract top TF-IDF keywords for a class."""
        class_texts = self.class_texts[target_class]
        if not class_texts:
            return []

        tfidf_matrix = self.tfidf.transform(class_texts)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = np.argsort(mean_scores)[-top_n:][::-1]
        return [self.feature_names[i] for i in top_idx if mean_scores[i] > 0]

    def improve_prompt(
        self,
        current_prompt: str,
        batch_diagnosis: BatchDiagnosis,
        target_class: str,
        iteration: int,
        current_temperature: float
    ) -> Tuple[str, float, List[PromptModification]]:
        """
        Generate an improved prompt based on rejection diagnosis.

        Returns:
            (improved_prompt, new_temperature, list_of_modifications_applied)
        """
        modifications: List[PromptModification] = []
        new_temperature = current_temperature

        if batch_diagnosis.dominant_failure == "NONE":
            return current_prompt, current_temperature, []

        dominant = batch_diagnosis.dominant_failure

        # Apply the handler for the dominant failure mode
        if dominant == CROSS_CLASS:
            mods = self._handle_cross_class(
                target_class, batch_diagnosis.confused_classes,
                batch_diagnosis.mean_severity
            )
            modifications.extend(mods)

        elif dominant == DISTANCE_OUTLIER:
            mods = self._handle_distance_outlier(
                target_class, batch_diagnosis.mean_severity
            )
            modifications.extend(mods)

        elif dominant == DENSITY_OUTLIER:
            mods = self._handle_density_outlier(
                target_class, batch_diagnosis.mean_severity
            )
            modifications.extend(mods)

        elif dominant == DIRECTION_OUTLIER:
            mods = self._handle_direction_outlier(
                target_class, batch_diagnosis.mean_severity
            )
            modifications.extend(mods)

        # Check for generic collapse (batch-level, independent of dominant failure)
        if batch_diagnosis.diversity_ratio < 0.5:
            mods = self._handle_generic_collapse(current_temperature)
            modifications.extend(mods)
            for m in mods:
                if "temperature" in m.parameter_changes:
                    new_temperature = m.parameter_changes["temperature"]

        # Build the improved prompt
        improved = self._build_improved_prompt(current_prompt, modifications)

        return improved, new_temperature, modifications

    def select_examples_for_iteration(
        self,
        target_class: str,
        diagnosis: BatchDiagnosis,
        n_examples: int = 25
    ) -> List[str]:
        """
        Select n-shot examples optimized for the diagnosed failure mode.

        Different failure modes benefit from different example selection strategies.
        """
        class_texts = self.class_texts[target_class]
        class_embs = self.class_embeddings[target_class]
        n_select = min(n_examples, len(class_texts))

        if diagnosis.dominant_failure == "NONE" or n_select == len(class_texts):
            return class_texts[:n_select]

        dominant = diagnosis.dominant_failure

        if dominant == DISTANCE_OUTLIER:
            # Select examples nearest to centroid (most prototypical)
            dists = self.centroid_distances[target_class]
            nearest_idx = np.argsort(dists)[:n_select]
            return [class_texts[i] for i in nearest_idx]

        elif dominant == CROSS_CLASS:
            # Select examples most distant from the confused class
            confused = list(diagnosis.confused_classes.keys())
            if confused:
                confused_centroid = self.centroids[confused[0]].reshape(1, -1)
                dists = np.linalg.norm(class_embs - confused_centroid, axis=1)
                farthest_idx = np.argsort(dists)[-n_select:]
                return [class_texts[i] for i in farthest_idx]

        elif dominant == DENSITY_OUTLIER:
            # Select from the cluster with lowest rejection rate
            # (we don't track per-cluster rejection yet, so use largest cluster)
            km = self.class_clusters.get(target_class)
            if km is not None:
                labels = self.class_cluster_labels[target_class]
                cluster_sizes = np.bincount(labels)
                largest = np.argmax(cluster_sizes)
                cluster_idx = np.where(labels == largest)[0]
                selected = cluster_idx[:n_select]
                if len(selected) < n_select:
                    remaining = [i for i in range(len(class_texts)) if i not in selected]
                    selected = np.concatenate([
                        selected, np.array(remaining[:n_select - len(selected)])
                    ])
                return [class_texts[i] for i in selected]

        elif dominant == DIRECTION_OUTLIER:
            # Select examples spanning the angular range (diverse directions)
            return self._select_diverse_examples(target_class, n_select)

        elif dominant == GENERIC_COLLAPSE:
            return self._select_diverse_examples(target_class, n_select)

        # Fallback: random
        idx = np.random.choice(len(class_texts), n_select, replace=False)
        return [class_texts[i] for i in idx]

    def _select_diverse_examples(self, target_class: str, n: int) -> List[str]:
        """Farthest-point sampling for maximum diversity."""
        embs = self.class_embeddings[target_class]
        texts = self.class_texts[target_class]
        if len(texts) <= n:
            return texts

        # Start from centroid-nearest
        centroid = self.centroids[target_class]
        dists_to_centroid = np.linalg.norm(embs - centroid, axis=1)
        selected = [int(np.argmin(dists_to_centroid))]

        for _ in range(n - 1):
            # Find point farthest from all selected
            selected_embs = embs[selected]
            min_dists = cdist(embs, selected_embs).min(axis=1)
            min_dists[selected] = -1  # exclude already selected
            selected.append(int(np.argmax(min_dists)))

        return [texts[i] for i in selected]

    def _handle_distance_outlier(
        self, target_class: str, severity: float
    ) -> List[PromptModification]:
        """Samples too far from centroid -> stronger style anchoring."""
        return [PromptModification(
            modification_type="inject_section",
            content=(
                "\n# STYLE ANCHORING (CRITICAL)\n"
                "Your generated texts are drifting too far from the target style. "
                "Stay VERY close to the vocabulary, tone, and structure of the reference examples. "
                "Do NOT introduce topics, jargon, or patterns not present in the examples.\n"
            )
        )]

    def _handle_density_outlier(
        self, target_class: str, severity: float
    ) -> List[PromptModification]:
        """Samples in sparse regions -> guide toward dense regions."""
        keywords = self.class_keywords.get(target_class, [])[:8]
        keyword_str = ", ".join(keywords) if keywords else "the reference examples"
        return [PromptModification(
            modification_type="inject_section",
            content=(
                "\n# TOPIC FOCUS\n"
                f"Center your generated texts around these core themes: {keyword_str}. "
                "Avoid inventing new sub-topics not represented in the examples.\n"
            )
        )]

    def _handle_direction_outlier(
        self, target_class: str, severity: float
    ) -> List[PromptModification]:
        """Wrong semantic direction -> inject topic keywords."""
        keywords = self.class_keywords.get(target_class, [])[:10]
        if not keywords:
            return []
        keyword_str = ", ".join(keywords)
        return [PromptModification(
            modification_type="inject_section",
            content=(
                "\n# SEMANTIC DIRECTION GUIDE\n"
                f"Each generated text MUST relate to these topics: {keyword_str}. "
                "These are the defining themes of this class. Texts about unrelated "
                "topics will be rejected.\n"
            )
        )]

    def _handle_cross_class(
        self,
        target_class: str,
        confused_classes: Dict[str, int],
        severity: float
    ) -> List[PromptModification]:
        """Samples confused with another class -> add negative examples."""
        if not confused_classes:
            return []

        # Get the most common confused class
        confused_class = max(confused_classes, key=confused_classes.get)
        confused_texts = self.class_texts.get(confused_class, [])

        if not confused_texts:
            return [PromptModification(
                modification_type="inject_section",
                content=(
                    f"\n# CLASS BOUNDARY WARNING\n"
                    f"Some of your outputs are being confused with class \"{confused_class}\". "
                    f"Make sure each text CLEARLY belongs to \"{target_class}\" "
                    f"and could NOT be mistaken for \"{confused_class}\".\n"
                )
            )]

        # Show negative examples (up to 3)
        neg_examples = confused_texts[:3]
        neg_text = "\n".join(f"  - {ex[:150]}" for ex in neg_examples)

        return [PromptModification(
            modification_type="inject_section",
            content=(
                f"\n# CLASS BOUNDARY - DO NOT GENERATE LIKE THESE\n"
                f"The following are examples from class \"{confused_class}\" — "
                f"your outputs must be DIFFERENT from these:\n{neg_text}\n\n"
                f"Make sure every generated text clearly belongs to \"{target_class}\" "
                f"and could NOT be mistaken for \"{confused_class}\".\n"
            )
        )]

    def _handle_generic_collapse(
        self, current_temperature: float
    ) -> List[PromptModification]:
        """Accepted pool lacks diversity -> increase temperature + diversity instruction."""
        new_temp = min(1.2, current_temperature + 0.1)
        return [PromptModification(
            modification_type="inject_section",
            content=(
                "\n# DIVERSITY REQUIREMENT (CRITICAL)\n"
                "Your previous outputs were too similar to each other. "
                "Ensure MAXIMUM diversity: vary topics, sentence structures, "
                "perspectives, and vocabulary across all generated texts. "
                "Each text should feel distinctly different from the others.\n"
            ),
            parameter_changes={"temperature": new_temp}
        )]

    def _build_improved_prompt(
        self,
        base_prompt: str,
        modifications: List[PromptModification]
    ) -> str:
        """Apply modifications to the base prompt."""
        if not modifications:
            return base_prompt

        # Collect all injection sections
        injections = [
            m.content for m in modifications
            if m.modification_type == "inject_section"
        ]

        if not injections:
            return base_prompt

        injection_block = "\n".join(injections)

        # Inject before the OUTPUT FORMAT section if it exists
        if "# OUTPUT FORMAT" in base_prompt:
            parts = base_prompt.split("# OUTPUT FORMAT", 1)
            return parts[0] + injection_block + "\n# OUTPUT FORMAT" + parts[1]
        elif "# CONSTRAINTS" in base_prompt:
            parts = base_prompt.split("# CONSTRAINTS", 1)
            return parts[0] + injection_block + "\n# CONSTRAINTS" + parts[1]
        else:
            # Append before the last line
            lines = base_prompt.rstrip().rsplit("\n", 1)
            if len(lines) == 2:
                return lines[0] + "\n" + injection_block + "\n" + lines[1]
            return base_prompt + "\n" + injection_block
