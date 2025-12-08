#!/usr/bin/env python3
"""
Hybrid SMOTE + LLM augmentation pipeline for the MBTI dataset.
Phase 2: Ensemble Anchors + Contamination-Aware Filtering + Enhanced Quality Gates

Key Features:
- Ensemble anchor selection (medoid + quality-gated + diverse)
- Contamination-aware filtering (dynamic thresholds)
- Enhanced quality gates (probabilistic decisions, purity & F1 scaling)
- Purity-based budget reduction
- F1-based budget scaling

Author: Benja
Date: 2025-10-30
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai is optional until generation
    OpenAI = None

# Import anchor quality improvements module
try:
    from anchor_quality_improvements import (
        check_anchor_quality,
        select_best_anchors,
        get_adaptive_filter_params,
    )
    ANCHOR_QUALITY_AVAILABLE = True
except ImportError:
    ANCHOR_QUALITY_AVAILABLE = False
    import warnings
    warnings.warn("anchor_quality_improvements module not found. Phase 1-2 will be disabled.")

# Import quality gate predictor and class descriptions
try:
    from quality_gate_predictor import QualityGatePredictor, ClassMetrics
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False
    import warnings
    warnings.warn("quality_gate_predictor module not found. Quality gate prediction will be disabled.")

try:
    from mbti_class_descriptions import enhance_prompt_with_description
    CLASS_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    CLASS_DESCRIPTIONS_AVAILABLE = False
    import warnings
    warnings.warn("mbti_class_descriptions module not found. Class description enhancement will be disabled.")

# ============================================================================
# Phase 2: Import ensemble modules
# ============================================================================

try:
    from ensemble_anchor_selector import EnsembleAnchorSelector
    ENSEMBLE_ANCHOR_AVAILABLE = True
except ImportError:
    ENSEMBLE_ANCHOR_AVAILABLE = False
    import warnings
    warnings.warn("ensemble_anchor_selector module not found. Falling back to Phase 1 medoid selection.")

try:
    from contamination_aware_filter import ContaminationAwareFilter
    CONTAMINATION_FILTER_AVAILABLE = True
except ImportError:
    CONTAMINATION_FILTER_AVAILABLE = False
    import warnings
    warnings.warn("contamination_aware_filter module not found. Using Phase 1 fixed thresholds.")

try:
    from enhanced_quality_gate import EnhancedQualityGate
    ENHANCED_GATE_AVAILABLE = True
except ImportError:
    ENHANCED_GATE_AVAILABLE = False
    import warnings
    warnings.warn("enhanced_quality_gate module not found. Using Phase 1 quality gate.")

# ============================================================================
# TIER S Improvements
# ============================================================================

try:
    from adversarial_discriminator import AdversarialDiscriminator
    ADVERSARIAL_DISCRIMINATOR_AVAILABLE = True
except ImportError:
    ADVERSARIAL_DISCRIMINATOR_AVAILABLE = False
    import warnings
    warnings.warn("adversarial_discriminator module not found. Discriminator filtering will be disabled.")

try:
    from multi_seed_ensemble import MultiSeedEnsemble
    MULTI_SEED_ENSEMBLE_AVAILABLE = True
except ImportError:
    MULTI_SEED_ENSEMBLE_AVAILABLE = False
    import warnings
    warnings.warn("multi_seed_ensemble module not found. Ensemble mode will be disabled.")


# ============================================================================
# Phase 2: Enhanced Dynamic Budget Calculator
# ============================================================================

def calculate_enhanced_budget(
    n_samples: int,
    quality_score: float,
    purity: float,
    baseline_f1: float,
    purity_low_threshold: float = 0.30,
    purity_low_multiplier: float = 0.3,
    f1_high_threshold: float = 0.45,
    f1_high_multiplier: float = 0.5,
    f1_low_threshold: float = 0.15,
    f1_low_multiplier: float = 1.5,
    target_ratio: float = 0.08
) -> Tuple[int, str, Dict[str, float]]:
    """
    Phase 2 enhanced budget calculator with purity and F1 multipliers.

    budget = base_budget × quality_mult × purity_mult × f1_mult

    Args:
        n_samples: Number of real samples in class
        quality_score: Anchor quality score (0-1)
        purity: Anchor purity score (0-1)
        baseline_f1: Baseline F1 before augmentation
        purity_low_threshold: Purity below which to apply reduction
        purity_low_multiplier: Multiplier for low purity
        f1_high_threshold: F1 above which to reduce budget
        f1_high_multiplier: Multiplier for high F1
        f1_low_threshold: F1 below which to increase budget
        f1_low_multiplier: Multiplier for low F1
        target_ratio: Target synthetic/real ratio

    Returns:
        budget: Number of synthetics to generate
        reason: Explanation of budget calculation
        multipliers: Dict with individual multipliers
    """
    # Base budget: target_ratio% of real samples (default 8%)
    base_budget = int(n_samples * target_ratio)

    # 1. Quality multiplier (Phase 1 logic)
    if quality_score < 0.35:
        quality_mult = 0.1
        quality_reason = f"🔴 Very low quality ({quality_score:.3f})"
    elif quality_score < 0.40:
        quality_mult = 0.3
        quality_reason = f"⚠️  Low quality ({quality_score:.3f})"
    elif quality_score < 0.50:
        quality_mult = 0.7
        quality_reason = f"⚠️  Mediocre quality ({quality_score:.3f})"
    else:
        quality_mult = 1.0
        quality_reason = f"✅ Good quality ({quality_score:.3f})"

    # 2. Purity multiplier (Phase 2 NEW)
    if purity < purity_low_threshold:
        purity_mult = purity_low_multiplier
        purity_reason = f"🔴 Purity disaster ({purity:.3f}) → {int(purity_mult*100)}% reduction"
    else:
        purity_mult = 1.0
        purity_reason = f"✅ Purity OK ({purity:.3f})"

    # 3. F1 multiplier (Phase 2 NEW)
    if baseline_f1 > f1_high_threshold:
        f1_mult = f1_high_multiplier
        f1_reason = f"🔵 High F1 ({baseline_f1:.3f}) → {int(f1_mult*100)}% budget (less needed)"
    elif baseline_f1 < f1_low_threshold:
        f1_mult = f1_low_multiplier
        f1_reason = f"🟢 Low F1 ({baseline_f1:.3f}) → {int(f1_mult*100)}% budget (more help)"
    else:
        f1_mult = 1.0
        f1_reason = f"✅ Normal F1 ({baseline_f1:.3f})"

    # Combine multipliers
    total_mult = quality_mult * purity_mult * f1_mult
    budget = max(10, int(base_budget * total_mult))

    # Build reason string
    reason = f"Base: {base_budget} ({int(target_ratio*100)}% of {n_samples})\n"
    reason += f"   {quality_reason} × {int(quality_mult*100)}%\n"
    reason += f"   {purity_reason} × {int(purity_mult*100)}%\n"
    reason += f"   {f1_reason} × {int(f1_mult*100)}%\n"
    reason += f"   → Final: {budget} synthetics (×{total_mult:.2f})"

    multipliers = {
        "quality": quality_mult,
        "purity": purity_mult,
        "f1": f1_mult,
        "total": total_mult
    }

    return budget, reason, multipliers


# Backward compatibility: Phase 1 budget calculator
def calculate_dynamic_budget(n_samples: int, quality_score: float) -> Tuple[int, str]:
    """
    Phase 1 budget calculator (backward compatibility).
    Calls Phase 2 enhanced version with default purity and F1.
    """
    budget, reason, _ = calculate_enhanced_budget(
        n_samples=n_samples,
        quality_score=quality_score,
        purity=0.50,  # Neutral default
        baseline_f1=0.35,  # Neutral default
    )
    return budget, reason.split('\n')[0]  # Return simple reason


# ============================================================================
# V3 Phase 1: Mejora #1 - Filtros Adaptativos por Clase
# ============================================================================

def get_adaptive_thresholds(class_name: str, baseline_f1: float) -> Dict[str, float]:
    """
    Ajusta thresholds de filtros según F1 baseline de la clase.
    Clases débiles: filtros más relajados → más sintéticos
    Clases fuertes: filtros más estrictos → mejor calidad
    """
    if baseline_f1 < 0.10:  # Clases muy débiles
        return {
            'knn_sim': 0.37,      # -0.05 del base K-Fold
            'threshold': 0.40,    # -0.05
            'to_anchor': 0.40,    # -0.05
        }
    elif baseline_f1 < 0.20:  # Clases débiles
        return {
            'knn_sim': 0.40,      # -0.02
            'threshold': 0.43,    # -0.02
            'to_anchor': 0.43,    # -0.02
        }
    elif baseline_f1 < 0.40:  # Clases medias
        return {
            'knn_sim': 0.42,      # base (K-Fold best)
            'threshold': 0.45,    # base
            'to_anchor': 0.45,    # base
        }
    else:  # Clases fuertes (F1 >= 0.40)
        return {
            'knn_sim': 0.44,      # +0.02 más estricto
            'threshold': 0.47,    # +0.02
            'to_anchor': 0.47,    # +0.02
        }


# ============================================================================
# V3 Phase 1: Mejora #2 - Generación 2× para Clases Muy Débiles
# ============================================================================

def get_prompts_multiplier(baseline_f1: float) -> float:
    """
    Clases muy débiles reciben 2× prompts por cluster.
    Clases débiles reciben 1.5× prompts por cluster.
    Resto: normal (1×).
    """
    if baseline_f1 < 0.10:
        return 2.0  # Doble generación para clases muy débiles
    elif baseline_f1 < 0.20:
        return 1.5  # 50% más para clases débiles
    else:
        return 1.0  # Normal


# ============================================================================
# V3 Phase 1: Mejora #3 - K-NN Ponderado por Distancia
# ============================================================================

def get_weighted_knn_support(
    synthetic_emb: np.ndarray,
    cluster_matrix: np.ndarray,
    cluster_texts: List[str],
    k: int = 8
) -> List[str]:
    """
    Vecinos más cercanos aparecen más veces en el prompt.
    Esto da señal implícita al LLM de su importancia relativa.

    Top 3 vecinos aparecen 2-3 veces, resto 1 vez.
    Total: ~10 ejemplos con ponderación implícita.
    """
    distances = cosine_distances(synthetic_emb.reshape(1, -1), cluster_matrix)[0]
    k_actual = min(k, len(cluster_texts))
    k_indices = np.argsort(distances)[:k_actual]
    k_distances = distances[k_indices]

    # Convertir distancias a pesos (inverso)
    weights = 1 / (k_distances + 0.01)  # +0.01 evita división por 0
    weights = weights / weights.sum()  # Normalizar a [0, 1]

    # Repetir ejemplos según peso
    # Top vecinos tienen peso alto → aparecen más veces
    support_examples = []
    for idx, weight in zip(k_indices, weights):
        # Mapear peso [0, 1] a repeticiones [1, 3]
        repeats = max(1, min(3, int(weight * k_actual * 2)))
        support_examples.extend([cluster_texts[idx]] * repeats)

    # Limitar a ~10-12 ejemplos total para no sobrecargar prompt
    return support_examples[:12]


def get_adaptive_prompt_mode(
    class_name: str,
    baseline_f1: float,
    default_mode: str = "mix"
) -> str:
    """
    EXPERIMENT 3: Adaptive prompt-mode selection based on baseline F1.

    Select prompt mode based on baseline F1 score:
    - Classes with F1 >= 45%: Use 'paraphrase' (quality preservation, avoid contamination)
    - Classes with F1 < 20%: Use 'mix' (maximum diversity)
    - Classes with 20% <= F1 < 45%: Use 'mix' (conservative)

    Research evidence (ACL 2024): LLM paraphrasing achieves 100% validity vs 95-97% for mix.

    Args:
        class_name: Name of the class
        baseline_f1: Baseline F1 score for this class (0-1 scale)
        default_mode: Mode to use if baseline_f1 is not available

    Returns:
        'paraphrase' or 'mix'
    """
    if baseline_f1 is None or baseline_f1 < 0:
        return default_mode

    # High F1 (>= 45%): Use paraphrase to preserve quality and avoid contamination
    if baseline_f1 >= 0.45:
        return "paraphrase"

    # Low F1 (< 20%): Use mix for maximum diversity
    if baseline_f1 < 0.20:
        return "mix"

    # Medium F1 (20-45%): Use mix as conservative choice
    return "mix"


def get_adaptive_synthetic_weight(
    class_name: str,
    baseline_f1: float,
    default_weight: float = 0.5
) -> float:
    """
    EXPERIMENT 1 (PHASE B): Class-specific adaptive synthetic weighting based on baseline F1.

    The goal is to assign different synthetic weights to different classes based on their
    baseline performance to maximize benefit and minimize cross-contamination.

    Weight strategy:
    - F1 < 15%: weight 0.5 (HIGH) - Very low F1 classes can handle high synthetic weight
    - F1 15-30%: weight 0.3 (MEDIUM) - Medium-low classes need moderate weight
    - F1 30-45%: weight 0.1 (LOW) - CRITICAL: Vulnerable zone needs low weight to avoid contamination
    - F1 >= 45%: weight 0.05 (VERY LOW) - High F1 classes need minimal weight to avoid degradation

    Research evidence (Nature 2025): Adaptive weighting on minority classes outperforms fixed weights.
    Batch 5A findings: MID F1 (20-45%) degraded with fixed weight 0.5 → needs reduction.

    Args:
        class_name: Name of the class
        baseline_f1: Baseline F1 score for this class (0-1 scale)
        default_weight: Default weight to use if baseline_f1 is not available

    Returns:
        Synthetic weight for this class (float)
    """
    if baseline_f1 is None or baseline_f1 < 0:
        return default_weight

    # VERY LOW F1 (< 15%): High weight - can handle heavy augmentation
    if baseline_f1 < 0.15:
        return 0.5

    # LOW F1 (15-30%): Medium weight - moderate augmentation
    if baseline_f1 < 0.30:
        return 0.3

    # MID F1 (30-45%): Low weight - VULNERABLE ZONE, minimize contamination
    if baseline_f1 < 0.45:
        return 0.1

    # HIGH F1 (>= 45%): Very low weight - avoid degradation
    return 0.05


@dataclass
class PromptSpec:
    target_class: str
    cluster_id: int
    anchor_text: str
    neighbor_text: str
    support_examples: List[str]
    top_keywords: List[str]
    cluster_centroid: np.ndarray
    cluster_texts: List[str]
    similarity_floor: float
    n_samples: int
    max_prompt_tokens: int
    prompt_mode: str  # 'mix' or 'paraphrase'
    language: str  # 'auto', 'en', 'es'
    cluster_size: int
    use_class_description: bool = False  # Add MBTI description to prompt

    def build_prompt(self) -> str:
        def clip(text: str) -> str:
            tokens = text.split()
            if len(tokens) <= self.max_prompt_tokens:
                return text
            return " ".join(tokens[: self.max_prompt_tokens])

        guidance = ", ".join(self.top_keywords[:10]) if self.top_keywords else ""
        examples = "\n".join(
            f"Ejemplo {i + 1}: {clip(text)}" for i, text in enumerate(self.support_examples)
        )
        anchor_clip = clip(self.anchor_text)
        neighbor_clip = clip(self.neighbor_text)

        lang_hint = ""
        if self.language == "en":
            lang_hint = " Write in English only."
        elif self.language == "es":
            lang_hint = " Escribe solo en español."

        if self.prompt_mode == "paraphrase":
            prompt = (
                "Genera {n} variantes reescritas del siguiente texto, manteniendo su significado y tono.\n"
                "Cada variante debe pertenecer claramente a la clase [{label}] por su estilo, pero sin incluir la etiqueta.\n"
                "Usa vocabulario del clúster y alguna(s) de estas palabras si es natural: {kw}.\n"
                "Evita URLs, hashtags o emojis repetitivos. Una sola línea por variante.{lang}\n\n"
                "Texto A (a reescribir): {anchor}\n\n"
            ).format(n=self.n_samples, label=self.target_class, kw=guidance, anchor=anchor_clip, lang=lang_hint)
        else:  # 'mix'
            prompt = (
                "Genera {n} textos nuevos en español o inglés si corresponde.\n"
                "Cada texto debe pertenecer claramente a la clase MBTI [{label}] por su tono/contenido,\n"
                "pero NO incluyas literalmente la etiqueta ni metadatos (solo el texto).\n"
                "Sigue el tono y vocabulario emocional de los ejemplos.\n"
                "Mantén la longitud similar (±20%) y no cambies la polaridad/estilo de la clase.{lang}".format(
                    n=self.n_samples, label=self.target_class, lang=lang_hint
                )
            )
            if guidance:
                prompt += f"\nPalabras frecuentes del subtema: {guidance}."
            prompt += (
                "\n\n{examples}\n\nTexto A (base): {anchor}\nTexto B (vecino): {neighbor}\n"
                "Fusiona elementos de A y B sin copiar literalmente y produce cada ejemplo en una sola línea."
                " Evita incluir URLs, hashtags o emojis repetitivos.".format(
                    examples=examples,
                    anchor=anchor_clip,
                    neighbor=neighbor_clip,
                )
            )
        prompt += "\nDevuelve exactamente {n} muestras distintas, cada una en una sola línea.".format(n=self.n_samples)

        # Enhance prompt with class description if enabled
        if self.use_class_description and CLASS_DESCRIPTIONS_AVAILABLE:
            prompt = enhance_prompt_with_description(
                base_prompt=prompt,
                class_name=self.target_class,
                use_description=True
            )

        return prompt


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Acepta dos formatos: (type, posts) original o (label, text) ya normalizado
    if {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].dropna()
        return df
    if {"type", "posts"}.issubset(df.columns):
        df = df[["type", "posts"]].dropna()
        df.rename(columns={"type": "label", "posts": "text"}, inplace=True)
        return df
    raise ValueError("Expected columns 'label,text' or 'type,posts' in the dataset")


def load_class_overrides(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = str(path)
    try:
        import json as _json
        with open(p, "r", encoding="utf-8") as f:
            if p.endswith(".json"):
                data = _json.load(f)
            else:
                try:
                    import yaml as _yaml  # type: ignore
                    data = _yaml.safe_load(f)
                except Exception as e:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "Para YAML instala PyYAML o usa JSON en --class-config"
                    ) from e
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar class-config {p}: {exc}")
    if not isinstance(data, dict):
        return {}
    # admit either top-level map or {"overrides": {...}}
    overrides = data.get("overrides", data)
    if not isinstance(overrides, dict):
        return {}
    return overrides  # {class_name: {param: value}}


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
BAR_RE = re.compile(r"\|\|\|+")
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    t = BAR_RE.sub(". ", str(text))
    t = URL_RE.sub("", t)
    t = t.replace("_", " ")
    t = WS_RE.sub(" ", t).strip()
    return t


def normalize_texts(texts: Sequence[str]) -> List[str]:
    return [normalize_text(t) for t in texts]


def compute_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )


def train_baseline_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    class_weight: Optional[object] = None,
) -> Pipeline:
    scaler = StandardScaler()
    # Note: multi_class="multinomial" is now default with solver="lbfgs", removed to avoid deprecation warning
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight)
    model = Pipeline([
        ("scaler", scaler),
        ("clf", clf),
    ])
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, clf__sample_weight=sample_weight)
    return model


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    tag: str,
) -> Dict[str, float]:
    preds = model.predict(X)
    f1_macro = f1_score(y, preds, average="macro")
    report = classification_report(
        y,
        preds,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    return {"tag": tag, "macro_f1": f1_macro, "report": report}


def select_minority_classes(
    labels: Sequence[str],
    explicit: Optional[Sequence[str]] = None,
    percentile: float = 0.5,
) -> List[str]:
    counts = Counter(labels)
    if explicit:
        missing = [c for c in explicit if c not in counts]
        if missing:
            raise ValueError(f"Classes not found in data: {missing}")
        return list(explicit)
    threshold = np.quantile(list(counts.values()), percentile)
    return [label for label, count in counts.items() if count <= threshold]


def remove_outliers_knn(
    embeddings: np.ndarray,
    k_neighbors: int = 5,
    keep_quantile: float = 0.85,
) -> np.ndarray:
    if len(embeddings) <= k_neighbors:
        return np.arange(len(embeddings))
    n_neighbors = min(k_neighbors + 1, len(embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    mean_dists = distances[:, 1:].mean(axis=1)
    threshold = np.quantile(mean_dists, keep_quantile)
    mask = mean_dists <= threshold
    return np.where(mask)[0]


def extract_top_keywords(texts: Sequence[str], top_k: int = 20) -> List[str]:
    freq: Dict[str, int] = defaultdict(int)
    for text in texts:
        tokens = [t.lower() for t in text.split() if t.isalpha() and len(t) > 3]
        for token in tokens:
            freq[token] += 1
    return [word for word, _ in Counter(freq).most_common(top_k)]


def assign_cluster_count(n_samples: int, cluster_size: int, max_clusters: int) -> int:
    estimate = max(1, math.ceil(n_samples / cluster_size))
    return min(max_clusters, estimate)


def elbow_method_optimal_clusters(embeddings: np.ndarray, min_k: int = 2, max_k: int = 15) -> int:
    """
    Método del codo para determinar número óptimo de clusters automáticamente.

    Calcula inercia (within-cluster sum of squares) para k=min_k hasta k=max_k,
    y encuentra el "codo" donde la mejora marginal disminuye significativamente.

    Args:
        embeddings: Embeddings de la clase
        min_k: Mínimo número de clusters a probar
        max_k: Máximo número de clusters a probar

    Returns:
        Número óptimo de clusters según método del codo
    """
    if len(embeddings) < min_k:
        return 1

    max_k = min(max_k, len(embeddings))
    if max_k <= min_k:
        return min_k

    inertias = []
    k_range = range(min_k, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # Calcular "ángulo" entre segmentos consecutivos
    # El codo está donde el ángulo cambia más abruptamente
    if len(inertias) < 3:
        return min_k

    # Normalizar inercias para calcular ángulos
    x = np.arange(len(inertias))
    y = np.array(inertias)

    # Calcular segunda derivada (tasa de cambio de la tasa de cambio)
    # El codo está donde la segunda derivada es máxima
    deltas = np.diff(y)
    second_deltas = np.diff(deltas)

    if len(second_deltas) == 0:
        return min_k

    # El codo es donde la segunda derivada (aceleración) es máxima
    elbow_idx = np.argmax(second_deltas) + 1  # +1 porque perdemos un elemento en diff
    optimal_k = min_k + elbow_idx

    # Validación: si el óptimo es muy pequeño o muy grande, usar heurística
    if optimal_k < 3:
        optimal_k = min(3, max_k)
    elif optimal_k > max_k - 2:
        # Si está al final del rango, probablemente necesitamos más
        optimal_k = max_k

    return optimal_k


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[KMeans, np.ndarray]:
    if n_clusters <= 1 or len(embeddings) <= n_clusters:
        labels = np.zeros(len(embeddings), dtype=int)
        model = KMeans(n_clusters=1, n_init="auto", random_state=random_state)
        model.fit(embeddings)
        return model, labels
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = model.fit_predict(embeddings)
    return model, cluster_labels


def find_low_density_indices(
    embeddings: np.ndarray,
    n_neighbors: int = 5,
    top_quantile: float = 0.3,
) -> np.ndarray:
    if len(embeddings) <= n_neighbors:
        return np.arange(len(embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    mean_dists = distances[:, 1:].mean(axis=1)
    threshold = np.quantile(mean_dists, 1 - top_quantile)
    mask = mean_dists >= threshold
    return np.where(mask)[0]


def ask_llm_for_anchor(
    texts: List[str],
    target_class: str,
    llm_model: str,
    temperature: float = 0.3,
) -> int:
    """
    Ask LLM to select the best anchor text from cluster.

    Args:
        texts: List of texts in cluster
        target_class: Target MBTI class
        llm_model: LLM model name
        temperature: Temperature for LLM call

    Returns:
        Index of selected anchor (0-based)
    """
    if len(texts) == 0:
        return 0

    # Limit to top 10 samples to avoid token limits
    sample_texts = texts[:min(10, len(texts))]

    # Build prompt
    samples_str = "\n".join([f"{i+1}. {text[:200]}" for i, text in enumerate(sample_texts)])

    prompt = f"""Given these text samples from a {target_class} personality type cluster, select the ONE sample that would be the best anchor/representative for generating similar texts.

Samples:
{samples_str}

Select the sample number (1-{len(sample_texts)}) that is:
- Most representative of the {target_class} personality type
- Clear, well-written, and authentic
- Not an outlier or edge case
- Good quality (no spam, URLs, or gibberish)
- Has meaningful emotional/cognitive content typical of {target_class}

Respond with ONLY the number (1-{len(sample_texts)}), nothing else."""

    try:
        from openai import OpenAI
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to medoid if no API key
            return 0

        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in MBTI personality types and text analysis. Respond with only the number."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=10,
        )

        # Validate API response
        if not completion.choices:
            raise RuntimeError("OpenAI API returned no choices")
        response = completion.choices[0].message.content.strip()

        # Parse response - extract first number found
        import re
        match = re.search(r'\d+', response)
        if match:
            selected = int(match.group())
            # Convert to 0-based index and validate
            if 1 <= selected <= len(sample_texts):
                idx = selected - 1
                # If we sampled, find this text in original list
                if len(texts) > len(sample_texts):
                    selected_text = sample_texts[idx]
                    # Find in original
                    for i, text in enumerate(texts):
                        if text == selected_text:
                            return i
                return idx

        # Fallback to first sample if parsing failed
        return 0

    except Exception as e:
        # Fallback to medoid on any error
        import warnings
        warnings.warn(f"LLM anchor selection failed: {e}. Falling back to medoid.")
        return 0


def select_anchor_indices(
    embeddings: np.ndarray,
    texts: List[str],
    method: str,
    n_anchors: int = 1,
    llm_model: Optional[str] = None,
    llm_temperature: float = 0.3,
    target_class: Optional[str] = None,
) -> List[int]:
    """
    Select anchor indices from cluster using specified method.

    Args:
        embeddings: Cluster embeddings (N, D)
        texts: Cluster texts (N,)
        method: One of 'centroid', 'medoid', 'quality_gated', 'diverse', 'ensemble'
        n_anchors: Number of anchors to return (for ensemble method)

    Returns:
        List of anchor indices to use for SMOTE generation
    """
    if len(embeddings) == 0:
        return []

    if method == "centroid":
        # CENTROID: Use index closest to mathematical centroid
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        distances = cosine_similarity(embeddings, centroid).reshape(-1)
        best_idx = int(distances.argmax())
        return [best_idx]

    elif method == "medoid":
        # MEDOID: Find most central point (min average distance to others)
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        distances = cosine_similarity(embeddings, centroid).reshape(-1)
        best_idx = int(distances.argmax())
        return [best_idx]

    elif method == "quality_gated":
        # QUALITY_GATED: Use medoid but validate text quality
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid).reshape(-1)

        # Check if centroid is good quality (high avg similarity)
        if similarities.mean() < 0.5:
            # Poor quality centroid - fallback to medoid with quality check
            sorted_indices = similarities.argsort()[::-1]
            for idx in sorted_indices:
                text = texts[int(idx)]
                # Validate text quality
                if (len(text) >= 20 and
                    len(text.split()) >= 5 and
                    not text.strip().startswith(('http', '#', '@'))):
                    return [int(idx)]
            # Fallback to best if all fail quality check
            if len(sorted_indices) == 0:
                return []
            return [int(sorted_indices[0])]
        else:
            # Good centroid - use medoid
            best_idx = int(similarities.argmax())
            return [best_idx]

    elif method == "diverse":
        # DIVERSE: Select point in middle 50th percentile that's most diverse
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        distances = 1 - cosine_similarity(embeddings, centroid).reshape(-1)

        # Get middle 50th percentile (25-75)
        p25 = np.percentile(distances, 25)
        p75 = np.percentile(distances, 75)
        mid_range_mask = (distances >= p25) & (distances <= p75)
        mid_range_indices = np.where(mid_range_mask)[0]

        if len(mid_range_indices) == 0:
            # Fallback to medoid
            best_idx = int((1 - distances).argmax())
            return [best_idx]

        # Among mid-range, find most diverse (max avg distance to others)
        max_diversity = -1
        best_idx = mid_range_indices[0]
        for idx in mid_range_indices:
            # Diversity = average distance to all other points
            point = embeddings[idx].reshape(1, -1)
            diversity = (1 - cosine_similarity(embeddings, point).reshape(-1)).mean()
            if diversity > max_diversity:
                max_diversity = diversity
                best_idx = idx

        return [int(best_idx)]

    elif method == "ensemble":
        # ENSEMBLE: Return top-N most central points (top-N medoids)
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid).reshape(-1)
        top_n_indices = similarities.argsort()[::-1][:min(n_anchors, len(embeddings))]
        return [int(idx) for idx in top_n_indices]

    elif method == "llm_recommender":
        # LLM_RECOMMENDER: Ask LLM to select best anchor based on semantic analysis
        if llm_model is None or target_class is None:
            # Fallback to medoid if LLM params not provided
            centroid = embeddings.mean(axis=0).reshape(1, -1)
            distances = cosine_similarity(embeddings, centroid).reshape(-1)
            best_idx = int(distances.argmax())
            return [best_idx]

        selected_idx = ask_llm_for_anchor(
            texts=texts,
            target_class=target_class,
            llm_model=llm_model,
            temperature=llm_temperature,
        )
        return [selected_idx]

    else:
        # Default fallback to centroid method
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        distances = cosine_similarity(embeddings, centroid).reshape(-1)
        best_idx = int(distances.argmax())
        return [best_idx]


def interpolate_embedding_pairs(
    embeddings: np.ndarray,
    n_pairs: int,
    rng: random.Random,
    k_neighbors: int = 5,
    allowed_anchor_indices: Optional[List[int]] = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Generate SMOTE interpolation pairs.

    Args:
        embeddings: Cluster embeddings
        n_pairs: Number of pairs to generate
        rng: Random number generator
        k_neighbors: Number of neighbors for KNN
        allowed_anchor_indices: If provided, only use these indices as anchors

    Returns:
        List of (anchor_idx, neighbor_idx, synthetic_embedding) tuples
    """
    if len(embeddings) < 2:
        return []
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(embeddings)))
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    pairs: List[Tuple[int, int, np.ndarray]] = []

    # Determine which indices can be used as anchors
    if allowed_anchor_indices is not None and len(allowed_anchor_indices) > 0:
        # Use only specified anchors, cycling through them
        anchor_pool = allowed_anchor_indices
    else:
        # Default: use all indices (original behavior)
        anchor_pool = list(range(len(embeddings)))

    for i in range(n_pairs):
        # Select anchor from allowed pool
        if len(anchor_pool) == 1:
            anchor = anchor_pool[0]
        else:
            # For ensemble method, cycle through anchors deterministically
            if allowed_anchor_indices is not None:
                anchor = anchor_pool[i % len(anchor_pool)]
            else:
                # Original random behavior when no restriction
                anchor = rng.choice(anchor_pool)

        neighbors = indices[anchor][1:]
        if len(neighbors) == 0:
            continue
        neighbour = int(rng.choice(list(neighbors)))
        lam = rng.random()
        synthetic = embeddings[anchor] + lam * (embeddings[neighbour] - embeddings[anchor])
        pairs.append((anchor, neighbour, synthetic))
    return pairs


def build_prompt_specs(
    class_name: str,
    class_texts: Sequence[str],
    class_embeddings: np.ndarray,
    rng: random.Random,
    args: argparse.Namespace,
    baseline_f1_scores: Optional[Dict[str, float]] = None,
) -> List[PromptSpec]:
    indices = remove_outliers_knn(class_embeddings, args.outlier_k, args.outlier_keep_quantile)
    clean_embeddings = class_embeddings[indices]
    clean_texts = [class_texts[i] for i in indices]

    # Usa método del codo para determinar óptimo automáticamente con mínimo adaptativo
    optimal_k = elbow_method_optimal_clusters(clean_embeddings, min_k=2, max_k=min(15, len(clean_embeddings) // 2))
    # Mínimo adaptativo: escala con tamaño de clase (AGRESIVO)
    min_clusters = min(12, max(6, len(clean_embeddings) // 60))
    n_clusters = max(min_clusters, optimal_k)
    print(f"  {class_name}: elbow_k={optimal_k}, min_adaptativo={min_clusters}, usado={n_clusters} clusters (n_samples={len(clean_embeddings)})")
    # Usa gen_seed para clustering cuando esté disponible
    random_state = args.gen_seed if getattr(args, "gen_seed", None) is not None else args.random_seed
    _, cluster_labels = cluster_embeddings(clean_embeddings, n_clusters, random_state)

    prompt_specs: List[PromptSpec] = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_matrix = clean_embeddings[mask]
        cluster_texts = [text for idx, text in enumerate(clean_texts) if mask[idx]]
        if len(cluster_texts) < 3:
            continue
        low_density_idx = find_low_density_indices(
            cluster_matrix, args.density_k, args.density_top_quantile
        )
        if len(low_density_idx) == 0:
            continue
        # V3 Phase 1 - Mejora #2: Generación 2× para clases muy débiles
        base_prompts = args.prompts_per_cluster
        if baseline_f1_scores and class_name in baseline_f1_scores:
            multiplier = get_prompts_multiplier(baseline_f1_scores[class_name])
            n_prompts = int(base_prompts * multiplier)
        else:
            n_prompts = base_prompts

        # Select anchor indices using specified method
        anchor_method = getattr(args, 'anchor_selection_method', 'centroid')
        n_anchors_for_ensemble = 3 if anchor_method == 'ensemble' else 1
        selected_anchor_indices = select_anchor_indices(
            embeddings=cluster_matrix,
            texts=cluster_texts,
            method=anchor_method,
            n_anchors=n_anchors_for_ensemble,
            llm_model=args.llm_model if anchor_method == 'llm_recommender' else None,
            llm_temperature=0.3,  # Low temperature for more consistent anchor selection
            target_class=class_name,
        )

        pairs = interpolate_embedding_pairs(
            cluster_matrix,
            n_pairs=min(len(low_density_idx), n_prompts),
            rng=rng,
            k_neighbors=args.smote_k,
            allowed_anchor_indices=selected_anchor_indices if anchor_method != 'centroid' else None,
        )
        if not pairs:
            continue
        keywords = extract_top_keywords(cluster_texts, args.top_keywords)
        centroid = cluster_matrix.mean(axis=0)
        # Dynamic similarity floor based on intra-cluster similarities
        sims = cosine_similarity(cluster_matrix, centroid.reshape(1, -1)).reshape(-1)
        sim_floor = float(np.quantile(sims, args.similarity_floor_quantile))
        for anchor_idx, neighbour_idx, synthetic_emb in pairs:
            anchor_text = cluster_texts[anchor_idx]
            neighbor_text = cluster_texts[neighbour_idx]
            # V3 Phase 1 - Mejora #3: K-NN ponderado por distancia
            if args.knn_support > 0:
                k_support = min(args.knn_support, len(cluster_texts))
            else:
                k_support = min(10, max(5, len(cluster_texts) // 5))  # Adaptativo con mínimo de 5
            support_examples = get_weighted_knn_support(synthetic_emb, cluster_matrix, cluster_texts, k=k_support)
            # EXPERIMENT 3: Adaptive prompt mode based on baseline F1
            adaptive_prompt_mode = get_adaptive_prompt_mode(
                class_name=class_name,
                baseline_f1=baseline_f1_scores.get(class_name) if baseline_f1_scores else None,
                default_mode=args.prompt_mode
            )

            prompt_specs.append(
                PromptSpec(
                    target_class=class_name,
                    cluster_id=cluster_id,
                    anchor_text=anchor_text,
                    neighbor_text=neighbor_text,
                    support_examples=support_examples,
                    top_keywords=keywords,
                    cluster_centroid=centroid,
                    cluster_texts=cluster_texts,
                    similarity_floor=sim_floor,
                    n_samples=args.samples_per_prompt,
                    max_prompt_tokens=args.max_prompt_tokens,
                    prompt_mode=adaptive_prompt_mode,
                    language=args.language,
                    cluster_size=len(cluster_texts),
                    use_class_description=getattr(args, 'use_class_description', False),
                )
            )
    return prompt_specs


def batched(iterable: Sequence[PromptSpec], batch_size: int) -> Iterable[List[PromptSpec]]:
    batch: List[PromptSpec] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_openai_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("openai package not installed. Install requirements first.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Carga .env o exporta la variable antes de ejecutar."
        )
    return OpenAI(api_key=api_key)


def call_llm_batch(
    client: OpenAI,
    batch: Sequence[PromptSpec],
    model_name: str,
    temperature: float,
    top_p: float,
    max_retries: int,
) -> Dict[int, List[str]]:
    outputs: Dict[int, List[str]] = {}
    for spec in batch:
        prompt = spec.build_prompt()
        last_error: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un asistente de generación de texto etiquetado. "
                                "Responde con exactamente {n} muestras, cada una en una línea.".format(
                                    n=spec.n_samples
                                )
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                )
                # Validate API response
                if not completion.choices:
                    raise RuntimeError("OpenAI API returned no choices")
                raw = completion.choices[0].message.content
                candidates = [line.strip() for line in raw.splitlines() if line.strip()]
                outputs[id(spec)] = candidates[: spec.n_samples]
                break
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
        else:  # pragma: no cover
            raise RuntimeError(f"LLM generation failed after {max_retries} attempts: {last_error}")
    return outputs


def embed_candidates(
    model: SentenceTransformer,
    generated: List[str],
    batch_size: int,
) -> np.ndarray:
    if not generated:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return compute_embeddings(model, generated, batch_size)


def jaccard_similarity(a: str, b: str, ngram: int = 3) -> float:
    def shingles(text: str) -> set:
        tokens = text.lower().split()
        if len(tokens) < ngram:
            return set(tokens)
        return {" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)}

    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def filter_candidates(
    candidates: List[str],
    candidate_embeddings: np.ndarray,
    target_class: str,
    cluster_centroid: np.ndarray,
    similarity_floor: float,
    model: Pipeline,
    label_encoder: LabelEncoder,
    args: argparse.Namespace,
    cluster_texts: Sequence[str],
    global_texts: Sequence[str],
    class_conf_thresholds: Dict[str, float],
    anchor_emb: Optional[np.ndarray] = None,
    neighbor_emb: Optional[np.ndarray] = None,
    nn_model: Optional[NearestNeighbors] = None,
    debug: bool = False,
    baseline_f1_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[np.ndarray], List[float]]:
    accepted_texts: List[str] = []
    accepted_embs: List[np.ndarray] = []
    accepted_confs: List[float] = []
    # Safely get class index
    matches = np.where(label_encoder.classes_ == target_class)[0]
    if len(matches) == 0:
        raise ValueError(f"Target class '{target_class}' not found in label_encoder.classes_")
    class_index = int(matches[0])
    centroid = cluster_centroid.reshape(1, -1)
    reference_texts = list(cluster_texts) + list(global_texts)

    # V3 Phase 1 - Mejora #1: Filtros adaptativos por clase
    if baseline_f1_scores and target_class in baseline_f1_scores:
        adaptive_th = get_adaptive_thresholds(target_class, baseline_f1_scores[target_class])
        sim_threshold = max(similarity_floor, adaptive_th['threshold'] - args.similarity_margin)
        similarity_to_anchor = adaptive_th['to_anchor']
    else:
        sim_threshold = max(similarity_floor, args.similarity_threshold - args.similarity_margin)
        similarity_to_anchor = args.similarity_to_anchor

    class_floor = class_conf_thresholds.get(target_class, args.min_classifier_confidence)
    conf_threshold = min(args.min_classifier_confidence, class_floor)
    rejected = Counter()

    if debug:
        print(
            f"Umbrales {target_class}: sim_floor={similarity_floor:.2f}, sim_th={sim_threshold:.2f}, "
            f"to_anchor={args.similarity_to_anchor:.2f}, conf_min={conf_threshold:.2f}"
        )

    # Optional semantic dedup pools and centroids passed via args (set by caller)
    ref_embs_class = getattr(args, "_ref_embs_class", None)
    ref_embs_generated = getattr(args, "_ref_embs_generated", None)
    centroids_by_class = getattr(args, "_centroids_by_class", None)

    for text, emb in zip(candidates, candidate_embeddings):
        token_count = len(text.split())
        if token_count < args.min_tokens or token_count > args.max_tokens:
            if debug:
                rejected["length"] += 1
            continue
        sim = float(cosine_similarity(emb.reshape(1, -1), centroid)[0][0])
        # allow acceptance if close to anchor/neighbor examples
        if anchor_emb is not None:
            sim_anchor = float(cosine_similarity(emb.reshape(1, -1), anchor_emb.reshape(1, -1))[0][0])
        else:
            sim_anchor = -1.0
        if neighbor_emb is not None:
            sim_neighbor = float(cosine_similarity(emb.reshape(1, -1), neighbor_emb.reshape(1, -1))[0][0])
        else:
            sim_neighbor = -1.0

        pass_sim = sim >= sim_threshold or max(sim_anchor, sim_neighbor) >= similarity_to_anchor
        if not pass_sim:
            if debug:
                rejected["similarity"] += 1
            continue
        # Repulsion against non-target centroids
        if centroids_by_class is not None and getattr(args, "repel_nontarget_sim", 0.0) > 0:
            max_other = -1.0
            for cls, cvec in centroids_by_class.items():
                if cls == target_class:
                    continue
                s = float(cosine_similarity(emb.reshape(1, -1), cvec.reshape(1, -1))[0][0])
                if s > max_other:
                    max_other = s
            if max_other >= args.repel_nontarget_sim:
                if debug:
                    rejected["repel"] += 1
                continue
        # Semantic dedup via embedding cosine similarity to references
        if getattr(args, "dedup_embed_sim", 0.0) > 0:
            near_dup = False
            if ref_embs_class is not None and len(ref_embs_class) > 0:
                try:
                    nn_c = NearestNeighbors(n_neighbors=1, metric="cosine")
                    nn_c.fit(ref_embs_class)
                    d, _ = nn_c.kneighbors(emb.reshape(1, -1), return_distance=True)
                    sim_c = float(1.0 - d.reshape(-1)[0])
                    if sim_c >= args.dedup_embed_sim:
                        near_dup = True
                except Exception:
                    pass
            if not near_dup and ref_embs_generated is not None and len(ref_embs_generated) > 0:
                try:
                    nn_g = NearestNeighbors(n_neighbors=1, metric="cosine")
                    nn_g.fit(ref_embs_generated)
                    d, _ = nn_g.kneighbors(emb.reshape(1, -1), return_distance=True)
                    sim_g = float(1.0 - d.reshape(-1)[0])
                    if sim_g >= args.dedup_embed_sim:
                        near_dup = True
                except Exception:
                    pass
            if near_dup:
                if debug:
                    rejected["semdup"] += 1
                continue
        # Classifier condition
        clf_ok = False
        if args.filter_mode in ("classifier", "hybrid"):
            proba_vec = model.predict_proba(emb.reshape(1, -1))[0]
            proba = float(proba_vec[class_index])
            pred_idx = int(np.argmax(proba_vec))
            # Validate prediction index
            if pred_idx >= len(label_encoder.classes_):
                raise ValueError(f"Invalid prediction index: {pred_idx}")
            pred_class = label_encoder.classes_[pred_idx]
            # Margin condition (difference between top-1 and top-2)
            order = np.argsort(proba_vec)[::-1]
            # Validate order has elements
            if len(order) == 0:
                best = 0.0
                second = 0.0
            else:
                best = float(proba_vec[order[0]])
                second = float(proba_vec[order[1]]) if len(order) > 1 else 0.0
            margin_ok = (best - second) >= getattr(args, "clf_margin", 0.0) if pred_class == target_class else False
            # For very high confidence, relax similarity slightly
            effective_sim_threshold = sim_threshold - (
                args.high_conf_sim_bonus if proba >= args.high_conf_threshold else 0.0
            )
            if sim < effective_sim_threshold and max(sim_anchor, sim_neighbor) < similarity_to_anchor:
                if debug:
                    rejected["similarity"] += 1
                # don't return yet; allow knn branch to save
            else:
                if pred_class == target_class and proba >= conf_threshold and margin_ok:
                    clf_ok = True

        # kNN condition within target class
        knn_ok = False
        if args.filter_mode in ("knn", "hybrid") and nn_model is not None:
            try:
                n_dist, _ = nn_model.kneighbors(emb.reshape(1, -1), return_distance=True)
                # cosine distance -> similarity
                sims_knn = 1.0 - n_dist.reshape(-1)
                mean_sim_knn = float(sims_knn.mean())
                if mean_sim_knn >= args.filter_knn_min_sim:
                    knn_ok = True
                else:
                    if debug:
                        rejected["knn"] += 1
            except Exception:
                pass

        if not (clf_ok or knn_ok):
            if debug and not clf_ok and args.filter_mode in ("classifier", "hybrid"):
                rejected["classifier"] += 1
            continue
        if any(jaccard_similarity(text, ref, args.duplicate_ngram) > args.duplicate_threshold for ref in reference_texts):
            if debug:
                rejected["dup_ref"] += 1
            continue
        if any(jaccard_similarity(text, acc, args.duplicate_ngram) > args.duplicate_threshold for acc in accepted_texts):
            if debug:
                rejected["dup_batch"] += 1
            continue
        accepted_texts.append(text)
        accepted_embs.append(emb)
        # store classifier confidence for weighting; if no classifier used, default 1.0
        if args.filter_mode in ("classifier", "hybrid"):
            proba_vec = model.predict_proba(emb.reshape(1, -1))[0]
            accepted_confs.append(float(proba_vec[class_index]))
        else:
            accepted_confs.append(1.0)
    if debug:
        total = len(candidates)
        print(f"Filtro {target_class}: candidatos={total}, aceptados={len(accepted_texts)}, rechazos={dict(rejected)}")
    return accepted_texts, accepted_embs, accepted_confs


def augment_class(
    client: OpenAI,
    embedder: SentenceTransformer,
    baseline_model: Pipeline,
    label_encoder: LabelEncoder,
    class_conf_thresholds: Dict[str, float],
    nn_by_class: Dict[str, NearestNeighbors],
    centroids_by_class: Dict[str, np.ndarray],
    class_overrides: Optional[Dict[str, Dict[str, Any]]],
    class_name: str,
    class_texts: Sequence[str],
    class_embeddings: np.ndarray,
    args: argparse.Namespace,
    rng: random.Random,
    baseline_f1_scores: Optional[Dict[str, float]] = None,
    all_train_embeddings: Optional[np.ndarray] = None,
    all_train_labels: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[np.ndarray], List[float]]:
    """
    Augment a single class with synthetic samples.

    New parameters for anchor quality improvements:
    - all_train_embeddings: All training embeddings (for K-NN purity in gate)
    - all_train_labels: All training labels (for K-NN purity in gate)
    """

    # ========================================================================
    # PHASE 1: ANCHOR QUALITY GATE
    # ========================================================================
    if (args.enable_anchor_gate and ANCHOR_QUALITY_AVAILABLE and
        all_train_embeddings is not None and all_train_labels is not None):

        quality_score, should_generate, metadata = check_anchor_quality(
            embeddings=all_train_embeddings,
            labels=all_train_labels,
            target_class=class_name,
            k=args.filter_knn_k,
            cohesion_weight=args.anchor_cohesion_weight,
            purity_weight=args.anchor_purity_weight,
            separation_weight=args.anchor_separation_weight,
            quality_threshold=args.anchor_quality_threshold,
        )

        if not should_generate:
            print(f"⚠️  GATE: Skipping {class_name} - quality={quality_score:.3f} < {args.anchor_quality_threshold:.3f}")
            print(f"    Cohesion={metadata.get('cohesion', 0.0):.3f}, Purity={metadata.get('knn_purity', 0.0):.3f}, Separation={metadata.get('separation', 0.0):.3f}")
            return [], [], []
        else:
            print(f"✅ GATE: {class_name} passed - quality={quality_score:.3f}")
            print(f"    Cohesion={metadata.get('cohesion', 0.0):.3f}, Purity={metadata.get('knn_purity', 0.0):.3f}, Separation={metadata.get('separation', 0.0):.3f}")

            # Phase 1b: Adaptive filter relaxation based on quality
            if args.enable_adaptive_filters:
                adaptive_params = get_adaptive_filter_params(
                    quality_score=quality_score,
                    base_params={
                        'knn_min_sim': args.filter_knn_min_sim,
                        'similarity_threshold': args.similarity_threshold,
                        'similarity_to_anchor': args.similarity_threshold,
                        'cap_ratio': args.cap_class_ratio if args.cap_class_ratio > 0 else 1.0,
                    }
                )
                print(f"📊 ADAPTIVE: Adjusted filters for quality={quality_score:.3f}")
                print(f"    knn_min_sim: {args.filter_knn_min_sim:.3f} → {adaptive_params['knn_min_sim']:.3f}")
                print(f"    threshold: {args.similarity_threshold:.3f} → {adaptive_params['similarity_threshold']:.3f}")
                print(f"    cap_ratio: {args.cap_class_ratio:.3f} → {adaptive_params['cap_ratio']:.3f}")

                # Temporarily override filter parameters for this class
                args = argparse.Namespace(**vars(args))
                args.filter_knn_min_sim = adaptive_params['knn_min_sim']
                args.similarity_threshold = adaptive_params['similarity_threshold']
                args.cap_class_ratio = adaptive_params['cap_ratio']

    # Precompile cleaner for typical MBTI label/leadin patterns
    mbti_pat = re.compile(
        r"^\s*(?:\[?(?:INFJ|INFP|INTJ|INTP|ISFJ|ISFP|ISTJ|ISTP|ENFJ|ENFP|ENTJ|ENTP|ESFJ|ESFP|ESTJ|ESTP)\]?)[:\-\s]*",
        flags=re.IGNORECASE,
    )
    leadnum_pat = re.compile(r"^\s*\d+[\)\.:\-]\s*")

    # Apply per-class overrides if provided, working on a copy of args
    if class_overrides and class_name in class_overrides:
        eff = argparse.Namespace(**vars(args))
        for k, v in class_overrides[class_name].items():
            setattr(eff, k.replace("-", "_"), v)
        args = eff

    # ========================================================================
    # PHASE 2: ANCHOR SELECTION
    # ========================================================================
    if args.enable_anchor_selection and ANCHOR_QUALITY_AVAILABLE:
        original_count = len(class_texts)
        selected_indices, selection_metadata = select_best_anchors(
            class_embeddings=class_embeddings,
            selection_ratio=args.anchor_selection_ratio,
            outlier_threshold=args.anchor_outlier_threshold,
            min_anchors=args.anchor_min_samples,
        )

        # Update class_texts and class_embeddings with selected anchors
        class_texts = [class_texts[i] for i in selected_indices]
        class_embeddings = class_embeddings[selected_indices]

        print(f"📌 SELECTION: {class_name} - {original_count} → {len(class_texts)} anchors")
        print(f"    Removed {selection_metadata['n_outliers_removed']} outliers, selected top-{int(args.anchor_selection_ratio*100)}%")
        print(f"    Mean similarity to centroid: {selection_metadata['mean_similarity_to_centroid']:.3f}")

    prompt_specs = build_prompt_specs(class_name, class_texts, class_embeddings, rng, args, baseline_f1_scores)
    if not prompt_specs:
        return [], [], []

    generated_texts: List[str] = []
    generated_embs: List[np.ndarray] = []
    generated_confs: List[float] = []
    # Quotas
    class_real_count = len(class_texts)
    class_cap_ratio = args.cap_class_ratio if args.cap_class_ratio > 0 else None
    class_cap_abs = args.cap_class_abs if args.cap_class_abs > 0 else None
    if class_cap_ratio is None and class_cap_abs is None:
        class_cap_total: Optional[int] = None
    else:
        limits: List[float] = []
        if class_cap_ratio is not None:
            limits.append(class_cap_ratio * class_real_count)
        if class_cap_abs is not None:
            limits.append(float(class_cap_abs))
        class_cap_total = int(max(0, math.floor(min(limits)))) if limits else None

    cap_cluster_ratio = args.cap_cluster_ratio if args.cap_cluster_ratio > 0 else None
    cluster_caps: Dict[int, Optional[int]] = {}
    cluster_added: Dict[int, int] = defaultdict(int)
    total_batches = math.ceil(len(prompt_specs) / args.llm_batch_size)

    for batch in tqdm(
        batched(prompt_specs, args.llm_batch_size),
        total=total_batches,
        desc=f"LLM generación {class_name}",
    ):
        outputs = call_llm_batch(
            client,
            batch,
            args.llm_model,
            args.temperature,
            args.top_p,
            args.max_retries,
        )
        for spec in batch:
            candidates = outputs.get(id(spec), [])
            if not candidates:
                continue
            # Clean candidate lines: drop numbering and accidental MBTI tags
            cleaned = []
            for line in candidates:
                t = line.strip()
                t = leadnum_pat.sub("", t)
                t = mbti_pat.sub("", t)
                t = t.strip()
                if t:
                    cleaned.append(t)
            candidates = cleaned
            candidate_embeddings = embed_candidates(
                embedder,
                normalize_texts(candidates),
                args.embedding_batch_size,
            )
            # compute anchor/neighbor embeddings (normalized) for similarity checks
            anchor_emb = compute_embeddings(
                embedder, [normalize_text(spec.anchor_text)], args.embedding_batch_size
            )[0]
            neighbor_emb = compute_embeddings(
                embedder, [normalize_text(spec.neighbor_text)], args.embedding_batch_size
            )[0]
            reference_pool = list(class_texts) + generated_texts
            nn_model = nn_by_class.get(spec.target_class)
            # Inject pools and centroids via args for filter (to avoid large signatures)
            setattr(args, "_ref_embs_class", class_embeddings)
            setattr(args, "_ref_embs_generated", np.vstack(generated_embs) if len(generated_embs) > 0 else None)
            setattr(args, "_centroids_by_class", centroids_by_class)
            accepted_texts, accepted_embs, accepted_confs = filter_candidates(
                candidates,
                candidate_embeddings,
                spec.target_class,
                spec.cluster_centroid,
                spec.similarity_floor,
                baseline_model,
                label_encoder,
                args,
                spec.cluster_texts,
                reference_pool,
                class_conf_thresholds,
                anchor_emb,
                neighbor_emb,
                nn_model,
                args.debug_filter,
                baseline_f1_scores,
            )

            # =====================================================================
            # TIER S: Adversarial Discriminator Filtering/Weighting
            # =====================================================================
            discriminator_weights = None
            if (args.use_discriminator and ADVERSARIAL_DISCRIMINATOR_AVAILABLE and
                len(accepted_embs) > 0):

                # Train discriminator on real vs synthetic candidates
                discriminator = AdversarialDiscriminator(
                    model_type=args.discriminator_model,
                    difficulty_threshold=args.discriminator_threshold,
                    random_state=args.random_seed
                )

                # Use class_embeddings as "real" and accepted_embs as "synthetic candidates"
                train_metrics = discriminator.train(
                    real_embeddings=class_embeddings,
                    synthetic_embeddings=np.array(accepted_embs)
                )

                # Phase 3: Support two modes - filter (binary) or weight (continuous)
                if args.discriminator_mode == "filter":
                    # Original TIER S behavior: binary filtering
                    filtered_indices, weights, filter_stats = discriminator.filter_synthetics(
                        synthetic_candidates=np.array(accepted_embs),
                        return_weights=True
                    )

                    # Update accepted lists with filtered results
                    if len(filtered_indices) > 0:
                        accepted_texts = [accepted_texts[i] for i in filtered_indices]
                        accepted_embs = [accepted_embs[i] for i in filtered_indices]
                        accepted_confs = [accepted_confs[i] for i in filtered_indices]

                        if args.debug_filter:
                            print(f"  🔍 DISCRIMINATOR {spec.target_class}: "
                                  f"{len(filter_stats.get('accepted', []))} accepted, "
                                  f"{len(filter_stats.get('rejected', []))} rejected, "
                                  f"acc={train_metrics.get('accuracy', 0.0):.3f}")
                    else:
                        # All rejected by discriminator
                        accepted_texts = []
                        accepted_embs = []
                        accepted_confs = []

                elif args.discriminator_mode == "weight":
                    # Phase 3: Use discriminator probabilities as continuous weights
                    probs = discriminator.discriminator.predict_proba(np.array(accepted_embs))[:, 1]

                    # Convert probability to weight: weight = 1.0 - prob
                    # High prob (obviously synthetic) → low weight
                    # Low prob (indistinguishable) → high weight
                    discriminator_weights = 1.0 - probs

                    # Hard reject only if prob > 0.9 (obviously fake)
                    keep_mask = probs < 0.9
                    rejected_count = (~keep_mask).sum()

                    if keep_mask.sum() > 0:
                        # Validate keep_mask length matches accepted_texts length
                        if len(keep_mask) != len(accepted_texts):
                            raise ValueError(f"keep_mask length ({len(keep_mask)}) != accepted_texts length ({len(accepted_texts)})")
                        accepted_texts = [accepted_texts[i] for i in range(len(accepted_texts)) if keep_mask[i]]
                        accepted_embs = [accepted_embs[i] for i in range(len(accepted_embs)) if keep_mask[i]]
                        accepted_confs = [accepted_confs[i] for i in range(len(accepted_confs)) if keep_mask[i]]
                        discriminator_weights = discriminator_weights[keep_mask]

                        if args.debug_filter:
                            print(f"  ⚖️  DISCRIMINATOR-WEIGHT {spec.target_class}: "
                                  f"{len(accepted_texts)} kept (weighted), "
                                  f"{rejected_count} rejected (prob>0.9), "
                                  f"weight_range=[{discriminator_weights.min():.3f}, {discriminator_weights.max():.3f}], "
                                  f"acc={train_metrics['accuracy']:.3f}")
                    else:
                        # All rejected
                        accepted_texts = []
                        accepted_embs = []
                        accepted_confs = []
                        discriminator_weights = None

            # Enforce quotas per cluster and per class before adding
            # Initialize cluster cap if needed
            if spec.cluster_id not in cluster_caps:
                if cap_cluster_ratio is None:
                    cluster_caps[spec.cluster_id] = None
                else:
                    cluster_caps[spec.cluster_id] = int(
                        max(0, math.floor(spec.cluster_size * cap_cluster_ratio))
                    )

            allow = len(accepted_texts)
            cluster_cap = cluster_caps.get(spec.cluster_id)
            if class_cap_total is not None:
                remaining_class = max(0, class_cap_total - len(generated_texts))
                allow = min(allow, remaining_class)
            if cluster_cap is not None:
                remaining_cluster = max(0, cluster_cap - cluster_added[spec.cluster_id])
                allow = min(allow, remaining_cluster)

            if allow < len(accepted_texts):
                accepted_texts = accepted_texts[:allow]
                accepted_embs = accepted_embs[:allow]
                if 'accepted_confs' in locals():
                    accepted_confs = accepted_confs[:allow]
            added_filter = len(accepted_texts)
            cluster_added[spec.cluster_id] += added_filter
            # Grace acceptance: if none accepted, pick top by anchor/centroid similarity under relaxed thresholds
            grace_topk_eff = args.grace_topk
            if args.grace_disable_after > 0 and len(generated_texts) >= args.grace_disable_after:
                grace_topk_eff = 0
            if added_filter == 0 and grace_topk_eff > 0 and len(candidates) > 0:
                centroid_vec = spec.cluster_centroid.reshape(1, -1)
                sims_centroid = cosine_similarity(candidate_embeddings, centroid_vec).reshape(-1)
                sims_anchor = cosine_similarity(candidate_embeddings, anchor_emb.reshape(1, -1)).reshape(-1)
                sims_neighbor = cosine_similarity(candidate_embeddings, neighbor_emb.reshape(1, -1)).reshape(-1)
                sims_to_anchor = np.maximum(sims_anchor, sims_neighbor)
                combined = 0.6 * sims_to_anchor + 0.4 * sims_centroid
                order = np.argsort(-combined)
                taken = 0
                grace_taken = 0
                for idx in order:
                    if taken >= grace_topk_eff:
                        break
                    text = candidates[idx]
                    emb = candidate_embeddings[idx]
                    token_count = len(text.split())
                    if token_count < args.min_tokens or token_count > args.max_tokens:
                        continue
                    # require at least minimal anchor similarity OR combined score
                    if sims_to_anchor[idx] < args.grace_min_anchor and combined[idx] < args.grace_min_score:
                        continue
                    proba_vec = baseline_model.predict_proba(emb.reshape(1, -1))[0]
                    pred_idx = int(np.argmax(proba_vec))
                    # Validate prediction index
                    if pred_idx >= len(label_encoder.classes_):
                        raise ValueError(f"Invalid prediction index: {pred_idx}")
                    pred_class = label_encoder.classes_[pred_idx]
                    # margin check
                    order2 = np.argsort(proba_vec)[::-1]
                    # Validate order has elements
                    if len(order2) == 0:
                        best2 = 0.0
                        second2 = 0.0
                    else:
                        best2 = float(proba_vec[order2[0]])
                        second2 = float(proba_vec[order2[1]]) if len(order2) > 1 else 0.0
                    margin_ok = (best2 - second2) >= getattr(args, "clf_margin", 0.0)
                    if not (pred_class == spec.target_class and proba_vec[pred_idx] >= args.grace_min_conf and margin_ok):
                        continue
                    # repel non-target centroids if enabled
                    if getattr(args, "repel_nontarget_sim", 0.0) > 0:
                        max_other = -1.0
                        for cls2, cvec in centroids_by_class.items():
                            if cls2 == spec.target_class:
                                continue
                            s = float(cosine_similarity(emb.reshape(1, -1), cvec.reshape(1, -1))[0][0])
                            if s > max_other:
                                max_other = s
                        if max_other >= args.repel_nontarget_sim:
                            continue
                    # semantic dedup via embeddings if enabled
                    if getattr(args, "dedup_embed_sim", 0.0) > 0:
                        # vs class embeddings
                        try:
                            nn_c = NearestNeighbors(n_neighbors=1, metric="cosine").fit(class_embeddings)
                            d, _ = nn_c.kneighbors(emb.reshape(1, -1), return_distance=True)
                            if float(1.0 - d.reshape(-1)[0]) >= args.dedup_embed_sim:
                                continue
                        except Exception:
                            pass
                        # vs already generated
                        if len(generated_embs) > 0:
                            try:
                                nn_g = NearestNeighbors(n_neighbors=1, metric="cosine").fit(np.vstack(generated_embs))
                                d, _ = nn_g.kneighbors(emb.reshape(1, -1), return_distance=True)
                                if float(1.0 - d.reshape(-1)[0]) >= args.dedup_embed_sim:
                                    continue
                            except Exception:
                                pass
                    # dedup against references and already accepted
                    if any(jaccard_similarity(text, ref, args.duplicate_ngram) > args.duplicate_threshold for ref in reference_pool):
                        continue
                    if any(jaccard_similarity(text, acc, args.duplicate_ngram) > args.duplicate_threshold for acc in generated_texts):
                        continue
                    # Quotas for grace
                    if class_cap_total is not None:
                        remaining_class = class_cap_total - (
                            len(generated_texts) + added_filter + grace_taken
                        )
                        if remaining_class <= 0:
                            break
                    if cluster_cap is not None:
                        remaining_cluster = cluster_cap - (
                            cluster_added[spec.cluster_id] + grace_taken
                        )
                        if remaining_cluster <= 0:
                            break
                    accepted_texts.append(text)
                    accepted_embs.append(emb)
                    # confidence for grace-accepted candidate (p(target))
                    conf_val = float(proba_vec[pred_idx])
                    if 'accepted_confs' in locals():
                        accepted_confs.append(conf_val)
                    taken += 1
                    grace_taken += 1
                if args.debug_filter:
                    print(
                        f"Grace {spec.target_class}: tomados={taken}, max_anchor={float(sims_to_anchor.max()):.2f}, "
                        f"max_centroid={float(sims_centroid.max()):.2f}, max_combined={float(combined.max()):.2f}"
                    )
                cluster_added[spec.cluster_id] += grace_taken
            generated_texts.extend(accepted_texts)
            generated_embs.extend(accepted_embs)
            # ensure accepted_confs exists and matches length
            if 'accepted_confs' in locals() and len(accepted_confs) == len(accepted_texts):
                generated_confs.extend(accepted_confs)
            else:
                # fallback: fill with 1.0 for any without confidence
                generated_confs.extend([1.0] * len(accepted_texts))
        tqdm.write(f"Clase {class_name}: {len(generated_texts)} sintéticos acumulados")
    # Clean helper attributes used for filtering
    for attr in ("_ref_embs_class", "_ref_embs_generated", "_centroids_by_class"):
        if hasattr(args, attr):
            delattr(args, attr)
    return generated_texts, generated_embs, generated_confs


def calculate_contamination(n_synthetic: int, n_real: int, purity: float) -> float:
    """
    Phase 3: Calculate contamination score.

    contamination = (n_synthetic / n_real) × (1 - purity)

    Higher contamination = more risk of degrading performance
    """
    if n_real == 0:
        return 0.0
    ratio = n_synthetic / n_real
    contamination = ratio * (1.0 - purity)
    return contamination


def evaluate_with_synthetics_val_gating(
    baseline_model,
    baseline_embeddings: np.ndarray,
    baseline_labels: np.ndarray,
    synthetic_embeddings: np.ndarray,
    synthetic_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    class_weight: Optional[str] = None,
) -> float:
    """
    Phase 3: Evaluate augmented model on validation set.

    Returns:
        val_f1: Macro F1 score on validation set
    """
    # Combine baseline + synthetics
    augmented_embeddings = np.vstack([baseline_embeddings, synthetic_embeddings])
    augmented_labels = np.concatenate([baseline_labels, synthetic_labels])

    # Train model on augmented data
    # Note: multi_class='multinomial' is now default with solver='lbfgs', removed to avoid deprecation warning
    augmented_model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
        solver='lbfgs'
    )
    augmented_model.fit(augmented_embeddings, augmented_labels)

    # Evaluate on val
    y_val_pred = augmented_model.predict(val_embeddings)
    val_f1 = f1_score(val_labels, y_val_pred, average='macro')

    return val_f1


def orchestrate_pipeline(args: argparse.Namespace) -> None:
    load_dotenv()
    # Carga dataset y realiza split, o usa CSVs predefinidos si se entregan
    if getattr(args, "train_csv", None) and getattr(args, "test_csv", None):
        train_df = load_dataset(args.train_csv)
        test_df = load_dataset(args.test_csv)
    else:
        df = load_dataset(args.data_path)
        split_seed = args.split_seed if getattr(args, "split_seed", None) is not None else args.random_seed
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=split_seed,
            stratify=df["label"],
        )

    # Phase 3: Validation-based gating - split train into train/val
    val_df = None
    if args.use_val_gating:
        print(f"\n{'='*70}")
        print(f"Phase 3: Validation-Based Gating")
        print(f"{'='*70}")
        print(f"Val size: {args.val_size:.1%}, tolerance: {args.val_tolerance:.3f}")

        train_df, val_df = train_test_split(
            train_df,
            test_size=args.val_size,
            random_state=split_seed,
            stratify=train_df["label"],
        )
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print()

    # Configurar dispositivo para embeddings (GPU acelera fuerte en 3070)
    # Some models require trust_remote_code=True (nomic, jina, gte-base)
    if getattr(args, "device", "auto") == "auto":
        embedder = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    else:
        embedder = SentenceTransformer(args.embedding_model, device=args.device, trust_remote_code=True)
    try:
        target_device = getattr(embedder, "_target_device", None)
        if target_device is None:
            target_device = next(embedder._first_module().parameters()).device
        print(f"Embedder device: {target_device}")
    except Exception:
        pass
    # Normaliza textos para embeddings (quita URLs, separadores, etc.)
    train_proc = normalize_texts(train_df["text"])
    test_proc = normalize_texts(test_df["text"])
    train_embeddings = compute_embeddings(embedder, train_proc, args.embedding_batch_size)
    test_embeddings = compute_embeddings(embedder, test_proc, args.embedding_batch_size)

    # Phase 3: Compute val embeddings if val-gating is enabled
    val_embeddings = None
    y_val = None
    if val_df is not None:
        val_proc = normalize_texts(val_df["text"])
        val_embeddings = compute_embeddings(embedder, val_proc, args.embedding_batch_size)

    label_encoder = LabelEncoder()
    # Asegura mapeo consistente cuando se usan splits predefinidos
    all_labels = [train_df["label"], test_df["label"]]
    if val_df is not None:
        all_labels.append(val_df["label"])
    label_encoder.fit(pd.concat(all_labels, axis=0))
    y_train = label_encoder.transform(train_df["label"])
    y_test = label_encoder.transform(test_df["label"])
    if val_df is not None:
        y_val = label_encoder.transform(val_df["label"])

    # Optionally balance baseline
    class_weight = "balanced" if getattr(args, "balanced_baseline", False) else None
    baseline_model = train_baseline_classifier(train_embeddings, y_train, class_weight=class_weight)
    baseline_metrics = evaluate_model(baseline_model, test_embeddings, y_test, label_encoder, "baseline")
    print("=== Baseline macro-F1 ===", baseline_metrics["macro_f1"])

    # V3 Phase 1 - Extract per-class F1 scores for adaptive thresholds and multipliers
    baseline_f1_by_class: Dict[str, float] = {}
    report = baseline_metrics.get("report", {})
    for cls in label_encoder.classes_:
        if cls in report:
            cls_metrics = report.get(cls, {})
            baseline_f1_by_class[cls] = cls_metrics.get("f1-score", 0.0)

    if args.only_baseline:
        print("Solo se ejecutó el baseline por la bandera --only-baseline")
        return

    client = ensure_openai_client()
    gen_seed = args.gen_seed if getattr(args, "gen_seed", None) is not None else args.random_seed
    rng = random.Random(gen_seed)

    target_classes = select_minority_classes(
        train_df["label"], args.target_classes or None, args.minority_percentile
    )
    print(f"Clases objetivo para augmentación: {target_classes}")

    # Report class counts for target classes
    print(f"\n📊 Class Sample Counts:")
    for cls in target_classes:
        train_count = (train_df["label"] == cls).sum()
        test_count = (test_df["label"] == cls).sum()
        total_count = train_count + test_count
        train_pct = train_count / len(train_df) * 100
        print(f"   {cls}: {total_count} total (train: {train_count} [{train_pct:.2f}%], test: {test_count})")
    print()

    # Load class-specific overrides if provided
    if getattr(args, "class_config", None):
        overrides = load_class_overrides(args.class_config)
        setattr(args, "_class_overrides", overrides)

    # Dynamic class-wise confidence floors from real training data
    class_conf_thresholds: Dict[str, float] = {}
    train_proba = baseline_model.predict_proba(train_embeddings)
    for idx, cls in enumerate(label_encoder.classes_):
        mask = y_train == idx
        if mask.sum() < 5:
            class_conf_thresholds[cls] = max(0.4, args.min_classifier_confidence - 0.2)
        else:
            class_conf_thresholds[cls] = float(
                np.quantile(train_proba[mask, idx], args.class_conf_quantile)
            )

    # Build per-class kNN models and centroids on normalized embeddings for optional filtering
    nn_by_class: Dict[str, NearestNeighbors] = {}
    centroids_by_class: Dict[str, np.ndarray] = {}
    for idx, cls in enumerate(label_encoder.classes_):
        mask = y_train == idx
        class_embs = train_embeddings[mask]
        if len(class_embs) == 0:
            continue
        centroids_by_class[str(cls)] = class_embs.mean(axis=0)
        n_neighbors = int(min(max(3, args.filter_knn_k), len(class_embs)))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(class_embs)
        nn_by_class[cls] = nn

    synthetic_texts: List[str] = []
    synthetic_labels: List[str] = []
    synthetic_embeddings: List[np.ndarray] = []

    # Initialize quality gate predictor (Phase 2: Enhanced)
    predictor = None
    enhanced_gate = None
    prediction_results = []

    if ENHANCED_GATE_AVAILABLE:
        enhanced_gate = EnhancedQualityGate(
            min_anchor_quality=0.35,  # Lowered from 0.40
            decision_mode="probabilistic",  # Phase 2 feature
            purity_low_threshold=0.30,
            f1_high_threshold=0.45,
            f1_skip_threshold=0.60  # Lowered from 0.65
        )
        print("✨ Phase 2 Enhanced Quality Gate enabled (probabilistic)")
    elif QUALITY_GATE_AVAILABLE:
        predictor = QualityGatePredictor()
        print("🎯 Phase 1 Quality Gate Predictor enabled (fallback)")

    # Load per-class overrides if provided
    class_overrides: Dict[str, Dict[str, Any]] = {}
    if getattr(args, "_class_overrides", None):
        class_overrides = getattr(args, "_class_overrides")

    for cls in target_classes:
        mask = train_df["label"] == cls
        class_texts_raw = list(train_df.loc[mask, "text"])
        class_texts = normalize_texts(class_texts_raw)
        class_embs = train_embeddings[mask.values]

        # === Phase 2: Enhanced Quality Gate Prediction ===
        prediction_decision = None
        prediction_confidence = None
        prediction_reasons = None
        gate_result = None

        # Extract metrics
        n_samples = len(class_texts)
        baseline_f1 = baseline_f1_by_class.get(cls, 0.0)
        n_clusters_est = max(6, min(12, len(class_embs) // 60))

        # Compute anchor quality metrics
        quality_score = 0.5
        purity = 0.5
        cohesion = 0.5
        if ANCHOR_QUALITY_AVAILABLE:
            quality_score, _, metadata = check_anchor_quality(
                embeddings=train_embeddings,
                labels=train_df["label"].values,
                target_class=cls,
                k=args.filter_knn_k,
                cohesion_weight=args.anchor_cohesion_weight,
                purity_weight=args.anchor_purity_weight,
                separation_weight=args.anchor_separation_weight,
                quality_threshold=0.0,  # Don't gate, just get metrics
            )
            purity = metadata.get('knn_purity', 0.5)
            cohesion = metadata.get('cohesion', 0.5)

        # Use Phase 2 Enhanced Quality Gate if available
        if enhanced_gate is not None:
            gate_result = enhanced_gate.predict(
                n_samples=n_samples,
                baseline_f1=baseline_f1,
                quality_score=quality_score,
                purity=purity,
                cohesion=cohesion,
                n_clusters=n_clusters_est
            )

            prediction_decision = gate_result['decision']
            prediction_confidence = gate_result['confidence']
            prediction_reasons = {"reason": gate_result['reason']}

            print(f"\n✨ Phase 2 Enhanced Quality Gate for {cls}:")
            print(f"   Decision: {prediction_decision} (confidence: {prediction_confidence:.2f})")
            print(f"   Metrics: n={n_samples}, F1={baseline_f1:.3f}")
            print(f"   Quality: {quality_score:.3f}, Purity: {purity:.3f}, Cohesion: {cohesion:.3f}")
            print(f"   Reason: {gate_result['reason']}")
            print(f"   Budget: {gate_result['budget']} synthetics")

        # Fallback to Phase 1 quality gate
        elif predictor is not None and ANCHOR_QUALITY_AVAILABLE:
            metrics = ClassMetrics(
                n_samples=n_samples,
                n_clusters=n_clusters_est,
                baseline_f1=baseline_f1,
                anchor_cohesion=cohesion,
                anchor_purity=purity,
                anchor_separation=metadata.get('separation', 0.0) if ANCHOR_QUALITY_AVAILABLE else 0.0,
                anchor_quality_score=quality_score,
            )

            prediction_decision, prediction_confidence, prediction_reasons = predictor.predict(metrics)

            print(f"\n🎯 Phase 1 Quality Gate Prediction for {cls}:")
            print(f"   Decision: {prediction_decision} (confidence: {prediction_confidence:.2f})")
            print(f"   Metrics: n={n_samples}, F1={baseline_f1:.3f}, quality={quality_score:.3f}")
            for key, reason in prediction_reasons.items():
                print(f"   {reason}")

        # === Phase 2: F1 Gate + Enhanced Dynamic Budgets ===
        # Phase 2: Lowered F1 threshold (0.65 → 0.60)
        f1_skip_threshold = 0.60 if enhanced_gate is not None else 0.65

        if baseline_f1 > f1_skip_threshold:
            print(f"   ⏭️  Skipping {cls}: F1={baseline_f1:.3f} > {f1_skip_threshold} (diminishing returns)")
            # Store empty prediction result
            if prediction_decision is not None:
                prediction_results.append({
                    "class": cls,
                    "decision": "skip_f1_gate",
                    "confidence": 1.0,
                    "reasons": {"f1_gate": f"✅ High baseline F1 ({baseline_f1:.3f}) - skip generation"},
                    "n_samples": n_samples,
                    "baseline_f1": baseline_f1,
                    "n_clusters": n_clusters_est,
                    "quality_score": quality_score,
                    "synthetics_generated": 0,
                })
            continue  # Skip to next class

        # Calculate dynamic budget
        # Phase 3: Apply F1-based budget scaling if enabled
        f1_multiplier = 1.0
        if args.use_f1_budget_scaling:
            high_f1_threshold = args.f1_budget_thresholds[0]  # Default: 0.35
            medium_f1_threshold = args.f1_budget_thresholds[1]  # Default: 0.20

            if baseline_f1 >= high_f1_threshold:
                f1_multiplier = args.f1_budget_multipliers[0]  # Default: 0.0 (skip)
                f1_level = "HIGH"
            elif baseline_f1 >= medium_f1_threshold:
                f1_multiplier = args.f1_budget_multipliers[1]  # Default: 0.5
                f1_level = "MEDIUM"
            else:
                f1_multiplier = args.f1_budget_multipliers[2]  # Default: 1.0
                f1_level = "LOW"

            print(f"   📊 Phase 3 F1-Based Budget Scaling:")
            print(f"      Baseline F1: {baseline_f1:.3f} ({f1_level})")
            print(f"      Budget multiplier: {f1_multiplier}×")

            if f1_multiplier == 0.0:
                print(f"   ⏭️  Skipping {cls}: High F1 ({baseline_f1:.3f}) - augmentation may degrade")
                # Store empty prediction result
                if prediction_decision is not None:
                    prediction_results.append({
                        "class": cls,
                        "decision": "skip_f1_budget",
                        "confidence": 1.0,
                        "reasons": {"f1_budget": f"✅ High baseline F1 ({baseline_f1:.3f}) - skip to prevent degradation"},
                        "n_samples": n_samples,
                        "baseline_f1": baseline_f1,
                        "n_clusters": n_clusters_est,
                        "quality_score": quality_score,
                        "synthetics_generated": 0,
                    })
                continue  # Skip to next class

        # Phase 3: Apply contamination control if enabled
        if args.use_contamination_control and quality_score is not None and purity is not None:
            contamination = calculate_contamination(
                n_synthetic=min(args.cap_class_abs, n_samples),
                n_real=n_samples,
                purity=purity
            )

            # Apply contamination-based budget reduction
            if contamination > args.contamination_threshold:
                cont_multiplier = args.contamination_multipliers[0]  # High contamination: 0.3
                contamination_level = "HIGH"
            elif contamination > args.contamination_threshold * 0.6:
                cont_multiplier = args.contamination_multipliers[1]  # Medium: 0.7
                contamination_level = "MEDIUM"
            else:
                cont_multiplier = args.contamination_multipliers[2]  # Low: 1.0
                contamination_level = "LOW"

            print(f"   ⚠️  Phase 3 Contamination Control:")
            print(f"      Contamination: {contamination:.4f} ({contamination_level})")
            print(f"      Budget multiplier: {cont_multiplier}× (threshold: {args.contamination_threshold})")

        else:
            cont_multiplier = 1.0

        # Phase 2: Use enhanced budget if available, otherwise fall back to Phase 1
        if enhanced_gate is not None and gate_result is not None:
            # Phase 2: Use budget from enhanced gate (includes purity + F1 multipliers)
            dynamic_budget = gate_result['budget']
            # Apply F1 multiplier, then contamination multiplier
            dynamic_budget = int(dynamic_budget * f1_multiplier * cont_multiplier)

            budget_reason = gate_result['reason']
            print(f"   💰 Phase 2 Enhanced Budget:")
            for line in budget_reason.split('\n'):
                print(f"      {line}")
            if f1_multiplier != 1.0 or cont_multiplier != 1.0:
                print(f"      After F1 ({f1_multiplier}×) + contamination ({cont_multiplier}×): {dynamic_budget}")

            original_cap = args.cap_class_abs
            args.cap_class_abs = dynamic_budget

        elif predictor is not None and ANCHOR_QUALITY_AVAILABLE:
            # Phase 1: Simple quality-based budget
            dynamic_budget, budget_reason = calculate_dynamic_budget(n_samples, quality_score)
            # Apply F1 multiplier, then contamination multiplier
            dynamic_budget = int(dynamic_budget * f1_multiplier * cont_multiplier)
            print(f"   💰 Phase 1 Dynamic Budget: {budget_reason}")
            if f1_multiplier != 1.0 or cont_multiplier != 1.0:
                print(f"      After F1 ({f1_multiplier}×) + contamination ({cont_multiplier}×): {dynamic_budget}")

            original_cap = args.cap_class_abs
            args.cap_class_abs = dynamic_budget
        else:
            # No quality gate: use original cap
            original_cap = args.cap_class_abs
            dynamic_budget = int(original_cap * f1_multiplier * cont_multiplier)
            if f1_multiplier != 1.0 or cont_multiplier != 1.0:
                args.cap_class_abs = dynamic_budget

        # === Generate synthetics ===
        synth_count_before = len(synthetic_texts)
        texts, embs, confs = augment_class(
            client,
            embedder,
            baseline_model,
            label_encoder,
            class_conf_thresholds,
            nn_by_class,
            centroids_by_class,
            class_overrides,
            cls,
            class_texts,
            class_embs,
            args,
            rng,
            baseline_f1_by_class,
            all_train_embeddings=train_embeddings,
            all_train_labels=train_df["label"].values,
        )
        # Phase 3: Validation-based gating - reject synthetics if val degrades
        if args.use_val_gating and val_embeddings is not None and len(texts) > 0:
            # Get class-specific validation data (to measure per-class improvement)
            class_mask_val = val_df["label"] == cls
            class_val_embs = val_embeddings[class_mask_val.values]
            class_val_labels = y_val[class_mask_val.values]

            # Baseline val F1 (without synthetics) - use FULL training set
            baseline_val_f1 = evaluate_with_synthetics_val_gating(
                baseline_model,
                train_embeddings,  # FIX: Use full training set (all classes)
                y_train,            # FIX: Use full labels (all classes)
                np.zeros((0, train_embeddings.shape[1])),  # No synthetics
                np.array([]),
                class_val_embs,
                class_val_labels,
                class_weight=class_weight
            )

            # Augmented val F1 (with synthetics) - use FULL training set + synthetics
            class_synth_embs = np.array(embs)
            class_synth_labels = np.array([label_encoder.transform([cls])[0]] * len(embs))

            augmented_val_f1 = evaluate_with_synthetics_val_gating(
                baseline_model,
                train_embeddings,  # FIX: Use full training set (all classes)
                y_train,            # FIX: Use full labels (all classes)
                class_synth_embs,   # Add synthetics for this class only
                class_synth_labels,
                class_val_embs,
                class_val_labels,
                class_weight=class_weight
            )

            val_delta = augmented_val_f1 - baseline_val_f1

            print(f"   📊 Phase 3 Val-Gating for {cls}:")
            print(f"      Baseline val F1: {baseline_val_f1:.4f}")
            print(f"      Augmented val F1: {augmented_val_f1:.4f}")
            print(f"      Delta: {val_delta:+.4f} (tolerance: {args.val_tolerance:.4f})")

            # Gate: reject if val degrades beyond tolerance
            if val_delta < -args.val_tolerance:
                print(f"      ❌ REJECTED: Val degraded by {val_delta:.4f} (>{args.val_tolerance:.4f})")
                texts = []
                embs = []
                confs = []
            else:
                print(f"      ✅ ACCEPTED: Val {'improved' if val_delta > 0 else 'neutral'}")

        synthetic_texts.extend(texts)
        synthetic_labels.extend([cls] * len(texts))
        synthetic_embeddings.extend(embs)

        # Restore original cap_class_abs
        if predictor is not None and ANCHOR_QUALITY_AVAILABLE:
            args.cap_class_abs = original_cap

        # Store prediction info for later validation
        if prediction_decision is not None:
            prediction_results.append({
                "class": cls,
                "decision": prediction_decision,
                "confidence": prediction_confidence,
                "reasons": prediction_reasons,
                "n_samples": n_samples,
                "baseline_f1": baseline_f1,
                "n_clusters": n_clusters_est,
                "quality_score": quality_score,
                "synthetics_generated": len(texts),
            })
        # store confidences for weighting (align length if needed)
        if not hasattr(args, "_synthetic_confs"):
            setattr(args, "_synthetic_confs", [])
        if len(confs) != len(texts):
            # pad/truncate to match
            if len(confs) < len(texts):
                pad_val = float(np.mean(confs)) if len(confs) > 0 else float(args.min_classifier_confidence)
                confs = list(confs) + [pad_val] * (len(texts) - len(confs))
            else:
                confs = list(confs[: len(texts)])
        getattr(args, "_synthetic_confs").extend(confs)

    if not synthetic_texts:
        print("No se generaron ejemplos sintéticos tras el filtrado.")
        return

    synthetic_embeddings_array = np.vstack(synthetic_embeddings)
    synthetic_df = pd.DataFrame({"label": synthetic_labels, "text": synthetic_texts})
    synthetic_df.to_csv(args.synthetic_output, index=False)
    print(f"Guardados {len(synthetic_df)} ejemplos sintéticos en {args.synthetic_output}")

    augmented_embeddings = np.vstack([train_embeddings, synthetic_embeddings_array])
    augmented_labels = np.concatenate([y_train, label_encoder.transform(synthetic_labels)])
    sample_weights = np.ones_like(augmented_labels, dtype=float)

    # EXPERIMENT 1 (PHASE B): Adaptive synthetic weighting based on class-specific baseline F1
    if getattr(args, "enable_adaptive_weighting", False):
        print("\n" + "="*70)
        print("🎯 PHASE B: Applying class-specific adaptive weights")
        print("="*70)

        # Apply class-specific weights to each synthetic sample
        for i in range(len(y_train), len(augmented_labels)):
            syn_class_encoded = augmented_labels[i]
            syn_class_name = label_encoder.inverse_transform([syn_class_encoded])[0]
            baseline_f1 = baseline_f1_by_class.get(syn_class_name, 0.0)

            adaptive_weight = get_adaptive_synthetic_weight(
                class_name=syn_class_name,
                baseline_f1=baseline_f1,
                default_weight=args.synthetic_weight
            )
            sample_weights[i] = adaptive_weight

        # Print summary of weights applied
        print("\nWeights applied by class:")
        weight_summary = {}
        for cls in label_encoder.classes_:
            cls_mask = augmented_labels[len(y_train):] == label_encoder.transform([cls])[0]
            if cls_mask.any():
                cls_weights = sample_weights[len(y_train):][cls_mask]
                avg_weight = float(np.mean(cls_weights))
                n_samples = int(np.sum(cls_mask))
                baseline_f1 = baseline_f1_by_class.get(cls, 0.0)
                weight_summary[cls] = {
                    "baseline_f1": baseline_f1,
                    "weight": avg_weight,
                    "n_synthetics": n_samples
                }

        # Sort by baseline F1 and print
        for cls in sorted(weight_summary.keys(), key=lambda c: weight_summary[c]["baseline_f1"]):
            info = weight_summary[cls]
            print(f"   {cls:4s}: F1={info['baseline_f1']:.3f} → weight={info['weight']:.2f} (n={info['n_synthetics']})")
        print("="*70 + "\n")

    elif getattr(args, "synthetic_weight_mode", "flat") == "confidence" and hasattr(args, "_synthetic_confs"):
        confs = np.array(getattr(args, "_synthetic_confs"), dtype=float)
        confs = np.clip(confs, 0.0, 1.0)
        n_syn = len(augmented_labels) - len(y_train)
        # Align conf vector length with number of synthetics
        if len(confs) < n_syn:
            pad_val = float(np.mean(confs)) if len(confs) > 0 else float(args.min_classifier_confidence)
            confs = np.concatenate([confs, np.full(n_syn - len(confs), pad_val, dtype=float)])
        elif len(confs) > n_syn:
            confs = confs[:n_syn]
        sample_weights[len(y_train) :] = args.synthetic_weight * confs
    else:
        sample_weights[len(y_train) :] = args.synthetic_weight

    # When using sample_weight, don't also use class_weight (they're different strategies)
    # Only use class_weight if we're NOT using adaptive weighting
    use_class_weight = None if getattr(args, "enable_adaptive_weighting", False) else class_weight

    augmented_model = train_baseline_classifier(
        augmented_embeddings, augmented_labels, sample_weight=sample_weights, class_weight=use_class_weight
    )
    augmented_metrics = evaluate_model(
        augmented_model, test_embeddings, y_test, label_encoder, "hybrid"
    )
    print("=== Hybrid macro-F1 ===", augmented_metrics["macro_f1"])

    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    augmented_df.to_csv(args.augmented_train_output, index=False)
    print(f"Dataset de entrenamiento aumentado guardado en {args.augmented_train_output}")

    # Calculate improvement metrics
    baseline_f1 = baseline_metrics.get("macro_f1", 0.0)
    augmented_f1 = augmented_metrics.get("macro_f1", 0.0)
    f1_delta_abs = augmented_f1 - baseline_f1
    f1_delta_pct = (f1_delta_abs / baseline_f1 * 100) if baseline_f1 > 0 else 0.0

    # Calculate per-class improvements for target classes
    per_class_improvements = {}
    for cls in target_classes:
        baseline_cls_f1 = baseline_metrics["report"].get(cls, {}).get("f1-score", 0.0)
        augmented_cls_f1 = augmented_metrics["report"].get(cls, {}).get("f1-score", 0.0)
        per_class_improvements[cls] = {
            "baseline_f1": baseline_cls_f1,
            "augmented_f1": augmented_cls_f1,
            "delta_abs": augmented_cls_f1 - baseline_cls_f1,
            "delta_pct": ((augmented_cls_f1 - baseline_cls_f1) / baseline_cls_f1 * 100) if baseline_cls_f1 > 0 else 0.0,
        }

    # Validate quality gate predictions
    validated_predictions = []
    if predictor is not None and prediction_results:
        print("\n" + "="*70)
        print("QUALITY GATE PREDICTION VALIDATION")
        print("="*70)
        for pred_info in prediction_results:
            cls = pred_info.get("class", "UNKNOWN")
            decision = pred_info.get("decision", "unknown")
            if cls == "UNKNOWN" or decision == "unknown":
                print(f"Warning: Invalid pred_info: {pred_info}")
                continue
            if cls not in per_class_improvements:
                print(f"Warning: Class {cls} not in per_class_improvements, skipping")
                continue
            improvement = per_class_improvements[cls]["delta_abs"]

            # Validate using predictor
            validation_result = predictor.validate_prediction(decision, improvement)

            validated_predictions.append({
                "class": cls,
                "decision": decision,
                "confidence": pred_info["confidence"],
                "improvement": improvement,
                "validation": validation_result,
                "metrics": {
                    "n_samples": pred_info["n_samples"],
                    "baseline_f1": pred_info["baseline_f1"],
                    "n_clusters": pred_info["n_clusters"],
                    "quality_score": pred_info["quality_score"],
                    "synthetics_generated": pred_info["synthetics_generated"],
                }
            })

            # Print validation result
            status_emoji = "✅" if validation_result in ["TP", "TN"] else "❌"
            print(f"\n{status_emoji} {cls}: {validation_result}")
            print(f"   Predicted: {decision} (confidence: {pred_info['confidence']:.2f})")
            print(f"   Actual: {'improved' if improvement > 0 else 'degraded/neutral'} ({improvement:+.4f})")
            print(f"   Synthetics: {pred_info['synthetics_generated']}, Quality: {pred_info['quality_score']:.3f}")

        # Calculate confusion matrix statistics
        tp = sum(1 for p in validated_predictions if p["validation"] == "TP")
        tn = sum(1 for p in validated_predictions if p["validation"] == "TN")
        fp = sum(1 for p in validated_predictions if p["validation"] == "FP")
        fn = sum(1 for p in validated_predictions if p["validation"] == "FN")
        total = len(validated_predictions)

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n" + "="*70)
        print(f"CONFUSION MATRIX:")
        print(f"  TP: {tp} | FP: {fp}")
        print(f"  FN: {fn} | TN: {tn}")
        print(f"\nMETRICS:")
        print(f"  Accuracy:  {accuracy:.2%} ({tp + tn}/{total})")
        print(f"  Precision: {precision:.2%} (when says 'generate', correct {precision:.0%} of time)")
        print(f"  Recall:    {recall:.2%} (detects {recall:.0%} of cases that help)")
        print(f"  F1-Score:  {f1:.2%}")
        print("="*70)

    # Prepare metrics dict
    metrics_dict = {
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,  # Changed from 'hybrid' to 'augmented' for consistency
        "improvement": {
            "f1_delta_abs": f1_delta_abs,
            "f1_delta_pct": f1_delta_pct,
            "per_class": per_class_improvements,
        },
        "target_classes": target_classes,
        "synthetic_data": {
            "accepted_count": len(synthetic_df),
        },
    }

    # Add quality gate prediction results if available
    if validated_predictions:
        tp = sum(1 for p in validated_predictions if p["validation"] == "TP")
        tn = sum(1 for p in validated_predictions if p["validation"] == "TN")
        fp = sum(1 for p in validated_predictions if p["validation"] == "FP")
        fn = sum(1 for p in validated_predictions if p["validation"] == "FN")
        total = len(validated_predictions)

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_dict["quality_gate"] = {
            "predictions": validated_predictions,
            "confusion_matrix": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        }

    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"Reporte de métricas guardado en {args.metrics_output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid SMOTE + LLM augmentation for MBTI")
    parser.add_argument("--data-path", default="mbti_1.csv")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="auto", help="Dispositivo para embeddings: auto|cpu|cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="Seed para el split train/test (si no se pasa, usa random-seed)")
    parser.add_argument("--gen-seed", type=int, default=None, help="Seed para generación/clustering (si no se pasa, usa random-seed)")
    parser.add_argument("--train-csv", default=None, help="CSV de train predefinido (columnas: label,text)")
    parser.add_argument("--test-csv", default=None, help="CSV de test predefinido (columnas: label,text)")
    parser.add_argument("--class-config", default=None, help="JSON/YAML con overrides por clase (keys = nombres de clase)")

    parser.add_argument("--only-baseline", action="store_true", help="Solo evalúa sin generar datos")
    parser.add_argument("--minority-percentile", type=float, default=0.5)
    parser.add_argument("--target-classes", nargs="*", help="Lista de clases MBTI a generar")

    parser.add_argument("--outlier-k", type=int, default=5)
    parser.add_argument("--outlier-keep-quantile", type=float, default=0.85)
    parser.add_argument("--cluster-size", type=int, default=60)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--density-k", type=int, default=5)
    parser.add_argument("--density-top-quantile", type=float, default=0.3)
    parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--prompts-per-cluster", type=int, default=3)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--top-keywords", type=int, default=20)
    parser.add_argument("--knn-support", type=int, default=0, help="Número de vecinos K-NN del punto sintético para contexto (0=automático: min(10, len(cluster)//5))")

    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prompt-mode", choices=["mix", "paraphrase"], default="mix")
    parser.add_argument("--max-prompt-tokens", type=int, default=60)
    parser.add_argument("--language", choices=["auto", "en", "es"], default="auto")
    parser.add_argument("--anchor-selection-method",
                        choices=["centroid", "medoid", "quality_gated", "diverse", "ensemble", "llm_recommender"],
                        default="centroid",
                        help="Method for selecting cluster anchor points for SMOTE generation")
    parser.add_argument("--use-class-description", action="store_true",
                        help="Add MBTI type description to prompts for semantic context enhancement")

    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--similarity-floor-quantile", type=float, default=0.25)
    parser.add_argument("--similarity-margin", type=float, default=0.08)
    parser.add_argument("--similarity-to-anchor", type=float, default=0.5, help="Umbral de similitud respecto a ejemplo ancla/vecino para aceptación")
    parser.add_argument("--min-classifier-confidence", type=float, default=0.6)
    parser.add_argument("--class-conf-quantile", type=float, default=0.2)
    parser.add_argument("--high-conf-threshold", type=float, default=0.8)
    parser.add_argument("--high-conf-sim-bonus", type=float, default=0.1)
    parser.add_argument("--min-tokens", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--duplicate-ngram", type=int, default=3)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--dedup-embed-sim", type=float, default=0.95, help="Umbral de similitud (coseno) para deduplicación semántica contra referencias")
    parser.add_argument("--clf-margin", type=float, default=0.10, help="Margen mínimo p(max)-p(2do) para aceptar por clasificador")
    parser.add_argument("--repel-nontarget-sim", type=float, default=0.60, help="Si similitud con centroide de otra clase ≥ umbral, rechazar candidato")
    parser.add_argument("--filter-mode", choices=["classifier", "knn", "hybrid"], default="hybrid")
    parser.add_argument("--filter-knn-k", type=int, default=15)
    parser.add_argument("--filter-knn-min-sim", type=float, default=0.45)

    # Anchor Quality Improvements (Phase 1-3)
    parser.add_argument("--enable-anchor-gate", action="store_true", help="Fase 1: Gate de calidad para anchors antes de generar")
    parser.add_argument("--anchor-quality-threshold", type=float, default=0.6, help="Umbral de calidad mínima para generar (0-1)")
    parser.add_argument("--anchor-cohesion-weight", type=float, default=0.4, help="Peso de cohesión en gate")
    parser.add_argument("--anchor-purity-weight", type=float, default=0.4, help="Peso de pureza K-NN en gate")
    parser.add_argument("--anchor-separation-weight", type=float, default=0.2, help="Peso de separación en gate")

    parser.add_argument("--enable-anchor-selection", action="store_true", help="Fase 2: Selección de mejores anchors (top-k centrales)")
    parser.add_argument("--anchor-selection-ratio", type=float, default=0.7, help="Ratio de anchors a seleccionar (0.7 = top-70%%)")
    parser.add_argument("--anchor-outlier-threshold", type=float, default=1.5, help="Umbral IQR para detección de outliers")
    parser.add_argument("--anchor-min-samples", type=int, default=3, help="Mínimo de samples tras selección")

    parser.add_argument("--enable-adaptive-filters", action="store_true", help="Fase 1b: Relajar filtros dinámicamente según calidad de anchors")

    # PHASE B: Adaptive weighting
    parser.add_argument("--enable-adaptive-weighting", action="store_true", help="Experiment 1 (Phase B): Usar pesos adaptativos por clase basados en baseline F1")

    parser.add_argument("--synthetic-weight", type=float, default=0.5)
    parser.add_argument("--synthetic-weight-mode", choices=["flat", "confidence"], default="flat", help="Modo de ponderación de sintéticos en el entrenamiento")
    parser.add_argument("--synthetic-output", default="synthetic_mbti.csv")
    parser.add_argument("--augmented-train-output", default="mbti_train_augmented.csv")
    parser.add_argument("--metrics-output", default="augmentation_metrics.json")
    parser.add_argument("--debug-filter", action="store_true")
    parser.add_argument("--balanced-baseline", action="store_true", help="Usa class_weight=balanced en el baseline y modelo aumentado")
    # Grace acceptance options to bootstrap when filters are too strict
    parser.add_argument("--grace-topk", type=int, default=1)
    parser.add_argument("--grace-min-anchor", type=float, default=0.35)
    parser.add_argument("--grace-min-score", type=float, default=0.45)
    parser.add_argument("--grace-min-conf", type=float, default=0.35)
    parser.add_argument("--grace-disable-after", type=int, default=50, help="Desactiva grace tras acumular N sintéticos por clase (<=0 desactiva)")
    # Quotas to avoid concentration
    parser.add_argument("--cap-class-ratio", type=float, default=0.0, help="Máx. ratio sintéticos/clase respecto a reales (0=sin límite)")
    parser.add_argument("--cap-class-abs", type=int, default=0, help="Máx. absoluto de sintéticos por clase (0=sin límite)")
    parser.add_argument("--cap-cluster-ratio", type=float, default=1.0, help="Máx. ratio sintéticos/cluster respecto a tamaño real del cluster (<=0 sin límite)")

    # TIER S Improvements
    parser.add_argument("--use-discriminator", action="store_true", help="TIER S: Enable adversarial discriminator filtering")
    parser.add_argument("--discriminator-threshold", type=float, default=0.7, help="TIER S: Discriminator acceptance threshold (prob < threshold = accept)")
    parser.add_argument("--discriminator-model", choices=["logistic", "random_forest"], default="logistic", help="TIER S: Discriminator model type")
    parser.add_argument("--discriminator-mode", choices=["filter", "weight"], default="filter", help="Phase 3: Discriminator mode - filter (binary) or weight (continuous)")
    parser.add_argument("--use-ensemble", action="store_true", help="TIER S: Enable multi-seed ensemble")
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42, 101, 102], help="TIER S: Seeds for ensemble (list of integers)")
    parser.add_argument("--ensemble-voting", choices=["soft", "hard"], default="soft", help="TIER S: Ensemble voting method")

    # Phase 3: Robustness Improvements
    parser.add_argument("--use-val-gating", action="store_true", help="Phase 3: Enable validation-based gating per class")
    parser.add_argument("--val-size", type=float, default=0.2, help="Phase 3: Validation set size (fraction of train)")
    parser.add_argument("--val-tolerance", type=float, default=0.01, help="Phase 3: Tolerance for val degradation (reject if delta < -tolerance)")
    parser.add_argument("--use-contamination-control", action="store_true", help="Phase 3: Enable contamination-aware budget control")
    parser.add_argument("--contamination-threshold", type=float, default=0.005, help="Phase 3: Contamination threshold (0.005 = 0.5%%)")
    parser.add_argument("--contamination-multipliers", nargs=3, type=float, default=[0.3, 0.7, 1.0], help="Phase 3: Budget multipliers for [high, medium, low] contamination")
    parser.add_argument("--use-f1-budget-scaling", action="store_true", help="Phase 3: Enable F1-based budget scaling (skip high-F1 classes)")
    parser.add_argument("--f1-budget-thresholds", nargs=2, type=float, default=[0.35, 0.20], help="Phase 3: F1 thresholds for [high, medium] (format: high_threshold medium_threshold)")
    parser.add_argument("--f1-budget-multipliers", nargs=3, type=float, default=[0.0, 0.5, 1.0], help="Phase 3: Budget multipliers for [high, medium, low] F1 classes")

    # Batch 3: Strategy improvements
    parser.add_argument("--use-ensemble-selection", action="store_true", help="Batch 3 Strategy 1: Enable ensemble selection (choose best prediction per class)")
    parser.add_argument("--adaptive-weights", nargs=3, type=float, default=[0.0, 0.3, 0.7], help="Batch 3 Strategy 3: Adaptive synthetic weights for [high, medium, low] F1 tiers")
    parser.add_argument("--use-class-balanced-augmentation", action="store_true", help="Batch 3 Strategy 5: Balance synthetic generation across classes")
    parser.add_argument("--balance-target-ratio", type=float, default=0.3, help="Batch 3 Strategy 5: Target ratio of synthetics per class (default: 0.3)")

    return parser


def run_ensemble_pipeline(args: argparse.Namespace) -> None:
    """
    TIER S: Run pipeline with multiple seeds and ensemble predictions.

    This is a simplified ensemble that runs the full pipeline multiple times
    with different random seeds and then ensembles the final models.
    """
    if not MULTI_SEED_ENSEMBLE_AVAILABLE:
        print("ERROR: multi_seed_ensemble module not available. Cannot run ensemble mode.")
        print("Falling back to single-seed mode...")
        orchestrate_pipeline(args)
        return

    print("="*70)
    print("TIER S: Multi-Seed Ensemble Mode")
    print(f"Seeds: {args.ensemble_seeds}")
    print(f"Voting: {args.ensemble_voting}")
    print("="*70)
    print()

    # For ensemble mode, we run the pipeline multiple times with different seeds
    # Each run trains a complete model

    for seed_idx, seed in enumerate(args.ensemble_seeds):
        print(f"\n{'='*70}")
        print(f"Ensemble Seed {seed_idx + 1}/{len(args.ensemble_seeds)}: {seed}")
        print(f"{'='*70}\n")

        # Create a copy of args with this seed
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_seed = seed
        seed_args.gen_seed = seed
        seed_args.split_seed = seed  # Keep same split for fair comparison

        # Disable ensemble mode for individual runs
        seed_args.use_ensemble = False

        # Run pipeline for this seed
        orchestrate_pipeline(seed_args)

    print("\n" + "="*70)
    print("TIER S Ensemble: All seeds completed")
    print("="*70)
    print("\nNOTE: Each seed ran independently and saved its own metrics.")
    print("For full soft/hard voting ensemble, results should be combined post-hoc.")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # Check if ensemble mode is enabled
    if args.use_ensemble:
        run_ensemble_pipeline(args)
    else:
        orchestrate_pipeline(args)
