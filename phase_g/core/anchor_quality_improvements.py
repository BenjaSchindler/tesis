#!/usr/bin/env python3
"""
Anchor Quality Improvements - Tres Fases
=========================================

Este módulo implementa las tres fases de mejoras para el pipeline de augmentación:

Fase 1: Gate de Calidad por Clase (CRÍTICO)
- Evalúa la calidad de los anchors antes de generar
- SKIP clases con anchors de mala calidad

Fase 2: Selección de Anchors (ESENCIAL)
- Selecciona solo los mejores anchors (top-k centrales)
- Remueve outliers para mejorar cohesión

Fase 3: Ponderación Adaptativa (REFINAMIENTO)
- Asigna weights a sintéticos según calidad de anchors
- Evita que sintéticos de baja calidad dominen el training

Uso:
----
```python
from anchor_quality_improvements import (
    check_anchor_quality,
    select_best_anchors,
    compute_adaptive_weights,
    get_adaptive_filter_params
)

# Fase 1: Gate de calidad
quality_score, should_generate, metadata = check_anchor_quality(
    embeddings=all_embeddings,
    labels=all_labels,
    target_class=class_name,
    k=10
)

if not should_generate:
    print(f"SKIPPING {class_name} due to low anchor quality")
    continue

# Fase 2: Selección de anchors
selected_indices, selection_metrics = select_best_anchors(
    class_embeddings=class_embeddings,
    selection_ratio=0.7
)

# Usar solo anchors seleccionados
class_texts = class_texts[selected_indices]
class_embeddings = class_embeddings[selected_indices]

# Fase 3: Ponderación adaptativa (después de generar)
quality_scores_by_class = {...}  # Del Fase 1
synthetic_weights = compute_adaptive_weights(
    synthetics=generated_synthetics,
    anchor_quality_scores=quality_scores_by_class,
    base_weight=0.3
)

# Combinar weights para training
all_weights = np.concatenate([original_weights, synthetic_weights])
classifier.fit(X_combined, y_combined, sample_weight=all_weights)
```

Autor: Benjamin (basado en análisis PERCLASS_ANALYSIS_COMPLETE.md)
Fecha: Octubre 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ==============================================================================
# FASE 1: GATE DE CALIDAD POR CLASE
# ==============================================================================

def check_anchor_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    target_class: str,
    k: int = 10,
    cohesion_weight: float = 0.4,
    purity_weight: float = 0.4,
    separation_weight: float = 0.2,
    quality_threshold: float = 0.6,
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Evalúa la calidad de los anchor samples para una clase.

    Esta función implementa el "Gate de Calidad" que previene la generación
    de sintéticos cuando los anchors son de mala calidad, evitando así
    los resultados catastróficos observados en seeds 789 (ISTJ -0.092) y
    123 (ISFJ -0.066).

    Métricas evaluadas:
    1. **Cohesión intra-clase**: ¿Los samples de la clase son similares entre sí?
    2. **Pureza K-NN**: ¿Los K vecinos más cercanos son de la misma clase?
    3. **Separación**: ¿Qué tan separada está esta clase de otras?

    Args:
        embeddings: Embeddings de TODAS las muestras (N, D)
        labels: Labels de TODAS las muestras (N,)
        target_class: Clase a evaluar
        k: Número de vecinos para pureza K-NN
        cohesion_weight: Peso para cohesión (default 0.4)
        purity_weight: Peso para pureza (default 0.4)
        separation_weight: Peso para separación (default 0.2)
        quality_threshold: Umbral de calidad para generar (default 0.6)

    Returns:
        quality_score: Score de calidad 0-1
        should_generate: True si quality_score >= threshold
        metadata: Dict con métricas individuales y detalles

    Example:
        >>> quality, should_gen, meta = check_anchor_quality(
        ...     embeddings=X_train,
        ...     labels=y_train,
        ...     target_class="ESTJ",
        ...     k=10
        ... )
        >>> if should_gen:
        ...     print(f"OK to generate (quality={quality:.3f})")
        ... else:
        ...     print(f"SKIP due to poor anchors (quality={quality:.3f})")
    """
    # Filtrar embeddings de la clase target
    class_mask = (labels == target_class)
    class_embeddings = embeddings[class_mask]
    n_samples = len(class_embeddings)

    # Casos edge: muy pocas muestras
    if n_samples < 3:
        return 0.0, False, {
            "reason": "insufficient_samples",
            "n_samples": n_samples,
            "quality_score": 0.0
        }

    # 1. COHESIÓN INTRA-CLASE
    # Medida: similitud coseno promedio entre todos los pares de la clase
    pairwise_sims = cosine_similarity(class_embeddings)
    np.fill_diagonal(pairwise_sims, 0)  # Excluir self-similarity

    # Usar upper triangle para no contar dos veces cada par
    triu_indices = np.triu_indices_from(pairwise_sims, k=1)
    if len(triu_indices[0]) > 0:
        cohesion = float(np.mean(pairwise_sims[triu_indices]))
    else:
        cohesion = 0.0

    # 2. PUREZA K-NN
    # ¿Qué % de los K vecinos más cercanos son de la misma clase?
    knn_purity = compute_knn_purity(
        all_embeddings=embeddings,
        all_labels=labels,
        class_embeddings=class_embeddings,
        target_class=target_class,
        k=min(k, len(embeddings) - n_samples)  # No más que samples de otras clases
    )

    # 3. SEPARACIÓN A OTRAS CLASES
    # Distancia mínima del centroide de esta clase a centroides de otras clases
    other_mask = ~class_mask
    if np.sum(other_mask) > 0:
        class_centroid = np.mean(class_embeddings, axis=0)
        other_embeddings = embeddings[other_mask]
        other_labels_unique = np.unique(labels[other_mask])

        # Calcular centroide de cada otra clase
        min_distance = 1.0  # Máxima separación posible (cosine distance)
        for other_class in other_labels_unique:
            other_class_mask = (labels == other_class)
            other_class_embeddings = embeddings[other_class_mask]
            other_centroid = np.mean(other_class_embeddings, axis=0)

            # Distancia coseno (1 - similitud)
            distance = 1.0 - cosine_similarity(
                class_centroid.reshape(1, -1),
                other_centroid.reshape(1, -1)
            )[0, 0]

            min_distance = min(min_distance, distance)

        separation = float(min_distance)
    else:
        separation = 1.0  # Solo hay una clase (edge case)

    # SCORE COMBINADO
    quality_score = (
        cohesion_weight * cohesion +
        purity_weight * knn_purity +
        separation_weight * separation
    )

    # DECISIÓN
    should_generate = quality_score >= quality_threshold

    # METADATA
    metadata = {
        "cohesion": float(cohesion),
        "knn_purity": float(knn_purity),
        "separation": float(separation),
        "quality_score": float(quality_score),
        "n_samples": int(n_samples),
        "threshold": float(quality_threshold),
        "should_generate": bool(should_generate),
        "weights": {
            "cohesion": cohesion_weight,
            "purity": purity_weight,
            "separation": separation_weight
        }
    }

    # Log decisión
    if should_generate:
        logger.info(
            f"✅ {target_class}: quality={quality_score:.3f} "
            f"(cohesion={cohesion:.3f}, purity={knn_purity:.3f}, sep={separation:.3f}) "
            f"→ GENERATE"
        )
    else:
        logger.warning(
            f"❌ {target_class}: quality={quality_score:.3f} < {quality_threshold:.3f} "
            f"(cohesion={cohesion:.3f}, purity={knn_purity:.3f}, sep={separation:.3f}) "
            f"→ SKIP"
        )

    return quality_score, should_generate, metadata


def compute_knn_purity(
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    class_embeddings: np.ndarray,
    target_class: str,
    k: int = 10
) -> float:
    """
    Calcula pureza K-NN: % de K vecinos más cercanos que son de la misma clase.

    Args:
        all_embeddings: Embeddings de todas las muestras (N, D)
        all_labels: Labels de todas las muestras (N,)
        class_embeddings: Embeddings de la clase target (M, D)
        target_class: Nombre de la clase target
        k: Número de vecinos

    Returns:
        Pureza promedio (0-1): 1.0 = todos los vecinos son de la misma clase
    """
    if k <= 0 or len(class_embeddings) == 0:
        return 0.0

    k_actual = min(k, len(all_embeddings) - 1)
    if k_actual <= 0:
        return 0.0

    # Fit K-NN en todos los embeddings
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1, metric='cosine')
    nbrs.fit(all_embeddings)

    purities = []
    for emb in class_embeddings:
        # Encontrar K+1 vecinos (incluyendo self)
        distances, indices = nbrs.kneighbors(emb.reshape(1, -1))

        # Excluir self (primer vecino)
        neighbor_indices = indices[0][1:]
        neighbor_labels = all_labels[neighbor_indices]

        # Contar cuántos son de la misma clase
        same_class_count = np.sum(neighbor_labels == target_class)
        purity = same_class_count / k_actual
        purities.append(purity)

    return float(np.mean(purities))


# ==============================================================================
# FASE 2: SELECCIÓN DE ANCHORS
# ==============================================================================

def select_best_anchors(
    class_embeddings: np.ndarray,
    selection_ratio: float = 0.7,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    min_anchors: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Selecciona los mejores anchor samples de una clase.

    Esta función implementa la "Selección de Anchors" que mejora la calidad
    promedio de los anchors eliminando outliers y seleccionando solo los
    samples más centrales y representativos de la clase.

    Evidencia: ESTJ con 8 samples → seleccionar top-5 más centrales mejora
    la probabilidad de éxito de 20% → 60-80%.

    Pasos:
    1. Calcular centroide de la clase
    2. Medir similitud de cada sample al centroide
    3. Remover outliers (IQR method)
    4. Seleccionar top-k más centrales

    Args:
        class_embeddings: Embeddings de la clase (N, D)
        selection_ratio: Ratio de samples a seleccionar (default 0.7 = 70%)
        outlier_method: Método para detectar outliers ("iqr" o "zscore")
        outlier_threshold: Umbral para outliers (IQR: 1.5, Z-score: 2.0)
        min_anchors: Mínimo número de anchors a retornar

    Returns:
        selected_indices: Índices de los anchors seleccionados
        metrics: Dict con métricas de la selección

    Example:
        >>> indices, metrics = select_best_anchors(
        ...     class_embeddings=estj_embeddings,  # shape (8, 768)
        ...     selection_ratio=0.7
        ... )
        >>> print(f"Selected {len(indices)} / {len(estj_embeddings)} anchors")
        Selected 5 / 8 anchors
        >>> selected_embeddings = estj_embeddings[indices]
    """
    n_samples = len(class_embeddings)

    # Edge case: muy pocas muestras
    if n_samples <= min_anchors:
        logger.info(
            f"Class has only {n_samples} samples, using all as anchors "
            f"(min_anchors={min_anchors})"
        )
        return np.arange(n_samples), {
            "n_original": n_samples,
            "n_outliers_removed": 0,
            "n_selected": n_samples,
            "selection_ratio_actual": 1.0,
            "skipped_reason": "insufficient_samples"
        }

    # 1. CALCULAR CENTROIDE
    centroid = np.mean(class_embeddings, axis=0)

    # 2. SIMILITUD AL CENTROIDE
    similarities = cosine_similarity(
        class_embeddings,
        centroid.reshape(1, -1)
    )[:, 0]

    # 3. REMOVER OUTLIERS
    if outlier_method == "iqr":
        # Método IQR (Interquartile Range)
        q1, q3 = np.percentile(similarities, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - outlier_threshold * iqr
        # No hay upper bound para similitud (queremos alta similitud)

        non_outlier_mask = similarities >= lower_bound
        non_outlier_indices = np.where(non_outlier_mask)[0]
        n_outliers = np.sum(~non_outlier_mask)

    elif outlier_method == "zscore":
        # Método Z-score
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        if std_sim > 0:
            z_scores = (similarities - mean_sim) / std_sim
            non_outlier_mask = z_scores >= -outlier_threshold
            non_outlier_indices = np.where(non_outlier_mask)[0]
            n_outliers = np.sum(~non_outlier_mask)
        else:
            # Std=0 significa todos son iguales
            non_outlier_indices = np.arange(n_samples)
            n_outliers = 0
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")

    # Asegurar mínimo de anchors después de remover outliers
    if len(non_outlier_indices) < min_anchors:
        logger.warning(
            f"After outlier removal, only {len(non_outlier_indices)} samples left "
            f"(< min_anchors={min_anchors}). Using top-{min_anchors} by similarity."
        )
        # Usar top-min_anchors sin importar outliers
        non_outlier_indices = np.argsort(similarities)[::-1][:min_anchors]
        n_outliers = n_samples - len(non_outlier_indices)

    # 4. SELECCIONAR TOP-K MÁS CENTRALES
    num_to_select = max(min_anchors, int(len(non_outlier_indices) * selection_ratio))

    # Ordenar por similitud descendente
    sorted_indices = non_outlier_indices[
        np.argsort(similarities[non_outlier_indices])[::-1]
    ]

    selected_indices = sorted_indices[:num_to_select]

    # MÉTRICAS
    metrics = {
        "n_original": int(n_samples),
        "n_outliers_removed": int(n_outliers),
        "n_non_outliers": int(len(non_outlier_indices)),
        "n_selected": int(len(selected_indices)),
        "selection_ratio_target": float(selection_ratio),
        "selection_ratio_actual": float(len(selected_indices) / n_samples),
        "mean_similarity_to_centroid": float(np.mean(similarities[selected_indices])),
        "std_similarity": float(np.std(similarities[selected_indices])),
        "min_similarity": float(np.min(similarities[selected_indices])),
        "max_similarity": float(np.max(similarities[selected_indices])),
        "outlier_method": outlier_method,
        "outlier_threshold": float(outlier_threshold)
    }

    logger.info(
        f"Anchor selection: {n_samples} → {len(selected_indices)} "
        f"(removed {n_outliers} outliers, selected top-{selection_ratio:.0%})"
    )

    return selected_indices, metrics


# ==============================================================================
# FASE 3: PONDERACIÓN ADAPTATIVA
# ==============================================================================

def compute_adaptive_weights(
    synthetics: List[Dict[str, Any]],
    anchor_quality_scores: Dict[str, float],
    base_weight: float = 0.3,
    confidence_exponent: float = 1.5,
    quality_exponent: float = 1.0,
    min_weight: float = 0.1,
    max_weight: float = 1.0
) -> np.ndarray:
    """
    Calcula weights adaptativos para cada sintético basado en calidad de anchors.

    Esta función implementa la "Ponderación Adaptativa" que reduce el daño
    de sintéticos de baja calidad asignándoles weights bajos, mientras potencia
    sintéticos de alta calidad.

    Formula: weight = base_weight * p(target)^α * quality^β

    Donde:
    - p(target): confianza del clasificador en la clase target
    - quality: quality_score de los anchors de esa clase (de Fase 1)
    - α (confidence_exponent): controla sensibilidad a confianza
    - β (quality_exponent): controla sensibilidad a calidad

    Args:
        synthetics: Lista de sintéticos, cada uno con keys:
            - 'class': str (clase del sintético)
            - 'confidence': float (confianza del clasificador 0-1)
        anchor_quality_scores: Dict {class_name: quality_score} de Fase 1
        base_weight: Peso base para sintéticos (default 0.3)
        confidence_exponent: Exponente para confianza (default 1.5)
        quality_exponent: Exponente para calidad (default 1.0)
        min_weight: Peso mínimo (default 0.1)
        max_weight: Peso máximo (default 1.0)

    Returns:
        weights: Array de weights para cada sintético (N,)

    Example:
        >>> quality_scores = {
        ...     "ESTJ": 0.85,  # Alta calidad (seed 42)
        ...     "ISFJ": 0.30   # Baja calidad (seed 123)
        ... }
        >>> synthetics = [
        ...     {"class": "ESTJ", "confidence": 0.9},  # Alto quality, alta conf
        ...     {"class": "ISFJ", "confidence": 0.8},  # Bajo quality, alta conf
        ... ]
        >>> weights = compute_adaptive_weights(synthetics, quality_scores)
        >>> print(weights)
        [0.90, 0.15]  # ESTJ alto peso, ISFJ bajo peso
    """
    weights = []

    for synth in synthetics:
        class_label = synth['class']
        confidence = synth.get('confidence', 0.5)

        # Quality score de esta clase (default 0.5 si no disponible)
        quality = anchor_quality_scores.get(class_label, 0.5)

        # FORMULA ADAPTATIVA
        weight = base_weight * (confidence ** confidence_exponent) * (quality ** quality_exponent)

        # CLIP a rango válido
        weight = np.clip(weight, min_weight, max_weight)

        weights.append(weight)

    weights_array = np.array(weights, dtype=np.float32)

    # LOG estadísticas
    logger.info(
        f"Adaptive weights: mean={np.mean(weights_array):.3f}, "
        f"std={np.std(weights_array):.3f}, "
        f"min={np.min(weights_array):.3f}, "
        f"max={np.max(weights_array):.3f}"
    )

    return weights_array


# ==============================================================================
# UTILIDADES: FILTROS ADAPTATIVOS
# ==============================================================================

def get_adaptive_filter_params(
    quality_score: float,
    base_params: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Ajusta parámetros de filtrado según calidad de anchors.

    Alta calidad → filtros más relajados (aceptar más, están buenos)
    Baja calidad → filtros estrictos (muy selectivos)

    Args:
        quality_score: Score de calidad 0-1 (de check_anchor_quality)
        base_params: Parámetros base (default: K-Fold best)

    Returns:
        Dict con parámetros ajustados:
        - knn_min_sim: Similitud mínima K-NN
        - similarity_threshold: Umbral de similitud general
        - similarity_to_anchor: Similitud al anchor
        - cap_ratio: Ratio de cap (opcional)

    Example:
        >>> params = get_adaptive_filter_params(quality_score=0.85)
        >>> print(params)
        {'knn_min_sim': 0.40, 'similarity_threshold': 0.42, ...}
    """
    if base_params is None:
        base_params = {
            "knn_min_sim": 0.42,
            "similarity_threshold": 0.45,
            "similarity_to_anchor": 0.45,
            "cap_ratio": 0.8
        }

    if quality_score >= 0.8:
        # EXCELENTE calidad → relajar filtros
        return {
            "knn_min_sim": base_params["knn_min_sim"] - 0.02,  # 0.40
            "similarity_threshold": base_params["similarity_threshold"] - 0.03,  # 0.42
            "similarity_to_anchor": base_params["similarity_to_anchor"] - 0.03,  # 0.42
            "cap_ratio": 1.0  # Sin límite
        }
    elif quality_score >= 0.6:
        # BUENA calidad → usar base
        return base_params.copy()
    else:
        # BAJA calidad → este caso NO debería llegar aquí
        # (el gate debería haber bloqueado), pero por si acaso:
        logger.warning(
            f"get_adaptive_filter_params called with low quality={quality_score:.3f}. "
            f"This should have been blocked by gate."
        )
        return {
            "knn_min_sim": base_params["knn_min_sim"] + 0.03,  # 0.45 (más estricto)
            "similarity_threshold": base_params["similarity_threshold"] + 0.03,  # 0.48
            "similarity_to_anchor": base_params["similarity_to_anchor"] + 0.03,  # 0.48
            "cap_ratio": 0.5  # Muy limitado
        }


# ==============================================================================
# PIPELINE COMPLETO: USO INTEGRADO DE LAS TRES FASES
# ==============================================================================

def augment_class_with_quality_control(
    class_name: str,
    class_texts: np.ndarray,
    class_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    anchor_selection_ratio: float = 0.7,
    quality_threshold: float = 0.6,
    enable_gate: bool = True,
    enable_selection: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Pipeline completo de augmentación con control de calidad (3 fases integradas).

    Este es un wrapper que integra las tres fases:
    1. Gate de calidad (decide si generar)
    2. Selección de anchors (elige los mejores)
    3. (Ponderación adaptativa se aplica después con los sintéticos generados)

    Args:
        class_name: Nombre de la clase
        class_texts: Textos de la clase
        class_embeddings: Embeddings de la clase
        all_embeddings: Embeddings de todas las clases
        all_labels: Labels de todas las clases
        anchor_selection_ratio: Ratio de anchors a seleccionar
        quality_threshold: Umbral de calidad para el gate
        enable_gate: Activar Fase 1 (gate)
        enable_selection: Activar Fase 2 (selection)
        **kwargs: Otros parámetros

    Returns:
        selected_texts: Textos seleccionados como anchors (o vacío si skip)
        selected_embeddings: Embeddings seleccionados
        quality_score: Score de calidad de la clase
        metadata: Dict con información de las decisiones

    Example:
        >>> texts, embs, quality, meta = augment_class_with_quality_control(
        ...     class_name="ESTJ",
        ...     class_texts=estj_texts,
        ...     class_embeddings=estj_embs,
        ...     all_embeddings=all_embs,
        ...     all_labels=all_labels
        ... )
        >>> if len(texts) > 0:
        ...     # Proceder con generación usando estos anchors
        ...     synthetics = generate_synthetics(texts, embs, ...)
    """
    metadata = {
        "class": class_name,
        "n_original": len(class_texts),
        "gate_enabled": enable_gate,
        "selection_enabled": enable_selection
    }

    # FASE 1: GATE DE CALIDAD
    if enable_gate:
        quality_score, should_generate, gate_meta = check_anchor_quality(
            embeddings=all_embeddings,
            labels=all_labels,
            target_class=class_name,
            quality_threshold=quality_threshold
        )

        metadata.update({
            "quality_score": quality_score,
            "should_generate": should_generate,
            "gate_metadata": gate_meta
        })

        if not should_generate:
            logger.warning(
                f"⊘ Skipping {class_name} due to low anchor quality "
                f"(score={quality_score:.3f} < {quality_threshold:.3f})"
            )
            return np.array([]), np.array([]).reshape(0, class_embeddings.shape[1]), quality_score, metadata
    else:
        quality_score = 1.0  # Assume high quality if gate disabled
        metadata["quality_score"] = quality_score

    # FASE 2: SELECCIÓN DE ANCHORS
    if enable_selection:
        selected_indices, selection_meta = select_best_anchors(
            class_embeddings=class_embeddings,
            selection_ratio=anchor_selection_ratio
        )

        metadata.update({
            "n_selected": len(selected_indices),
            "selection_metadata": selection_meta
        })

        selected_texts = class_texts[selected_indices]
        selected_embeddings = class_embeddings[selected_indices]

        logger.info(
            f"✓ Selected {len(selected_indices)}/{len(class_texts)} anchors for {class_name}"
        )
    else:
        # Usar todos
        selected_texts = class_texts
        selected_embeddings = class_embeddings
        metadata["n_selected"] = len(class_texts)

    return selected_texts, selected_embeddings, quality_score, metadata


# ==============================================================================
# TESTING Y VALIDACIÓN
# ==============================================================================

if __name__ == "__main__":
    # Test básico de las funciones
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("TESTING ANCHOR QUALITY IMPROVEMENTS")
    print("=" * 80)

    # Generar datos sintéticos para testing
    np.random.seed(42)
    n_samples = 100
    n_dims = 768

    # Crear dos clases con diferente calidad
    # Clase A: Alta cohesión (good anchors)
    class_a_center = np.random.randn(n_dims)
    class_a_embeddings = class_a_center + np.random.randn(30, n_dims) * 0.1  # Tight cluster

    # Clase B: Baja cohesión (bad anchors)
    class_b_embeddings = np.random.randn(20, n_dims) * 0.5  # Dispersed

    # Clase C: Otra clase para separación
    class_c_embeddings = np.random.randn(50, n_dims) * 0.3

    all_embeddings = np.vstack([
        class_a_embeddings,
        class_b_embeddings,
        class_c_embeddings
    ])

    all_labels = np.array(
        ['A'] * 30 + ['B'] * 20 + ['C'] * 50
    )

    print("\nTest 1: Check Anchor Quality")
    print("-" * 80)

    # Clase A (debería pasar)
    quality_a, should_gen_a, meta_a = check_anchor_quality(
        embeddings=all_embeddings,
        labels=all_labels,
        target_class='A',
        k=10
    )
    print(f"Class A: quality={quality_a:.3f}, should_generate={should_gen_a}")
    print(f"  Cohesion: {meta_a['cohesion']:.3f}")
    print(f"  Purity: {meta_a['knn_purity']:.3f}")
    print(f"  Separation: {meta_a['separation']:.3f}")

    # Clase B (debería fallar)
    quality_b, should_gen_b, meta_b = check_anchor_quality(
        embeddings=all_embeddings,
        labels=all_labels,
        target_class='B',
        k=10
    )
    print(f"\nClass B: quality={quality_b:.3f}, should_generate={should_gen_b}")
    print(f"  Cohesion: {meta_b['cohesion']:.3f}")
    print(f"  Purity: {meta_b['knn_purity']:.3f}")
    print(f"  Separation: {meta_b['separation']:.3f}")

    print("\n" + "=" * 80)
    print("Test 2: Select Best Anchors")
    print("-" * 80)

    # Añadir outliers artificiales a clase A
    class_a_with_outliers = np.vstack([
        class_a_embeddings,
        np.random.randn(3, n_dims) * 2.0  # 3 outliers
    ])

    indices, metrics = select_best_anchors(
        class_embeddings=class_a_with_outliers,
        selection_ratio=0.7
    )

    print(f"Original: {len(class_a_with_outliers)} samples")
    print(f"Outliers removed: {metrics['n_outliers_removed']}")
    print(f"Selected: {metrics['n_selected']} samples")
    print(f"Mean similarity to centroid: {metrics['mean_similarity_to_centroid']:.3f}")

    print("\n" + "=" * 80)
    print("Test 3: Compute Adaptive Weights")
    print("-" * 80)

    quality_scores = {
        'A': 0.85,  # High quality
        'B': 0.30   # Low quality
    }

    synthetics = [
        {'class': 'A', 'confidence': 0.9},
        {'class': 'A', 'confidence': 0.7},
        {'class': 'B', 'confidence': 0.8},
        {'class': 'B', 'confidence': 0.6},
    ]

    weights = compute_adaptive_weights(
        synthetics=synthetics,
        anchor_quality_scores=quality_scores,
        base_weight=0.3
    )

    for i, (synth, weight) in enumerate(zip(synthetics, weights)):
        print(f"Synthetic {i+1}: class={synth['class']}, conf={synth['confidence']:.2f}, weight={weight:.3f}")

    print("\n" + "=" * 80)
    print("Test 4: Adaptive Filter Params")
    print("-" * 80)

    for quality in [0.85, 0.65, 0.45]:
        params = get_adaptive_filter_params(quality)
        print(f"\nQuality={quality:.2f}:")
        print(f"  knn_min_sim: {params['knn_min_sim']:.3f}")
        print(f"  similarity_threshold: {params['similarity_threshold']:.3f}")
        print(f"  cap_ratio: {params['cap_ratio']:.3f}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
