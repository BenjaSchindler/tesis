"""Input-only acquisition policies for the production-evaluation pilot.

Only numpy and the Python standard library are required. Callers must supply a
construction-only pool: this module cannot detect reference leakage or whether
risk signals were measured before the evaluation outputs being predicted.

``select(policy, ids, budget, ...)`` returns original-row indices and normalized
weights in the same order. Probability policies also return first-order
inclusion probabilities for *every pool row*. The deterministic policies return
None, because representative weights are not sampling probabilities and do not
justify design-based confidence intervals. No uncertainty intervals are emitted.

IDs must be unique strings. All ties and RNG sampling operate in sorted ID order,
so reordering the pool does not change the selected IDs or their weights. Strata
must be predeclared strings; a budget smaller than the number of nonempty strata
is rejected when the policy needs to represent each stratum.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import re
from typing import Any, Sequence

import numpy as np


POLICIES = (
    "uniform_random", "stratified_random", "centroid_proximity",
    "facility_location", "minhash_farthest_first", "random_plus_targeted",
)


@dataclass(frozen=True)
class SelectionResult:
    indices: np.ndarray
    sample_weights: np.ndarray
    inclusion_probabilities: np.ndarray | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": self.indices.tolist(),
            "sample_weights": self.sample_weights.tolist(),
            "inclusion_probabilities": (
                None if self.inclusion_probabilities is None
                else self.inclusion_probabilities.tolist()
            ),
            "metadata": self.metadata,
        }


def _ordered_ids(ids: Sequence[str], budget: int) -> tuple[list[str], np.ndarray]:
    values = list(ids)
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("ids must be a nonempty sequence of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("ids must be unique")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be an integer")
    if not 1 <= budget <= len(values):
        raise ValueError("budget must be between 1 and pool size")
    order = np.array(sorted(range(len(values)), key=values.__getitem__), dtype=int)
    return [values[i] for i in order], order


def _strata(strata: Sequence[str] | None, order: np.ndarray) -> list[str]:
    if strata is None:
        return ["all"] * len(order)
    values = list(strata)
    if len(values) != len(order) or any(not isinstance(s, str) for s in values):
        raise ValueError("strata must contain one string per pool row")
    return [values[i] for i in order]


def _embeddings(embeddings: Any, order: np.ndarray, *, normalize: bool) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(order) or values.shape[1] == 0:
        raise ValueError("embeddings must have shape (pool size, positive dimension)")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    values = values[order]
    if normalize:
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("cosine embeddings must have finite nonzero norms")
        values = values / norms[:, None]
    return values


def _groups(labels: Sequence[Any]) -> tuple[list[Any], list[np.ndarray]]:
    keys = sorted(set(labels))
    return keys, [np.array([i for i, label in enumerate(labels) if label == key])
                  for key in keys]


def _quotas(sizes: np.ndarray, budget: int, mass: np.ndarray) -> np.ndarray:
    """Capacity-constrained largest-deficit apportionment, at least one per cell.

    Each additional slot goes to the cell furthest below its desired allocation.
    Equal deficits resolve by the previously sorted cell key. Lower bounds can
    change the requested allocation, which is recorded in the actual quotas.
    """
    if budget < len(sizes):
        raise ValueError("budget must cover every nonempty stratum")
    quota = np.ones(len(sizes), dtype=int)
    desired = budget * mass / mass.sum()
    for _ in range(budget - len(sizes)):
        deficits = desired - quota
        deficits[quota == sizes] = -np.inf
        quota[int(np.argmax(deficits))] += 1
    return quota


def _probability_sample(
    groups: list[np.ndarray], quotas: np.ndarray, n: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    probabilities = np.empty(n, dtype=float)
    for group, count in zip(groups, quotas):
        probabilities[group] = count / len(group)
        selected.extend(rng.choice(group, size=int(count), replace=False).tolist())
    indices = np.array(sorted(selected), dtype=int)
    # Horvitz-Thompson weights for a finite-pool mean, exactly normalized here
    # because each independently sampled cell has a fixed sample size.
    weights = 1.0 / (n * probabilities[indices])
    return indices, weights, probabilities


def _representative_weights(similarity: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Assign each pool item to a most-similar selected representative.

    Selected rows own themselves on a tie; other ties favor the smallest ID.
    This avoids assigning zero mass to duplicate representatives while remaining
    a valid nearest-representative assignment. These are approximation weights.
    """
    assignment = np.argmax(similarity, axis=1)
    assignment[selected] = np.arange(len(selected))
    return np.bincount(assignment, minlength=len(selected)).astype(float) / len(similarity)


def _facility(embedding: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    # Shifted cosine is nonnegative, enabling the monotone submodular objective
    # F(S) = sum_i max_{j in S} (1 + cosine(x_i, x_j)) / 2, with F(empty) = 0.
    # Dense matrix costs 8*N*N bytes; the pilot N=1000 uses about 8 MB.
    similarity = np.clip((embedding @ embedding.T + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    covered = np.zeros(len(embedding))
    heap = [(-float(gain), i, 0) for i, gain in enumerate(similarity.sum(axis=0))]
    heapq.heapify(heap)
    chosen: list[int] = []
    while len(chosen) < budget:
        negative_gain, index, evaluated_at = heapq.heappop(heap)
        if evaluated_at == len(chosen):
            chosen.append(index)
            covered = np.maximum(covered, similarity[:, index])
        else:
            gain = float(np.maximum(similarity[:, index] - covered, 0.0).sum())
            heapq.heappush(heap, (-gain, index, len(chosen)))
    selected = np.array(sorted(chosen), dtype=int)
    return selected, _representative_weights(similarity[:, selected], selected), chosen


def _minhash_signatures(
    texts: Sequence[str], num_hashes: int, shingle_size: int, minhash_seed: int,
) -> np.ndarray:
    if isinstance(num_hashes, bool) or not isinstance(num_hashes, int) or num_hashes < 1:
        raise ValueError("num_hashes must be a positive integer")
    if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size < 1:
        raise ValueError("shingle_size must be a positive integer")
    # 31-bit prime keeps a*x+b within signed int64 before taking the modulus.
    prime = 2**31 - 1
    rng = np.random.default_rng(minhash_seed)
    multipliers = rng.integers(1, prime, size=num_hashes, dtype=np.int64)
    offsets = rng.integers(0, prime, size=num_hashes, dtype=np.int64)
    signatures = np.empty((len(texts), num_hashes), dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        # Distinguish an empty document from any real word, and encode lengths
        # so the shingle representation cannot collide through concatenation.
        shingles = {tuple(tokens[i:i + shingle_size])
                    for i in range(max(1, len(tokens) - shingle_size + 1))}
        encoded = [repr(shingle).encode("utf-8") for shingle in sorted(shingles)]
        hashes = np.array([
            int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big") % prime
            for s in encoded
        ], dtype=np.int64)
        signatures[row] = ((hashes[:, None] * multipliers + offsets) % prime).min(axis=0)
    return signatures


def _minhash(
    texts: Sequence[str], budget: int, num_hashes: int, shingle_size: int,
    minhash_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    signatures = _minhash_signatures(texts, num_hashes, shingle_size, minhash_seed)
    chosen = [0]  # The initial point is the smallest stable ID, fixed in advance.
    covered = np.mean(signatures == signatures[0], axis=1)
    while len(chosen) < budget:
        distances = 1.0 - covered
        distances[chosen] = -np.inf
        next_index = int(np.argmax(distances))
        chosen.append(next_index)
        covered = np.maximum(covered, np.mean(signatures == signatures[next_index], axis=1))
    selected = np.array(sorted(chosen), dtype=int)
    similarity = np.column_stack([
        np.mean(signatures == signatures[index], axis=1) for index in selected
    ])
    return selected, _representative_weights(similarity, selected), chosen


def select(
    policy: str, ids: Sequence[str], budget: int, *, seed: int = 0,
    embeddings: Any = None, texts: Sequence[str] | None = None,
    strata: Sequence[str] | None = None, risk: Sequence[float] | None = None,
    targeted_fraction: float = 0.5, risk_threshold: float = 0.5,
    num_hashes: int = 64, shingle_size: int = 3, minhash_seed: int = 0,
) -> SelectionResult:
    """Select exactly ``budget`` unique items without observing target outcomes.

    ``risk`` is a finite input-side score. ``risk >= risk_threshold`` defines the
    targeted cell; choose threshold and targeted_fraction on development data.
    The risk policy crosses low/high cells with supplied strata, allocates quotas
    using a population-proportional/targeted mixture, then samples uniformly
    without replacement inside every nonempty cell. It therefore has known,
    positive first-order inclusion probabilities, unlike a deterministic top-k
    target list. ``targeted_fraction`` must be in [0, 1), preserving an explicit
    random population component. If no high-risk items exist it becomes
    proportional stratified sampling. Quota rounding/lower bounds are reported.

    Centroid proximity uses Euclidean distance to each construction stratum's
    centroid, mirroring the thesis's distance ranking; embeddings are *not*
    normalized for that policy. Facility location uses cosine similarity.
    MinHash estimates lexical word-shingle Jaccard similarity (not embeddings).
    Seed changes affect only randomized policies; minhash_seed is a separate,
    prespecified representation seed held fixed across acquisition repeats.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    stable_ids, order = _ordered_ids(ids, budget)
    n = len(order)
    labels = _strata(strata, order)
    metadata: dict[str, Any] = {"policy": policy, "pool_size": n, "budget": int(budget)}
    probabilities = None

    if policy in {"uniform_random", "stratified_random", "random_plus_targeted", "centroid_proximity"}:
        if policy == "uniform_random":
            labels = ["all"] * n
        if policy == "random_plus_targeted":
            if not np.isfinite(targeted_fraction) or not 0 <= targeted_fraction < 1:
                raise ValueError("targeted_fraction must be finite and in [0, 1)")
            if not np.isfinite(risk_threshold):
                raise ValueError("risk_threshold must be finite")
            values = np.asarray(risk, dtype=float)
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError("risk must contain one finite score per pool row")
            high = values[order] >= risk_threshold
            labels = [(label, bool(is_high)) for label, is_high in zip(labels, high)]
            metadata.update(targeted_fraction=float(targeted_fraction),
                            risk_threshold=float(risk_threshold), high_risk_count=int(high.sum()))
        keys, groups = _groups(labels)
        sizes = np.array([len(group) for group in groups])
        masses = sizes / n
        if policy == "random_plus_targeted" and high.any():
            target_masses = np.array([
                len(group) / int(high.sum()) if key[1] else 0.0
                for key, group in zip(keys, groups)
            ])
            masses = (1.0 - targeted_fraction) * masses + targeted_fraction * target_masses
        quotas = _quotas(sizes, budget, masses)
        metadata["cells"] = [
            {"key": key, "population_count": int(size), "sample_count": int(quota),
             "desired_mass": float(mass)}
            for key, size, quota, mass in zip(keys, sizes, quotas, masses)
        ]
        if policy == "centroid_proximity":
            embedding = _embeddings(embeddings, order, normalize=False)
            chosen, weights_by_row = [], np.zeros(n)
            for group, quota in zip(groups, quotas):
                distances = np.linalg.norm(embedding[group] - embedding[group].mean(axis=0), axis=1)
                picked = group[np.argsort(distances, kind="stable")[:quota]]
                chosen.extend(picked.tolist())
                weights_by_row[picked] = len(group) / (n * quota)
            selected = np.array(sorted(chosen), dtype=int)
            weights = weights_by_row[selected]
            metadata["estimator"] = "stratum-weighted deterministic representative mean; potentially biased"
        else:
            selected, weights, probabilities = _probability_sample(groups, quotas, n, seed)
            metadata.update(seed=seed, estimator="Horvitz-Thompson finite-pool mean")
    elif policy == "facility_location":
        embedding = _embeddings(embeddings, order, normalize=True)
        selected, weights, chosen = _facility(embedding, budget)
        metadata.update(selection_order=[stable_ids[i] for i in chosen],
                        similarity="(1 + cosine) / 2",
                        estimator="nearest-representative mass-weighted approximation; potentially biased")
    else:
        if texts is None or len(texts) != n or any(not isinstance(t, str) for t in texts):
            raise ValueError("texts must contain one string per pool row")
        selected, weights, chosen = _minhash(
            [texts[i] for i in order], budget, num_hashes, shingle_size, minhash_seed,
        )
        metadata.update(selection_order=[stable_ids[i] for i in chosen],
                        num_hashes=num_hashes, shingle_size=shingle_size, minhash_seed=minhash_seed,
                        estimator="nearest-representative MinHash mass-weighted approximation; potentially biased")

    metadata["selected_ids"] = [stable_ids[i] for i in selected]
    metadata["design_based_uncertainty_eligible"] = probabilities is not None
    if probabilities is not None:
        original_probabilities = np.empty(n)
        original_probabilities[order] = probabilities
    else:
        original_probabilities = None
    return SelectionResult(order[selected], weights, original_probabilities, metadata)
