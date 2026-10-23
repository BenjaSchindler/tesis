"""Label-bounded estimators for the planned golden-set refresh experiment.

The operational residual estimator receives only acquired current labels. Full
labels enter a separate research diagnostic for exact conditional design risk.
No model calls, dataset loading, representation selection or label repair here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def validate_pool(groups: Mapping[str, str], weights: Mapping[str, float]):
    if not groups or any(not isinstance(i, str) or not i or not isinstance(g, str) or not g for i, g in groups.items()):
        raise ValueError("Nonempty string population IDs and groups required")
    if set(groups.values()) != set(weights):
        raise ValueError("Weights must cover exactly the populated groups")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Group weights must be positive finite numbers")
    if not math.isclose(sum(weights.values()), 1., rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Group weights must sum to one")
    cells = defaultdict(list)
    for identifier, group in sorted(groups.items()):
        cells[group].append(identifier)
    return dict(sorted(cells.items()))


def finite_values(values, expected_ids, label):
    if set(values) != set(expected_ids):
        raise ValueError(label + " IDs do not match their declared scope")
    if any(not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) for value in values.values()):
        raise ValueError(label + " requires complete finite values")


def weighted_population_mean(values, groups, weights):
    cells = validate_pool(groups, weights)
    finite_values(values, groups, "Population outcome")
    return sum(weights[group] * sum(values[i] for i in ids) / len(ids) for group, ids in cells.items())


def residual_estimate(predictions, acquired_labels, sample_ids, groups, weights):
    """Prediction mean plus stratified sample residual; no unacquired labels."""
    cells = validate_pool(groups, weights)
    finite_values(predictions, groups, "Frozen predictor")
    if len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= set(groups):
        raise ValueError("Sample IDs must be unique members of the current pool")
    finite_values(acquired_labels, sample_ids, "Acquired label")
    sampled = defaultdict(list)
    for identifier in sample_ids:
        sampled[groups[identifier]].append(identifier)
    if set(sampled) != set(cells):
        raise ValueError("Every group needs a current sample")
    estimate = 0.
    for group, ids in cells.items():
        predicted_mean = sum(predictions[i] for i in ids) / len(ids)
        residual_mean = sum(acquired_labels[i] - predictions[i] for i in sampled[group]) / len(sampled[group])
        estimate += weights[group] * (predicted_mean + residual_mean)
    # Do not clip to [0,1]: that would silently change design unbiasedness.
    return float(estimate)


def current_sample_mean(acquired_labels, sample_ids, groups, weights):
    return residual_estimate(dict.fromkeys(groups, 0.), acquired_labels, sample_ids, groups, weights)


def exact_conditional_risk(predictions, population_labels, groups, weights, quotas):
    """Research-only variance over independent SRS without replacement in cells."""
    cells = validate_pool(groups, weights)
    finite_values(predictions, groups, "Frozen predictor")
    finite_values(population_labels, groups, "Full diagnostic outcome")
    if set(quotas) != set(cells):
        raise ValueError("Quotas must cover exactly the current groups")
    variance, components = 0., {}
    for group, ids in cells.items():
        quota = quotas[group]
        if type(quota) is not int or not 0 < quota <= len(ids):
            raise ValueError("Each quota must be a positive bounded integer")
        residuals = np.array([population_labels[i] - predictions[i] for i in ids], dtype=float)
        residual_variance = float(residuals.var(ddof=1)) if len(ids) > 1 else 0.
        component = weights[group] ** 2 * (1 - quota / len(ids)) * residual_variance / quota
        components[group] = {"population_count": len(ids), "sample_count": quota,
                             "residual_finite_variance_ddof1": residual_variance,
                             "weighted_estimator_variance": component}
        variance += component
    target = weighted_population_mean(population_labels, groups, weights)
    return {"expectation": float(target), "bias": 0., "variance": variance, "rmse": math.sqrt(variance),
            "components": components,
            "scope": "Exact conditional design risk with a fixed predictor and complete finite outcomes; no model/label/population uncertainty."}


def nested_stratified_samples(groups: Mapping[str, str], budgets_per_group: Sequence[int], seed: int):
    """One permutation per group; increasing sample budgets share a prefix."""
    if not groups:
        raise ValueError("Empty population")
    cells = validate_pool(groups, dict.fromkeys(set(groups.values()), 1 / len(set(groups.values()))))
    if not budgets_per_group or list(budgets_per_group) != sorted(set(budgets_per_group)):
        raise ValueError("Budgets must be nonempty, strictly increasing")
    if any(type(n) is not int or not 0 < n <= min(map(len, cells.values())) for n in budgets_per_group):
        raise ValueError("Budget exceeds a group or is invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    permutations = {group: rng.permutation(ids).tolist() for group, ids in cells.items()}
    return {budget * len(cells): sorted(i for ids in permutations.values() for i in ids[:budget])
            for budget in budgets_per_group}


def nearest_group_predictions(selected_ids, selected_labels, construction_groups, current_groups,
                              selected_embeddings, current_embeddings):
    """Nearest frozen golden example within group, with stable ID tie-breaking.

    Embeddings are supplied in selected_ids / sorted(current_groups) order.
    No current label is accepted. Normalization is rowwise; no pool fitting.
    """
    if list(selected_ids) != sorted(set(selected_ids)) or not set(selected_ids) <= set(construction_groups):
        raise ValueError("Selected construction IDs must be sorted, unique and known")
    if set(construction_groups) & set(current_groups):
        raise ValueError("Construction and current example IDs must be disjoint")
    finite_values(selected_labels, selected_ids, "Golden labels")
    current_ids = sorted(current_groups)
    selected = np.array(selected_embeddings, dtype=float, copy=True)
    current = np.array(current_embeddings, dtype=float, copy=True)
    if selected.ndim != 2 or current.ndim != 2 or selected.shape[0] != len(selected_ids) or current.shape[0] != len(current_ids) or selected.shape[1] != current.shape[1] or selected.shape[1] == 0:
        raise ValueError("Embedding shapes do not match explicit IDs")
    if not np.isfinite(selected).all() or not np.isfinite(current).all():
        raise ValueError("Embeddings must be finite")
    selected_norms, current_norms = np.linalg.norm(selected, axis=1), np.linalg.norm(current, axis=1)
    if (selected_norms == 0).any() or (current_norms == 0).any():
        raise ValueError("Zero embeddings cannot define cosine neighbors")
    selected /= selected_norms[:, None]
    current /= current_norms[:, None]
    predictions, neighbors = {}, {}
    for index, identifier in enumerate(current_ids):
        positions = [j for j, i in enumerate(selected_ids) if construction_groups[i] == current_groups[identifier]]
        if not positions:
            raise ValueError("Golden set lacks a current group")
        nearest = positions[int(np.argmax(selected[positions] @ current[index]))]
        neighbor = selected_ids[nearest]
        predictions[identifier] = float(selected_labels[neighbor])
        neighbors[identifier] = neighbor
    return predictions, neighbors


def resampling_diagnostics(predictions, population_labels, groups, weights, budgets_per_group, seeds):
    """Research simulation with a strict acquired-label boundary at each draw."""
    finite_values(population_labels, groups, "Diagnostic full outcomes")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Unique nonempty resampling seeds required")
    truth = weighted_population_mean(population_labels, groups, weights)
    errors = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        for budget, sample_ids in nested_stratified_samples(groups, budgets_per_group, seed).items():
            acquired = {i: population_labels[i] for i in sample_ids}
            corrected = residual_estimate(predictions, acquired, sample_ids, groups, weights)
            fresh = current_sample_mean(acquired, sample_ids, groups, weights)
            errors[budget]["residual_corrected"].append(corrected - truth)
            errors[budget]["same_fresh_sample_only"].append(fresh - truth)
    summary = []
    for budget, methods in sorted(errors.items()):
        for method, raw in methods.items():
            values = np.asarray(raw)
            summary.append({"budget": budget, "estimator": method, "draws": len(values),
                            "bias": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
                            "median_absolute_error": float(np.median(np.abs(values))),
                            "p90_absolute_error": float(np.quantile(np.abs(values), .9)),
                            "scope": "Sampling variability conditional on this observed finite population and fixed predictor; not independent study replications."})
    return summary
