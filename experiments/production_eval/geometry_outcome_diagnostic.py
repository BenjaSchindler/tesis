"""Descriptive leave-one-family-out smoothness of frozen embedding geometries.

This is a mechanism diagnostic on the construction pool, not a trained selector
or an independent generalization study. All declared k values are reported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.production_eval.analyze_outcomes_v2 import (
    DEFAULT_CONFIGS, domain, family_scores, grade_response, index_responses,
    load_jsonl, role, validate_records, verify_frozen_source_hashes,
)

ROOT = Path(__file__).resolve().parents[2]


def uniform_mean_variance(target, budget):
    """Exact design variance of SRS without replacement on a finite family pool."""
    values = np.asarray(target, dtype=float)
    if values.ndim != 1:
        raise ValueError("Finite-pool target must be one-dimensional")
    n = len(values)
    if values.ndim != 1 or n < 2 or isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or not 1 <= budget <= n or not np.isfinite(values).all():
        raise ValueError("Invalid finite pool or sample budget")
    return float(values.var() * (n - budget) / ((n - 1) * budget))


def smoothness(matrix, target, ks=(1, 3, 5), domains=None):
    matrix = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    if not ks or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("Neighborhood sizes must be unique positive integers")
    if matrix.ndim != 2 or y.ndim != 1 or len(y) != len(matrix) or len(y) <= max(ks):
        raise ValueError("Invalid matrix/target or neighborhood budget")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(y).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Nonfinite data or zero embedding")
    matrix = matrix / norms
    distances = 1 - np.clip(matrix @ matrix.T, -1, 1)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    baseline = (y.sum() - y) / (len(y) - 1)
    mse_baseline = float(np.mean((y - baseline) ** 2))
    labels = ["all"] * len(y) if domains is None else list(domains)
    if len(labels) != len(y) or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("One nonempty domain label is required per family")
    domain_baseline = np.empty_like(y)
    domain_counts = {}
    for label in sorted(set(labels)):
        members = np.array([index for index, value in enumerate(labels) if value == label], dtype=int)
        if len(members) < 2:
            raise ValueError("LOO domain mean requires at least two families per domain")
        domain_baseline[members] = (y[members].sum() - y[members]) / (len(members) - 1)
        domain_counts[label] = len(members)
    mse_domain_baseline = float(np.mean((y - domain_baseline) ** 2))
    # Exact within-domain constants have a perfect domain-only predictor. Avoid
    # promoting floating-point roundoff into a nonzero baseline denominator.
    constant_within_domain = all(np.ptp(y[np.array(labels) == label]) == 0 for label in domain_counts)
    if constant_within_domain:
        domain_baseline = y.copy()
        mse_domain_baseline = 0.0
    rows = []
    for k in ks:
        neighbors = order[:, :k]
        if any(i in indices for i, indices in enumerate(neighbors)):
            raise AssertionError("Own family included in neighborhood")
        prediction = y[neighbors].mean(axis=1)
        mse = float(np.mean((prediction - y) ** 2))
        rows.append({"k": k, "families": len(y), "target_mean": float(y.mean()),
                     "target_population_variance": float(y.var()),
                     "constant_target": bool(np.ptp(y) == 0),
                     "loo_global_mean_mse": mse_baseline,
                     "loo_domain_mean_mse": mse_domain_baseline,
                     "loo_neighbor_mean_mse": mse,
                     "loo_global_mean_mae": float(np.mean(np.abs(y - baseline))),
                     "loo_domain_mean_mae": float(np.mean(np.abs(y - domain_baseline))),
                     "loo_neighbor_mean_mae": float(np.mean(np.abs(y - prediction))),
                     "relative_mse_improvement": 1 - mse / mse_baseline if mse_baseline and np.ptp(y) != 0 else None,
                     "relative_mse_improvement_vs_domain_mean": 1 - mse / mse_domain_baseline if mse_domain_baseline else None,
                     "neighbor_minus_domain_mean_mse": mse - mse_domain_baseline,
                     "domain_baseline_perfect": bool(constant_within_domain),
                     "domain_family_counts": domain_counts,
                     "domain_baseline_interpretation": "Leave the target family out, average all other construction families from its known domain. Post-hoc mechanism control for domain-mixture composition; not an independent validation set.",
                     "mean_neighbor_absolute_target_difference": float(np.mean(np.abs(y[:, None] - y[neighbors]))),
                     "tie_policy": "Stable frozen family-ID order; arbitrary among equal-distance families",
                     "families_with_exact_distance_tie_at_k_boundary": int(np.sum(distances[np.arange(len(y)), order[:, k - 1]] == distances[np.arange(len(y)), order[:, k]])) if k < len(y) - 1 else 0,
                     "uncertainty": "Descriptive; overlapping neighbor sets and outcomes are dependent. No p value or confidence interval."})
    return rows


def load_frozen_feature(path, construction, manifest):
    """Bind file contents and its entire family pool to the pre-outcome freeze.

    MPNet's shared encoder writer used the key example_ids for IDs deliberately
    remapped to family_id by freeze_selections_v2.load_construction. Other files
    use family_ids. The frozen input manifest resolves either representation.
    """
    path = Path(path)
    evidence = next((row for row in manifest["feature_evidence"] if (ROOT / row["embedding_file"]).resolve() == path.resolve()), None)
    if evidence is None:
        raise ValueError("Feature file absent from frozen manifest")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != evidence["embedding_sha256"]:
        raise ValueError("Feature file differs from frozen content hash")
    dataset = evidence["dataset"]
    declared_inputs = manifest["input_manifest"][dataset]
    expected_ids = [row["family_id"] for row in declared_inputs]
    expected_by_example = {row["representative_example_id"]: row["family_id"] for row in declared_inputs}
    with np.load(path, allow_pickle=False) as feature:
        if "family_ids" in feature:
            ids = feature["family_ids"].tolist()
            source_key = "family_ids"
        elif "example_ids" in feature:
            raw_ids = feature["example_ids"].tolist()
            ids = raw_ids if set(raw_ids) <= set(expected_ids) else [expected_by_example[value] for value in raw_ids]
            source_key = "example_ids resolved against frozen family manifest"
        else:
            raise ValueError("Feature has no usable IDs")
        if ids != sorted(ids) or len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            raise ValueError("Feature must contain the complete frozen domain family pool in stable order")
        rows = [row for row in construction if row["family_id"] in set(ids)]
        if set(ids) != {row["family_id"] for row in rows}:
            raise ValueError("Feature includes nonconstruction family")
        if dataset != "balanced_mixture" and any(domain(row) != dataset for row in rows):
            raise ValueError("Feature family belongs to another domain")
        base_inputs = {row["family_id"]: row["input_hash"] for row in rows
                       if row["answer_type"] != "json" or row["metadata"]["variant"] == "base"}
        expected_hashes = {row["family_id"]: row["input_hash"] for row in declared_inputs}
        if feature["input_hashes"].tolist() != [base_inputs[i] for i in ids] or base_inputs != expected_hashes:
            raise ValueError("Feature input hashes differ from frozen base inputs")
        matrix = feature["embeddings"].copy()
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("Embedding matrix rows do not align with family IDs")
    return ids, matrix, rows, {"path": str(path), "sha256": file_hash, "dataset": dataset,
                               "embedding_variant": evidence["embedding_variant"], "id_source": source_key,
                               "complete_frozen_pool_verified": True, "base_input_hashes_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--frozen-selections", type=Path, help="Defaults to frozen_selections.json beside feature-root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen_path = args.frozen_selections or args.feature_root.parent / "frozen_selections.json"
    manifest = json.loads(frozen_path.read_text())
    source_audit = verify_frozen_source_hashes(manifest, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.records})
    if source_audit["status"] != "verified":
        raise ValueError("Frozen source hashes are required for the diagnostic")
    records = [r for p in args.records for r in load_jsonl(p)]
    responses = [r for p in args.responses for r in load_jsonl(p)]
    by_id = validate_records(records)
    indexed = index_responses(responses, by_id, DEFAULT_CONFIGS)
    construction = [r for r in records if role(r) == "construction"]
    grades = {(c, r["example_id"]): grade_response(r, indexed.get((c, r["example_id"])))
              for r in construction for c in DEFAULT_CONFIGS}
    output = []
    precision = []
    files = []
    paths = sorted(args.feature_root.glob("*/*.npz"))
    expected_paths = {(ROOT / row["embedding_file"]).resolve() for row in manifest["feature_evidence"]}
    if {path.resolve() for path in paths} != expected_paths:
        raise ValueError("Feature root must contain all and only the frozen feature files")
    for path in paths:
        ids, matrix, rows, provenance = load_frozen_feature(path, construction, manifest)
        dataset, variant = provenance["dataset"], provenance["embedding_variant"]
        family_domains = {row["family_id"]: domain(row) for row in rows}
        domain_labels = [family_domains[family] for family in ids]
        files.append(provenance)
        scores = {c: family_scores(rows, grades, c) for c in DEFAULT_CONFIGS}
        if any(set(values) != set(ids) for values in scores.values()):
            output.append({"dataset": dataset, "embedding_variant": variant,
                           "status": "pending_unknown_construction_outcomes"})
            continue
        targets = {"qwen_direct_failure_fraction": [1 - scores["qwen_direct"][i] for i in ids],
                   "qwen_work_minus_direct": [scores["qwen_work"][i] - scores["qwen_direct"][i] for i in ids],
                   "gemini_minus_qwen_direct": [scores["gemini_direct"][i] - scores["qwen_direct"][i] for i in ids]}
        for name, target in targets.items():
            for result in smoothness(matrix, target, domains=domain_labels):
                output.append({"dataset": dataset, "embedding_variant": variant,
                               "target": name, "status": "complete", **result})
            # Read the budget grid from the frozen protocol, not mutable code.
            if variant == "mpnet_head":
                for budget in manifest["protocol"]["budgets"][dataset]:
                    variance = uniform_mean_variance(target, budget)
                    precision.append({"dataset": dataset, "target": name,
                                      "population_families": len(ids), "budget_families": budget,
                                      "srs_mean_variance": variance,
                                      "srs_expected_rmse_pp": 100 * variance ** .5,
                                      "interpretation": "Exact finite-pool sampling RMSE for equal-probability family sampling; not an interval for a deterministic selector or future traffic."})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Post-run mechanism diagnostic: does local geometry predict construction outcome variation?",
              "not_used_to_select_k_or_revise_frozen_selectors": True,
              "posthoc_refinement": "Known-domain LOO mean added after observing mixture/global-baseline gains, to test domain-composition explanation. Frozen features, targets, k values and selections are unchanged.",
              "features": files, "results": output, "uniform_sampling_precision": precision,
              "frozen_manifest": {"path": str(frozen_path), "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest()},
              "source_hash_audit": source_audit,
              "limitations": ["Overlapping leave-one-out neighborhoods are not independent replications.",
                              "This diagnoses the observed construction pool; it does not establish future production validity.",
                              "Equal family weight, including the artificial balanced mixture.",
                              "Constant targets cannot identify a useful geometry.",
                              "A gain over the global mean may only recover known domain differences; the domain-mean baseline reports this alternative explanation.",
                              "All k values 1, 3, 5 are shown; no best-k-only conclusion."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"diagnostic_rows": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
