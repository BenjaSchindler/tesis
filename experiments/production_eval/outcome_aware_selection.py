"""Execute the frozen exploratory amendment using construction outcomes only.

These selectors deliberately observe old-system errors or disagreement. Their
naive subset means are not unbiased population estimates. No reference outcome
is available to the selection function, and the amendment timing is checked
against the local responses used to construct the risk signal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.production_eval.analyze_outcomes_v2 import (
    DEFAULT_CONFIGS, analyze_frozen_selections, domain, family_members,
    grade_response, index_responses, load_jsonl, parse_time, role, validate_records,
    verify_frozen_source_hashes,
)
from experiments.production_eval.selection_methods import select


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SIGNAL_CONFIGS = ("qwen_direct", "qwen_work")


class UnknownConstructionOutcomes(ValueError):
    def __init__(self, families):
        super().__init__("Unknown construction outcomes prevent risk selection")
        self.families = sorted(families)


def validate_grade_scope(construction, grades):
    """Reject extra keys, including reference grades and third-system outcomes."""
    if any(role(row) != "construction" for row in construction):
        raise ValueError("Reference/development rows must not reach risk selection")
    ids = [row["example_id"] for row in construction]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate construction example IDs")
    expected = {(config, identifier) for identifier in ids for config in SIGNAL_CONFIGS}
    if set(grades) != expected:
        raise ValueError("Risk selection accepts exactly the two local construction grade sets")
    for value in grades.values():
        if type(value.get("observed")) is not bool:
            raise ValueError("Invalid observed grade flag")
        if value["observed"] and any(type(value.get(key)) is not bool for key in ("correct", "format_valid")):
            raise ValueError("Completed grades require Boolean correctness and format flags")


def risk_flags(construction, grades):
    """Accept only construction families and their two completed local systems."""
    validate_grade_scope(construction, grades)
    errors, disagreements = {}, {}
    unknown = []
    for family, members in family_members(construction).items():
        pairs = [(grades[("qwen_direct", row["example_id"])],
                  grades[("qwen_work", row["example_id"])]) for row in members]
        if not all(a["observed"] and b["observed"] for a, b in pairs):
            unknown.append(family)
            continue
        errors[family] = any(not a["correct"] for a, _ in pairs)
        disagreements[family] = any(
            (a["format_valid"] or b["format_valid"]) and
            (a["format_valid"], a["canonical_prediction"]) !=
            (b["format_valid"], b["canonical_prediction"]) for a, b in pairs)
    if unknown:
        raise UnknownConstructionOutcomes(unknown)
    return {"error": errors, "disagreement": disagreements}


def targeted_naive(ids, risk, budget, seed):
    """Fixed-size enrichment, retaining zero-inclusion-probability warning."""
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Stable sorted unique family IDs required")
    if not 0 < budget <= len(ids) or len(risk) != len(ids):
        raise ValueError("Invalid budget/risk length")
    risk = np.asarray(risk)
    if risk.shape != (len(ids),) or risk.dtype != np.dtype(bool):
        raise ValueError("Risk must be one Boolean per construction family")
    rng = np.random.default_rng(seed)
    high = np.flatnonzero(risk)
    low = np.flatnonzero(~np.asarray(risk, dtype=bool))
    chosen = rng.choice(high, min(len(high), budget), replace=False).tolist()
    if len(chosen) < budget:
        chosen.extend(rng.choice(low, budget - len(chosen), replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def make_manifest(base, construction, grades, amendment):
    validate_grade_scope(construction, grades)
    if tuple(amendment["construction_signal_configs"]) != SIGNAL_CONFIGS:
        raise ValueError("Amendment does not declare the implemented source configurations")
    if not amendment.get("construction_selection_only") or not amendment.get("independent_reference_not_used_for_selection"):
        raise ValueError("Amendment must preserve construction-only selection")
    if not amendment["seeds"] or any(type(seed) is not int for seed in amendment["seeds"]) or len(set(amendment["seeds"])) != len(amendment["seeds"]):
        raise ValueError("Amendment seeds must be nonempty and unique")
    families = family_members(construction)
    if any(len({domain(row) for row in members}) != 1 for members in families.values()):
        raise ValueError("Construction family crosses domains")
    family_domains = {family: domain(rows[0]) for family, rows in families.items()}
    pools = {}
    for run in base["runs"]:
        key = (run["dataset"], run["budget"])
        if key in pools and pools[key]["construction_family_ids"] != run["construction_family_ids"]:
            raise ValueError("Base selection pool varies within dataset/budget")
        if key in pools and any(pools[key].get(field) != run.get(field) for field in ("family_domain", "reference_domain_weights")):
            raise ValueError("Base mixture metadata varies within dataset/budget")
        pools[key] = run
    runs, skipped = [], []
    for (dataset, budget), prototype in sorted(pools.items()):
        ids = prototype["construction_family_ids"]
        if ids != sorted(ids) or len(ids) != len(set(ids)) or not ids or not 0 < budget <= len(ids):
            raise ValueError("Construction IDs/budget must be unique, sorted and valid")
        if not set(ids) <= families.keys():
            raise ValueError("Selection pool includes a nonconstruction/unknown family")
        if dataset == "balanced_mixture":
            declared_domains = prototype.get("family_domain", base.get("family_domain"))
            declared_weights = prototype.get("reference_domain_weights", base.get("reference_domain_weights"))
            actual_domains = {family: family_domains[family] for family in ids}
            if declared_domains != actual_domains or not declared_weights or set(declared_weights) != set(actual_domains.values()):
                raise ValueError("Mixture domain metadata does not match its frozen construction pool")
            if not all(np.isfinite(weight) and weight > 0 for weight in declared_weights.values()) or not np.isclose(sum(declared_weights.values()), 1):
                raise ValueError("Invalid mixture reference weights")
            if any(not np.isclose(sum(value == key for value in actual_domains.values()) / len(ids), weight)
                   for key, weight in declared_weights.items()):
                raise ValueError("Finite-pool HT target does not match declared mixture weights")
        elif any(family_domains[family] != dataset for family in ids):
            raise ValueError("Single-domain selection pool contains another domain")
        scoped_rows = [row for family in ids for row in families[family]]
        scoped_grades = {(config, row["example_id"]): grades[(config, row["example_id"])]
                         for row in scoped_rows for config in SIGNAL_CONFIGS}
        try:
            flags = risk_flags(scoped_rows, scoped_grades)
        except UnknownConstructionOutcomes as error:
            # Frozen rule: abort this domain/pool, with no imputation. Other
            # independent domains can still be selected and remain auditable.
            skipped.append({"dataset": dataset, "budget": budget,
                            "reason": "unknown_construction_outcomes", "unknown_family_ids": error.families})
            continue
        for signal, family_flags in flags.items():
            risk = np.array([family_flags[family] for family in ids], dtype=bool)
            for seed in amendment["seeds"]:
                for kind in ("targeted_naive", "hybrid_ht"):
                    policy = f"{signal}_{kind}"
                    if policy not in amendment["policies"]:
                        raise ValueError("Policy was not frozen in amendment")
                    if kind == "hybrid_ht":
                        # The numeric sampler is reused, but its usual input-only
                        # risk convention is deliberately replaced by observed
                        # construction outcomes in this separately named study.
                        result = select("random_plus_targeted", ids, budget,
                                        risk=risk, seed=seed, targeted_fraction=.5,
                                        risk_threshold=.5)
                        indices, weights = result.indices, result.sample_weights
                        pi = result.inclusion_probabilities.tolist()
                        estimator = "horvitz_thompson_finite_pool_mean"
                        metadata = result.metadata
                    else:
                        indices = targeted_naive(ids, risk, budget, seed)
                        weights = np.full(budget, 1 / budget)
                        pi = None
                        estimator = "targeted_subset_mean_not_population_estimate"
                        metadata = {"population_estimation_warning": "Some families can have zero selection probability; unweighted enrichment is not a population estimate."}
                    run = {"dataset": dataset, "policy": policy, "budget": budget,
                           "seed": seed, "embedding_variant": None,
                           "selected_family_ids": [ids[i] for i in indices],
                           "sample_weights": weights.tolist(), "estimator": estimator,
                           "population_inclusion_probabilities": pi,
                           "construction_family_ids": ids,
                           "risk_signal": signal, "risk_family_count": int(risk.sum()),
                           "selection_metadata": metadata}
                    for field in ("family_domain", "reference_domain_weights"):
                        if field in prototype:
                            run[field] = prototype[field]
                        elif dataset == "balanced_mixture" and field in base:
                            run[field] = base[field]
                    runs.append(run)
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_type": "outcome_aware_construction_selection",
            "rule_freeze_at_utc": amendment["frozen_at_utc"],
            "selection_happened_after_source_outputs": True,
            "skipped_pools": skipped, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--frozen-selections", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    amendment = json.loads(args.amendment.read_text())
    if sha(args.frozen_selections) != amendment["base_selection_manifest_sha256"]:
        raise ValueError("Base selection manifest changed after amendment")
    base = json.loads(args.frozen_selections.read_text())
    source_hashes = {str(path): sha(path) for path in args.records}
    source_check = verify_frozen_source_hashes(base, source_hashes)
    if source_check["status"] != "verified":
        raise ValueError("Frozen complete source hashes are required for this amendment")
    records = [r for path in args.records for r in load_jsonl(path)]
    by_id = validate_records(records)
    responses = [r for path in args.responses for r in load_jsonl(path)]
    indexed = index_responses(responses, by_id, DEFAULT_CONFIGS)
    freeze = parse_time(amendment["frozen_at_utc"])
    audited = []
    construction = [r for r in records if role(r) == "construction"]
    for row in construction:
        for config in amendment["construction_signal_configs"]:
            response = indexed.get((config, row["example_id"]))
            if response is None:
                continue  # The affected pool is explicitly skipped below.
            if not response.get("started_at_utc"):
                raise ValueError("Missing source construction timestamp")
            if parse_time(response["started_at_utc"]) < freeze:
                raise ValueError("Source model response predates rule amendment")
            audited.append(response)
    construction_grades = {(c, r["example_id"]): grade_response(r, indexed.get((c, r["example_id"])))
                           for r in construction for c in SIGNAL_CONFIGS}
    manifest = make_manifest(base, construction, construction_grades, amendment)
    # Reference and third-system grades are constructed only after selection.
    grades = {(c, r["example_id"]): grade_response(r, indexed.get((c, r["example_id"])))
              for r in records for c in DEFAULT_CONFIGS}
    result = analyze_frozen_selections(manifest, records, grades, DEFAULT_CONFIGS)
    provenance = {"amendment_sha256": sha(args.amendment),
                  "base_selection_sha256": sha(args.frozen_selections),
                  "frozen_source_hash_audit": source_check,
                  "record_files": [{"path": path, "sha256": value} for path, value in source_hashes.items()],
                  "code_sha256": sha(__file__),
                  "response_files": [{"path": str(p), "sha256": sha(p)} for p in args.responses],
                  "source_construction_responses_used": len(audited),
                  "source_construction_latency_seconds": sum(r["latency_seconds"] for r in audited),
                  "source_construction_accounted_usd": sum(r["charged_nanodollars"] for r in audited) / 1e9,
                  "risk_source": "qwen_direct correctness from oracle; both local outputs for disagreement",
                  "no_reference_outcomes_used_in_selection": True,
                  "human_review_minutes": None,
                  "note": "All construction source inference costs precede selection. GPU runtime is not free because the API charge is zero. Timing records are local, not externally attested."}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("selected_families.json", manifest),
                        ("selector_analysis.json", result), ("provenance.json", provenance)):
        with (args.output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"selection_runs": len(manifest["runs"]),
                      "skipped_pools": len(manifest["skipped_pools"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
