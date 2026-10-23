"""Independent outcome grading and descriptive analysis of frozen v2 experiments.

No model calls and no selector fitting occur here. Infrastructure failures remain
unknown. A completed response with an invalid answer is an observed incorrect
result. Synthetic descendants are aggregated within their prespecified families.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.production_eval.benchmark_data import check_answer
from experiments.production_eval.synthetic_tasks import json_exact_grade


DEFAULT_CONFIGS = ("qwen_direct", "qwen_work", "gemini_direct")
VARIANTS = ("base", "paraphrase", "counterfactual", "counterfactual_paraphrase")
PRIMARY_ROLES = {"construction", "reference"}


def sha256(payload: bytes | str) -> str:
    return hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def average(values: list[float]) -> float | None:
    return mean(values) if values else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def domain(row: dict[str, Any]) -> str:
    return row["task_family"] if row["answer_type"] == "json" else row["dataset"]


def role(row: dict[str, Any]) -> str:
    value = row["data_role"]
    if value == "development_construction":
        if row.get("split") != "construction":
            raise ValueError("Inconsistent synthetic construction role")
        return "construction"
    if value == "controlled_reference":
        if row.get("split") != "reference":
            raise ValueError("Inconsistent synthetic reference role")
        return "reference"
    if value not in {"development", "construction", "reference"}:
        raise ValueError(f"Unknown analysis role: {value}")
    return value


def regime(row: dict[str, Any]) -> str:
    return row.get("metadata", {}).get("shift_regime", "random_benchmark_split")


def validate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    family_scopes: dict[str, tuple[str, str, str]] = {}
    family_variants: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row["example_id"] in by_id:
            raise ValueError("Duplicate normalized example_id")
        if sha256(row["input_text"]) != row["input_hash"]:
            raise ValueError("Normalized input hash mismatch")
        scope = (domain(row), role(row), regime(row))
        family = row["family_id"]
        if family in family_scopes and family_scopes[family] != scope:
            raise ValueError("Family crosses domain, data role, or shift regime")
        family_scopes[family] = scope
        if row["answer_type"] == "json":
            variant = row.get("metadata", {}).get("variant")
            if variant not in VARIANTS:
                raise ValueError("Unexpected synthetic family variant")
            family_variants[family].append(variant)
            if not json_exact_grade(row["answer"], row["answer"]):
                raise ValueError("Invalid synthetic reference answer")
        else:
            check_answer(row["answer"], row["answer"], row["answer_type"])
        by_id[row["example_id"]] = row
    for variants in family_variants.values():
        if sorted(variants) != sorted(VARIANTS):
            raise ValueError("Synthetic family requires exactly its four descendants")
    return by_id


def grade_response(record: dict[str, Any], response: dict[str, Any] | None) -> dict[str, Any]:
    """An independent grade from normalized ground truth, not provider scores."""
    if response is None:
        return {"observed": False, "correct": None, "format_valid": None,
                "canonical_prediction": None, "status": "missing"}
    if response["input_hash"] != record["input_hash"]:
        raise ValueError("Response input hash does not match frozen task")
    if response.get("status") != "success":
        return {"observed": False, "correct": None, "format_valid": None,
                "canonical_prediction": None, "status": response.get("status", "unknown_status")}
    answer = response.get("answer")
    outer_valid = response.get("format_status") == "valid"
    if record["answer_type"] == "json":
        serialized = None
        if outer_valid:
            try:
                serialized = canonical_json(answer)
            except (TypeError, ValueError, OverflowError):
                pass
        # The outer answer field must contain the object, not JSON-in-a-string.
        format_valid = serialized is not None and isinstance(answer, dict)
        correct = format_valid and json_exact_grade(serialized, record["answer"])
        prediction = serialized if format_valid else None
    else:
        checked = check_answer(answer if outer_valid else None, record["answer"], record["answer_type"])
        correct, format_valid, prediction = checked["correct"], checked["valid_format"], checked["canonical_answer"]
    return {"observed": True, "correct": bool(correct), "format_valid": bool(format_valid),
            "canonical_prediction": prediction, "status": "success"}


def index_responses(responses: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]],
                    configs: tuple[str, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for response in responses:
        key = (response["config_id"], response["example_id"])
        if key[0] not in configs:
            raise ValueError(f"Response has unrequested config {key[0]}")
        if key[1] not in records_by_id:
            raise ValueError("Response does not belong to the supplied frozen records")
        if key in indexed:
            raise ValueError("Duplicate config/example response: supply final results once, not an attempts ledger")
        if response["input_hash"] != records_by_id[key[1]]["input_hash"]:
            raise ValueError("Response input hash mismatch")
        indexed[key] = response
    return indexed


def group_records(records: list[dict[str, Any]], *, by_regime: bool = False) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (domain(row), role(row)) + ((regime(row),) if by_regime else ())
        groups[key].append(row)
    return dict(groups)


def family_members(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        result[row["family_id"]].append(row)
    return dict(result)


def family_scores(records: list[dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]],
                  config: str) -> dict[str, float]:
    """Means only for families whose entire declared descendant set completed."""
    result = {}
    for family, members in family_members(records).items():
        values = [grades[(config, row["example_id"])] for row in members]
        if all(item["observed"] for item in values):
            result[family] = mean(float(item["correct"]) for item in values)
    return result


def family_missingness_bounds(records: list[dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]],
                              config: str) -> list[float]:
    """Sharp bounds for the fixed-family mean when unknown scores lie in [0,1].

    These are identification bounds from missing outcomes, not sampling CIs.
    Every declared descendant retains its weight even when it has no response.
    """
    lower, upper = [], []
    for members in family_members(records).values():
        values = [grades[(config, row["example_id"])] for row in members]
        correct = sum(int(value["correct"]) for value in values if value["observed"])
        unknown = sum(not value["observed"] for value in values)
        lower.append(correct / len(values))
        upper.append((correct + unknown) / len(values))
    return [mean(lower), mean(upper)]


def summarize_groups(records: list[dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]],
                     configs: tuple[str, ...], *, by_regime: bool = False) -> list[dict[str, Any]]:
    output = []
    for key, rows in sorted(group_records(records, by_regime=by_regime).items()):
        total_families = len(family_members(rows))
        for config in configs:
            values = [grades[(config, row["example_id"])] for row in rows]
            observed = [value for value in values if value["observed"]]
            completed_families = family_scores(rows, grades, config)
            correct = sum(value["correct"] for value in observed)
            output.append({"dataset": key[0], "role": key[1], **({"regime": key[2]} if by_regime else {}),
                           "config_id": config, "expected_examples": len(rows), "observed_examples": len(observed),
                           "unknown_examples": len(rows) - len(observed), "status_counts": dict(Counter(value["status"] for value in values)),
                           "correct_examples": correct, "format_failures": sum(not value["format_valid"] for value in observed),
                           "accuracy_among_observed": correct / len(observed) if observed else None,
                           "coverage": len(observed) / len(rows), "expected_families": total_families,
                           "complete_families": len(completed_families),
                           "family_mean_accuracy_among_complete": average(list(completed_families.values())),
                           "full_population_family_mean_accuracy": average(list(completed_families.values())) if len(completed_families) == total_families else None,
                           "full_family_accuracy_bounds_from_unknown_outcomes": family_missingness_bounds(rows, grades, config),
                           "bounds_interpretation": "Fixed-population identification bounds assigning unknown correctness 0 or 1; not a confidence interval.",
                           "missingness_note": "Unknown outcomes are excluded, not treated as model mistakes; observed-only means may be biased if missingness is selective."})
    return output


def paired_groups(records: list[dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]],
                  configs: tuple[str, ...], *, by_regime: bool = False) -> list[dict[str, Any]]:
    output = []
    for key, rows in sorted(group_records(records, by_regime=by_regime).items()):
        for anchor, candidate in itertools.combinations(configs, 2):
            pairs = [(grades[(anchor, row["example_id"])], grades[(candidate, row["example_id"])]) for row in rows]
            completed = [(a, b) for a, b in pairs if a["observed"] and b["observed"]]
            valid_predictions = [(a, b) for a, b in completed if a["format_valid"] and b["format_valid"]]
            a_families, b_families = family_scores(rows, grades, anchor), family_scores(rows, grades, candidate)
            shared = sorted(a_families.keys() & b_families.keys())
            deltas = [b_families[family] - a_families[family] for family in shared]
            anchor_bounds = family_missingness_bounds(rows, grades, anchor)
            candidate_bounds = family_missingness_bounds(rows, grades, candidate)
            output.append({"dataset": key[0], "role": key[1], **({"regime": key[2]} if by_regime else {}),
                           "anchor_config": anchor, "candidate_config": candidate,
                           "delta_direction": "candidate_minus_anchor", "expected_examples": len(rows),
                           "paired_observed_examples": len(completed), "paired_complete_families": len(shared),
                           "expected_families": len(family_members(rows)),
                           "example_mean_delta": average([int(b["correct"]) - int(a["correct"]) for a, b in completed]),
                           "family_mean_delta": average(deltas),
                           "full_population_family_mean_delta": average(deltas) if len(shared) == len(family_members(rows)) else None,
                           "paired_families_with_nonzero_delta": sum(value != 0 for value in deltas),
                           "paired_family_delta_range": [min(deltas), max(deltas)] if deltas else None,
                           "full_family_delta_is_constant": len(set(deltas)) == 1 if len(shared) == len(family_members(rows)) else None,
                           "full_family_delta_bounds_from_unknown_outcomes": [candidate_bounds[0] - anchor_bounds[1], candidate_bounds[1] - anchor_bounds[0]],
                           "correctness_disagreement": average([float(a["correct"] != b["correct"]) for a, b in completed]),
                           "candidate_wins": sum(not a["correct"] and b["correct"] for a, b in completed),
                           "anchor_wins": sum(a["correct"] and not b["correct"] for a, b in completed),
                           "both_valid_answer_pairs": len(valid_predictions),
                           "answer_disagreement_among_valid": average([float(a["canonical_prediction"] != b["canonical_prediction"]) for a, b in valid_predictions]),
                           "uncertainty": "Descriptive only; descendants and acquisition seeds are not independent replicates."})
    return output


def synthetic_consistency(records: list[dict[str, Any]], grades: dict[tuple[str, str], dict[str, Any]],
                          configs: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    synthetic = [row for row in records if row["answer_type"] == "json"]
    for key, rows in sorted(group_records(synthetic, by_regime=True).items()):
        families = family_members(rows)
        for config in configs:
            family_stats = []
            incomplete = 0
            for family, siblings in families.items():
                by_variant = {row["metadata"]["variant"]: row for row in siblings}
                values = {variant: grades[(config, by_variant[variant]["example_id"])] for variant in VARIANTS}
                if not all(value["observed"] for value in values.values()):
                    incomplete += 1
                    continue
                paraphrases = [("base", "paraphrase"), ("counterfactual", "counterfactual_paraphrase")]
                counterfactuals = [("base", "counterfactual"), ("paraphrase", "counterfactual_paraphrase")]
                for a, b in paraphrases:
                    if not json_exact_grade(by_variant[a]["answer"], by_variant[b]["answer"]):
                        raise ValueError("Paraphrase pair does not preserve ground truth")
                for a, b in counterfactuals:
                    if json_exact_grade(by_variant[a]["answer"], by_variant[b]["answer"]):
                        raise ValueError("Counterfactual pair does not change ground truth")
                valid_paraphrases = [(a, b) for a, b in paraphrases if values[a]["format_valid"] and values[b]["format_valid"]]
                valid_counterfactuals = [(a, b) for a, b in counterfactuals if values[a]["format_valid"] and values[b]["format_valid"]]
                family_stats.append({
                    "family_id": family,
                    # Invalid answers fail the operational consistency test;
                    # never count two missing/invalid responses as consistent.
                    "paraphrase_consistency": mean(float(values[a]["format_valid"] and values[b]["format_valid"] and values[a]["canonical_prediction"] == values[b]["canonical_prediction"]) for a, b in paraphrases),
                    "paraphrase_both_correct": mean(float(values[a]["correct"] and values[b]["correct"]) for a, b in paraphrases),
                    "counterfactual_prediction_changed": mean(float(values[a]["format_valid"] and values[b]["format_valid"] and values[a]["canonical_prediction"] != values[b]["canonical_prediction"]) for a, b in counterfactuals),
                    "counterfactual_both_correct": mean(float(values[a]["correct"] and values[b]["correct"]) for a, b in counterfactuals),
                    "valid_paraphrase_pairs": len(valid_paraphrases), "valid_counterfactual_pairs": len(valid_counterfactuals),
                    "all_four_correct": all(value["correct"] for value in values.values())})
            metrics = ("paraphrase_consistency", "paraphrase_both_correct", "counterfactual_prediction_changed", "counterfactual_both_correct", "all_four_correct")
            output.append({"dataset": key[0], "role": key[1], "regime": key[2], "config_id": config,
                           "expected_families": len(families), "complete_families": len(family_stats), "incomplete_families": incomplete,
                           **{metric: average([float(value[metric]) for value in family_stats]) for metric in metrics},
                           "valid_paraphrase_pairs": sum(value["valid_paraphrase_pairs"] for value in family_stats),
                           "valid_counterfactual_pairs": sum(value["valid_counterfactual_pairs"] for value in family_stats),
                           "definition": "Equal family means, each over its two prompt-style pairs; completed invalid formats fail consistency/change. Prediction change alone does not establish correctness.",
                           "shift_interpretation": "Controlled synthetic regimes; no future-production inference."})
    return output


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Audit timestamps must include UTC offset")
    return result


def audit_freeze_timing(manifest: dict[str, Any], responses: list[dict[str, Any]],
                        records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    created = manifest.get("created_at") or manifest.get("created_at_utc")
    if not created:
        raise ValueError("Frozen selection manifest requires creation timestamp")
    freeze_time = parse_time(created)
    checked = missing = 0
    for response in responses:
        record = records_by_id[response["example_id"]]
        if role(record) not in PRIMARY_ROLES or response.get("status") != "success":
            continue
        timestamp = response.get("started_at_utc") or response.get("completed_at_utc")
        if timestamp is None:
            missing += 1
            continue
        if parse_time(timestamp) < freeze_time:
            raise ValueError("Primary model response predates selection freeze")
        checked += 1
    status = "no_primary_completed_responses_to_audit" if checked == 0 and missing == 0 else "verified_available_timestamps" if not missing else "partially_unverified_missing_timestamps"
    return {"status": status,
            "freeze_time": created, "primary_completed_responses_checked": checked,
            "primary_completed_responses_without_timestamp": missing,
            "note": "Recorded local timestamps support workflow ordering, not externally attested provenance; development smoke calls are excluded."}


def verify_frozen_source_hashes(manifest: dict[str, Any], actual_hashes: dict[str, str]) -> dict[str, Any]:
    expected = manifest.get("sources_hash")
    if expected is None:
        return {"status": "unverified_no_frozen_source_hashes"}
    # Relocated exact copies are allowed, but all and only frozen source contents
    # must be supplied. This binds labels and references, not merely input IDs.
    if sorted(actual_hashes.values()) != sorted(expected.values()):
        raise ValueError("Analysis source contents differ from the frozen selection manifest")
    return {"status": "verified", "frozen_sources": len(expected), "supplied_sources": len(actual_hashes),
            "binding": "Complete normalized file hashes include reference answers and role assignments."}


def analyze_frozen_selections(manifest: dict[str, Any], records: list[dict[str, Any]],
                              grades: dict[tuple[str, str], dict[str, Any]], configs: tuple[str, ...]) -> dict[str, Any]:
    construction = {key[0]: rows for key, rows in group_records(records).items() if key[1] == "construction"}
    references = {(key[0], key[2]): rows for key, rows in group_records(records, by_regime=True).items() if key[1] == "reference"}
    all_construction_families = family_members([row for row in records if role(row) == "construction"])
    output = []
    failure_discovery = []
    seen_runs = set()
    for run in manifest["runs"]:
        key = (run["dataset"], run["policy"], run["budget"], run["seed"], run["embedding_variant"])
        if key in seen_runs:
            raise ValueError("Duplicate frozen selection run")
        seen_runs.add(key)
        declared_pool = run["construction_family_ids"]
        mixture = run["dataset"] == "balanced_mixture"
        if mixture:
            if not set(declared_pool) <= all_construction_families.keys():
                raise ValueError("Mixture family pool includes unknown/nonconstruction families")
            rows = [row for family in declared_pool for row in all_construction_families[family]]
            actual_domain = {family: domain(all_construction_families[family][0]) for family in declared_pool}
            declared_domain = run.get("family_domain", manifest.get("family_domain"))
            if declared_domain is None or any(declared_domain.get(family) != value for family, value in actual_domain.items()):
                raise ValueError("Mixture family-domain mapping differs from normalized records")
            counts = Counter(actual_domain.values())
            if len(counts) != 4 or set(counts.values()) != {25}:
                raise ValueError("Prespecified balanced mixture requires four domains with 25 families each")
            reference_weights = run.get("reference_domain_weights", manifest.get("reference_domain_weights"))
            if reference_weights is None or set(reference_weights) != set(counts) or any(not math.isclose(weight, .25) for weight in reference_weights.values()):
                raise ValueError("Prespecified mixture reference requires equal quarter-weight per domain")
        else:
            rows = construction[run["dataset"]]
        pool = set(family_members(rows))
        if len(set(declared_pool)) != len(declared_pool) or set(declared_pool) != pool:
            raise ValueError("Frozen construction family IDs do not match supplied complete pool")
        ids, weights = run["selected_family_ids"], run["sample_weights"]
        if len(ids) != run["budget"] or len(set(ids)) != len(ids) or not set(ids) <= pool:
            raise ValueError("Invalid frozen selected family IDs/budget")
        if len(weights) != len(ids) or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Invalid frozen selection weights")
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Frozen weights must estimate a normalized population mean")
        probabilities = run.get("population_inclusion_probabilities")
        if probabilities is not None:
            if len(probabilities) != len(declared_pool) or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p <= 1 for p in probabilities):
                raise ValueError("Invalid frozen inclusion probabilities")
            if not math.isclose(sum(probabilities), run["budget"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Fixed-size inclusion probabilities must sum to the budget")
            probability_map = dict(zip(declared_pool, probabilities))
            if any(not math.isclose(weight, 1 / (len(pool) * probability_map[family]), rel_tol=1e-9, abs_tol=1e-9) for family, weight in zip(ids, weights)):
                raise ValueError("Declared probability weights differ from Horvitz-Thompson mean weights")
        members = family_members(rows)
        for config in configs:
            complete_scores = family_scores(rows, grades, config)
            complete_selected = [family for family in ids if family in complete_scores]
            selected_complete = len(complete_selected) == len(ids)
            selected_members = [row for family in ids for row in members[family]]
            descendant_grades = [grades[(config, row["example_id"])] for row in selected_members]
            observed_descendants = [value for value in descendant_grades if value["observed"]]
            failing_families = sum(complete_scores[family] < 1 for family in complete_selected)
            weighted_family_failure = sum(weight * float(complete_scores[family] < 1) for family, weight in zip(ids, weights)) if selected_complete else None
            full_family_failure = mean(float(value < 1) for value in complete_scores.values()) if len(complete_scores) == len(pool) else None
            weighted_descendant_error = sum(weight * (1 - complete_scores[family]) for family, weight in zip(ids, weights)) if selected_complete else None
            failure_discovery.append({
                "dataset": run["dataset"], "policy": run["policy"], "budget_families": run["budget"],
                "seed": run["seed"], "embedding_variant": run["embedding_variant"], "estimator": run["estimator"], "config_id": config,
                "expected_selected_families": len(ids), "complete_selected_families": len(complete_selected),
                "expected_selected_responses": len(selected_members), "observed_selected_responses": len(observed_descendants),
                "failing_complete_selected_families": failing_families,
                "failing_families_per_100_complete_selected_families": 100 * failing_families / len(complete_selected) if complete_selected else None,
                "full_selection_failing_family_yield_per_100": 100 * failing_families / len(ids) if selected_complete else None,
                "failed_responses_per_100_observed_selected_responses": 100 * sum(not value["correct"] for value in observed_descendants) / len(observed_descendants) if observed_descendants else None,
                "estimated_construction_family_failure_rate": weighted_family_failure,
                "full_construction_family_failure_rate": full_family_failure,
                "family_failure_estimation_absolute_error": abs(weighted_family_failure - full_family_failure) if weighted_family_failure is not None and full_family_failure is not None else None,
                "estimated_construction_family_mean_descendant_error": weighted_descendant_error,
                "population_estimator_interpretation": "Horvitz-Thompson finite-pool mean under the recorded randomized design and complete selected outcomes" if probabilities is not None else "Representative-weight approximation; no design-unbiasedness claim",
                "failure_definition": "A family fails if at least one descendant is incorrect under the fixed exact-answer checker; no claim of a distinct semantic bug or human verification.",
                "yield_interpretation": "Unweighted yield describes the chosen review set; targeted enrichment can increase it without improving population estimation. Missing families are excluded only from explicitly observed-only metrics.",
                "cost_interpretation": "Response counts are evaluation work units, not human reviewer time or equal monetary cost; synthetic families contain four descendants."})
        for anchor, candidate in itertools.combinations(configs, 2):
            a = family_scores(rows, grades, anchor)
            b = family_scores(rows, grades, candidate)
            delta = {family: b[family] - a[family] for family in a.keys() & b.keys()}
            selected_complete = all(family in delta for family in ids)
            estimate = sum(weight * delta[family] for family, weight in zip(ids, weights)) if selected_complete else None
            complete_pool = len(delta) == len(pool)
            full = average(list(delta.values())) if complete_pool else None
            base = {"dataset": run["dataset"], "policy": run["policy"], "budget_families": run["budget"],
                    "seed": run["seed"], "embedding_variant": run["embedding_variant"], "estimator": run["estimator"],
                    "anchor_config": anchor, "candidate_config": candidate, "delta_direction": "candidate_minus_anchor",
                    "construction_families": len(pool), "paired_complete_construction_families": len(delta),
                    "paired_complete_selected_families": sum(family in delta for family in ids),
                    "selected_estimated_delta": estimate, "full_construction_delta": full,
                    "full_construction_family_delta_is_constant": len(set(delta.values())) == 1 if complete_pool else None,
                    "constant_target_note": "If every construction family has the same delta, all normalized-weight selectors reproduce it; zero error then does not discriminate selector quality.",
                    "construction_signed_error": estimate - full if estimate is not None and full is not None else None,
                    "construction_absolute_error": abs(estimate - full) if estimate is not None and full is not None else None,
                    "confidence_interval": None, "uncertainty_note": "No CI is claimed. Weights are frozen selector-estimator weights; deterministic representation does not create sampling probabilities."}
            reference_results = []
            for (reference_domain, reference_regime), reference_rows in sorted(references.items()):
                if reference_domain != run["dataset"]:
                    continue
                reference_a = family_scores(reference_rows, grades, anchor)
                reference_b = family_scores(reference_rows, grades, candidate)
                ref_families = set(family_members(reference_rows))
                shared = reference_a.keys() & reference_b.keys()
                ref_delta = mean(reference_b[family] - reference_a[family] for family in shared) if len(shared) == len(ref_families) else None
                reference_results.append({"regime": reference_regime, "reference_families": len(ref_families),
                                          "paired_complete_reference_families": len(shared), "full_reference_delta": ref_delta,
                                          "reference_signed_error": estimate - ref_delta if estimate is not None and ref_delta is not None else None,
                                          "reference_absolute_error": abs(estimate - ref_delta) if estimate is not None and ref_delta is not None else None,
                                          "construction_to_reference_shift": ref_delta - full if ref_delta is not None and full is not None else None,
                                          "interpretation": "Descriptive prediction error for a held-out benchmark or controlled regime; not future production."})
            if mixture:
                domain_parts = []
                for reference_domain, reference_weight in sorted(reference_weights.items()):
                    reference_rows = [row for row in records if role(row) == "reference" and domain(row) == reference_domain]
                    reference_a = family_scores(reference_rows, grades, anchor)
                    reference_b = family_scores(reference_rows, grades, candidate)
                    ref_families = set(family_members(reference_rows))
                    shared = reference_a.keys() & reference_b.keys()
                    ref_delta = mean(reference_b[family] - reference_a[family] for family in shared) if ref_families and len(shared) == len(ref_families) else None
                    domain_parts.append({"dataset": reference_domain, "weight": reference_weight,
                                         "reference_families": len(ref_families), "paired_complete_reference_families": len(shared),
                                         "full_reference_delta": ref_delta})
                ref_delta = sum(part["weight"] * part["full_reference_delta"] for part in domain_parts) if all(part["full_reference_delta"] is not None for part in domain_parts) else None
                reference_results.append({"regime": "equal_domain_reference_mixture",
                                          "reference_families": sum(part["reference_families"] for part in domain_parts),
                                          "paired_complete_reference_families": sum(part["paired_complete_reference_families"] for part in domain_parts),
                                          "full_reference_delta": ref_delta, "domain_components": domain_parts,
                                          "reference_signed_error": estimate - ref_delta if estimate is not None and ref_delta is not None else None,
                                          "reference_absolute_error": abs(estimate - ref_delta) if estimate is not None and ref_delta is not None else None,
                                          "construction_to_reference_shift": ref_delta - full if ref_delta is not None and full is not None else None,
                                          "interpretation": "Secondary artificial mixture: quarter-weight per domain, equal family weight within domain, all declared reference regimes combined; not production traffic."})
            output.append({**base, "references": reference_results})
    aggregate_groups = defaultdict(list)
    for row in output:
        key = tuple(row[field] for field in ("dataset", "policy", "budget_families", "embedding_variant", "anchor_config", "candidate_config"))
        aggregate_groups[key].append(row)
    aggregates = []
    for key, runs in sorted(aggregate_groups.items()):
        errors = [run["construction_absolute_error"] for run in runs if run["construction_absolute_error"] is not None]
        ref_errors = defaultdict(list)
        for run in runs:
            for ref in run["references"]:
                if ref["reference_absolute_error"] is not None:
                    ref_errors[ref["regime"]].append(ref["reference_absolute_error"])
        aggregates.append({**dict(zip(("dataset", "policy", "budget_families", "embedding_variant", "anchor_config", "candidate_config"), key)),
                           "acquisition_runs": len(runs), "scorable_construction_runs": len(errors),
                           "mean_construction_absolute_error": average(errors),
                           "reference_mean_absolute_errors": {reg: mean(values) for reg, values in sorted(ref_errors.items())},
                           "note": "Means across acquisition seeds describe selection variability on the same fixed outcomes; they are not independent study replications."})
    return {"frozen_run_count": len(manifest["runs"]), "paired_run_results": output, "aggregates": aggregates,
            "failure_discovery": failure_discovery,
            "selection_performed_during_analysis": False,
            "family_embedding_rule": "Use the previously frozen representation; synthetic base variant only as specified by selection manifest. No labels or descendant outcomes select embeddings here."}


def analyze(records: list[dict[str, Any]], responses: list[dict[str, Any]], configs: tuple[str, ...] = DEFAULT_CONFIGS,
            frozen_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(configs) < 1 or len(set(configs)) != len(configs):
        raise ValueError("Configs must be unique and nonempty")
    by_id = validate_records(records)
    indexed = index_responses(responses, by_id, configs)
    grades = {(config, row["example_id"]): grade_response(row, indexed.get((config, row["example_id"]))) for row in records for config in configs}
    result = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "descriptive_analysis", "record_count": len(records), "response_count": len(responses),
              "config_ids": list(configs), "score_direction": "Higher exact correctness is better; paired deltas are candidate minus anchor.",
              "group_summary": summarize_groups(records, grades, configs),
              "regime_summary": summarize_groups(records, grades, configs, by_regime=True),
              "paired_comparisons": paired_groups(records, grades, configs),
              "paired_regime_comparisons": paired_groups(records, grades, configs, by_regime=True),
              "synthetic_consistency": synthetic_consistency(records, grades, configs),
              "limitations": ["Public benchmark pretraining contamination is unknown.",
                              "Random benchmark reference roles and controlled synthetic regimes do not establish production drift or future predictive validity.",
                              "Completed formatting failures count as incorrect. Infrastructure failures and missing calls remain unknown.",
                              "Synthetic descendants are dependent; family means use the entire prescribed family, with no partial-family imputation.",
                              "No statistical confidence intervals or significance claims are produced.",
                              "Comparisons are between these configurations on frozen inputs; no general superiority claim is warranted."]}
    if frozen_manifest is not None:
        result["freeze_timing_audit"] = audit_freeze_timing(frozen_manifest, responses, by_id)
        result["selector_estimation"] = analyze_frozen_selections(frozen_manifest, records, grades, configs)
    return result


def render_markdown(result: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "pending" if value is None else f"{100 * value:.2f}%"
    lines = ["# Verifiable outcome experiments", "", "Descriptive retrospective results; benchmark reference roles and controlled synthetic changes are not production traffic.", "",
             "| Dataset | Role | Config | Completed / expected | Format failures | Accuracy on completed | Complete families | Family accuracy |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in result["group_summary"]:
        lines.append(f"| {row['dataset']} | {row['role']} | {row['config_id']} | {row['observed_examples']} / {row['expected_examples']} | {row['format_failures']} | {fmt(row['accuracy_among_observed'])} | {row['complete_families']} / {row['expected_families']} | {fmt(row['family_mean_accuracy_among_complete'])} |")
    lines.extend(["", "Unknown infrastructure outcomes are excluded from accuracy and remain visible in coverage. Completed invalid formats count as incorrect. Family accuracy requires all descendants in that family to have completed.", "",
                  "## Paired comparisons", "", "Positive deltas favor the candidate; all values are descriptive.", "",
                  "| Dataset | Role | Anchor | Candidate | Complete paired families | Family delta (pp) | Correctness disagreement |",
                  "|---|---|---|---|---:|---:|---:|"])
    for row in result["paired_comparisons"]:
        delta = "pending" if row["family_mean_delta"] is None else f"{100 * row['family_mean_delta']:+.2f}"
        lines.append(f"| {row['dataset']} | {row['role']} | {row['anchor_config']} | {row['candidate_config']} | {row['paired_complete_families']} / {row['expected_families']} | {delta} | {fmt(row['correctness_disagreement'])} |")
    lines.extend(["", "Synthetic paraphrase/contrafactual results, controlled regimes, coverage counts, and frozen selector errors are in the accompanying JSON. No confidence interval is claimed for deterministic selections or repeated acquisition seeds.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--responses", type=Path, action="append", default=[])
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--frozen-selections", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = list(itertools.chain.from_iterable(load_jsonl(path) for path in args.records))
    responses = list(itertools.chain.from_iterable(load_jsonl(path) for path in args.responses))
    manifest = json.loads(args.frozen_selections.read_text()) if args.frozen_selections else None
    record_hashes = {str(path): sha256(path.read_bytes()) for path in args.records}
    source_check = verify_frozen_source_hashes(manifest, record_hashes) if manifest else None
    result = analyze(records, responses, tuple(args.configs), manifest)
    result["frozen_source_hash_audit"] = source_check
    result["input_provenance"] = {"record_files": [{"path": path, "sha256": value} for path, value in record_hashes.items()],
                                  "response_files": [{"path": str(path), "sha256": sha256(path.read_bytes())} for path in args.responses],
                                  "selection_manifest": None if args.frozen_selections is None else {"path": str(args.frozen_selections), "sha256": sha256(args.frozen_selections.read_bytes())},
                                  "analysis_code_sha256": sha256(Path(__file__).read_bytes())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "outcome_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "outcome_analysis.md").write_text(render_markdown(result))
    print(json.dumps({"records": len(records), "responses": len(responses), "output": str(args.output),
                      "selector_analysis_included": manifest is not None}))


if __name__ == "__main__":
    main()
