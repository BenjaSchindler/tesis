"""Read completed frozen SFT runs and compare strict outcomes by task family.

No model training, hyperparameter changes, input selection or output repair.
Intervals resample independent reference families within domain/regime strata.
They condition on the two fitted optimization seeds and are not uncertainty
intervals over a population of possible training runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_tasks import json_exact_grade
from experiments.production_eval.field_diagnostics import diagnose, aggregate as aggregate_fields, strict_json

BASE = ROOT / "output/research/production_eval/experiments_v3"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def format_diagnostics(response, domain):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Nonfinite JSON number")

    try:
        value = json.loads(response, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return {"strict_json_valid": False, "required_schema_valid": False}
    valid = isinstance(value, dict)
    if domain == "structured_records":
        valid = valid and set(value) == {"selected_ids", "total_units", "total_value"}
        valid = valid and isinstance(value["selected_ids"], list) and all(type(item) is str for item in value["selected_ids"])
        valid = valid and type(value["total_units"]) is int and type(value["total_value"]) is int
    elif domain == "inventory_tracking":
        valid = valid and set(value) == {"stock", "total_units"}
        valid = valid and type(value["total_units"]) is int and isinstance(value["stock"], list)
        valid = valid and all(isinstance(item, dict) and set(item) == {"item", "east", "west"}
                              and type(item["item"]) is str and type(item["east"]) is int
                              and type(item["west"]) is int for item in value["stock"])
    else:
        raise ValueError("Unknown task domain")
    return {"strict_json_valid": True, "required_schema_valid": bool(valid)}


def read_responses(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(expected) or len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Completed run has missing, duplicate or unexpected response counts")
    result = {}
    for row in rows:
        original = expected.get(row["example_id"])
        if original is None:
            raise ValueError("Response is outside frozen evaluation data")
        for key in ("family_id", "input_hash", "domain", "role", "regime", "variant"):
            if row[key] != original[key]:
                raise ValueError("Response provenance disagrees with frozen task")
        grade = json_exact_grade(row["response"], original["answer"])
        if type(row["exact_match"]) is not bool or row["exact_match"] != grade:
            raise ValueError("Stored grade differs from independently recomputed strict grade")
        formatting = format_diagnostics(row["response"], row["domain"])
        fields = diagnose(strict_json(row["response"]), json.loads(original["answer"]), row["domain"],
                          available=formatting["required_schema_valid"])
        result[row["example_id"]] = {**row, **formatting, "field_diagnostics": fields,
                                      "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest()}
    return result


def paired_family_rows(tasks, response_sets, baseline, role="reference"):
    grouped = defaultdict(list)
    for identifier, task in tasks.items():
        if task["role"] == role:
            grouped[task["family_id"]].append(identifier)
    rows = []
    for family, identifiers in sorted(grouped.items()):
        task = tasks[identifiers[0]]
        if len(identifiers) != 2 or {tasks[i]["variant"] for i in identifiers} != {"base", "paraphrase"}:
            raise ValueError("Each evaluation family must retain both sibling wordings")
        baseline_mean = float(np.mean([baseline[i]["exact_match"] for i in identifiers]))
        trained_mean = float(np.mean([responses[i]["exact_match"] for responses in response_sets for i in identifiers]))
        rows.append({"family_id": family, "stratum": task["domain"] + "/" + task["regime"],
                     "baseline": baseline_mean, "trained": trained_mean, "delta": trained_mean - baseline_mean})
    return rows


def bootstrap_families(rows, repetitions=2000, seed=2026092401):
    groups = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    if not groups:
        raise ValueError("No reference families")
    keys = ["baseline", "trained", "delta"]
    arrays = [np.array([[row[key] for key in keys] for row in group]) for _, group in sorted(groups.items())]
    estimate = np.stack([array.mean(axis=0) for array in arrays]).mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.stack([array[rng.integers(0, len(array), size=(repetitions, len(array)))].mean(axis=1)
                        for array in arrays]).mean(axis=0)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return {key: {"estimate": float(estimate[i]), "ci95_conditional_family_bootstrap": [float(lower[i]), float(upper[i])]}
            for i, key in enumerate(keys)}


def paired_arm_rows(tasks, treatment_by_seed, comparator_by_seed, metric="exact_match"):
    """Compare complete arms, averaging paired seeds and both family wordings.

    Seeds are conditioned on, never resampled or treated as independent cases.
    This is a read-only analysis addition after initial runs had completed.
    """
    if not treatment_by_seed or set(treatment_by_seed) != set(comparator_by_seed):
        raise ValueError("Arm comparison requires the same completed optimization seeds")
    grouped = defaultdict(list)
    for identifier, task in tasks.items():
        if task["role"] == "reference":
            grouped[task["family_id"]].append(identifier)
    rows = []
    for family, identifiers in sorted(grouped.items()):
        if len(identifiers) != 2 or {tasks[i]["variant"] for i in identifiers} != {"base", "paraphrase"}:
            raise ValueError("Arm comparison requires both sibling wordings")
        task = tasks[identifiers[0]]
        baseline = float(np.mean([responses[i][metric] for responses in comparator_by_seed.values() for i in identifiers]))
        trained = float(np.mean([responses[i][metric] for responses in treatment_by_seed.values() for i in identifiers]))
        rows.append({"family_id": family, "stratum": task["domain"] + "/" + task["regime"],
                     "baseline": baseline, "trained": trained, "delta": trained - baseline})
    return rows


def summarize_responses(tasks, responses):
    result = {}
    for role in ("validation", "reference"):
        selected = [response for identifier, response in responses.items() if tasks[identifier]["role"] == role]
        result[role] = {"examples": len(selected), "families": len({row["family_id"] for row in selected}),
                        "exact_match": float(np.mean([row["exact_match"] for row in selected])),
                        "strict_json_valid": float(np.mean([row["strict_json_valid"] for row in selected])),
                        "required_schema_valid": float(np.mean([row["required_schema_valid"] for row in selected])),
                        "schema_valid_but_wrong_count": sum(row["required_schema_valid"] and not row["exact_match"] for row in selected),
                        "generation_limit_count": sum(row["generation_limit_hit"] for row in selected)}
    grouped = defaultdict(list)
    for row in responses.values():
        if row["role"] == "reference":
            grouped[(row["domain"], row["regime"], row["variant"])].append(row["exact_match"])
    result["reference_slices"] = [{"domain": key[0], "regime": key[1], "variant": key[2],
                                   "examples": len(values), "exact_match": float(np.mean(values))}
                                  for key, values in sorted(grouped.items())]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest_path = BASE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha(manifest_path)
    data_path = ROOT / manifest["tasks_file"]
    if sha(data_path) != manifest["tasks_sha256"]:
        raise ValueError("Frozen tasks changed")
    all_tasks = [json.loads(line) for line in data_path.read_text().splitlines()]
    tasks = {row["example_id"]: row for row in all_tasks if row["role"] in {"validation", "reference"}}
    baseline_completion = BASE / "baseline/completed.json"
    if not baseline_completion.exists():
        raise RuntimeError("Baseline still incomplete; no outcome summary is valid yet")
    if json.loads(baseline_completion.read_text())["manifest_sha256"] != manifest_sha:
        raise ValueError("Baseline belongs to another manifest")
    baseline = read_responses(BASE / "baseline/responses.jsonl", tasks)
    def field_rows(config, responses):
        return [{"config_id": config, "example_id": row["example_id"], "family_id": row["family_id"],
                 "domain": row["domain"], "role": row["role"], "regime": row["regime"],
                 "availability": "available" if row["required_schema_valid"] else "completed_invalid_format",
                 **row["field_diagnostics"]} for row in responses.values()]
    fields = field_rows("base_checkpoint", baseline)
    run_rows, by_arm, by_arm_seed, completed_ids = [], defaultdict(list), defaultdict(dict), []
    for planned in manifest["runs"]:
        directory = BASE / "runs" / planned["run_id"]
        completion = directory / "completed.json"
        if not completion.exists():
            if args.allow_partial:
                continue
            raise RuntimeError("A planned run is still incomplete")
        status = json.loads(completion.read_text())
        if status["manifest_sha256"] != manifest_sha or status["run"] != planned:
            raise ValueError("Completed run disagrees with frozen plan")
        if status["training"]["optimizer_steps"] != 32 or status["training"]["example_presentations"] != 256:
            raise ValueError("Completed run violates matched training quantity")
        responses = read_responses(directory / "responses.jsonl", tasks)
        fields.extend(field_rows(planned["run_id"], responses))
        # Checkpoint hashing is a one-time, read-only provenance check.
        weights = sorted((directory / "checkpoint").glob("*.safetensors"))
        if not weights:
            raise ValueError("Completed training checkpoint weights are missing")
        hashes = {str(path.relative_to(ROOT)): sha(path) for path in weights}
        families = paired_family_rows(tasks, [responses], baseline)
        run_rows.append({"run_id": planned["run_id"], "arm": planned["arm"], "training_seed": planned["training_seed"],
                         "outcomes": summarize_responses(tasks, responses), "paired_reference": bootstrap_families(families),
                         "checkpoint_hashes": hashes, "training": {key: value for key, value in status["training"].items() if key != "records"},
                         "response_file_sha256": sha(directory / "responses.jsonl")})
        by_arm[planned["arm"]].append(responses)
        by_arm_seed[planned["arm"]][planned["training_seed"]] = responses
        completed_ids.append(planned["run_id"])
    arms = []
    for arm, responses in sorted(by_arm.items()):
        if len(responses) != len(manifest["parameters"]["training_seeds"]):
            continue  # Do not compare an arm with one completed seed as if both finished.
        families = paired_family_rows(tasks, responses, baseline)
        scores = [float(np.mean([row["exact_match"] for row in response.values() if row["role"] == "reference"])) for response in responses]
        arms.append({"arm": arm, "completed_training_seeds": len(responses),
                     "reference_seed_scores": scores, "reference_seed_score_range": [min(scores), max(scores)],
                     "paired_reference": bootstrap_families(families)})
    comparisons = []
    seed_set = set(manifest["parameters"]["training_seeds"])
    random_responses = by_arm_seed.get("random_base", {})
    if set(random_responses) == seed_set:
        for arm, responses in sorted(by_arm_seed.items()):
            if arm == "random_base" or set(responses) != seed_set:
                continue
            metrics = {}
            for metric in ("exact_match", "required_schema_valid"):
                family_rows = paired_arm_rows(tasks, responses, random_responses, metric)
                metrics[metric] = {"overall": bootstrap_families(family_rows), "strata": {}}
                for stratum in sorted({row["stratum"] for row in family_rows}):
                    metrics[metric]["strata"][stratum] = bootstrap_families([row for row in family_rows if row["stratum"] == stratum])
            comparisons.append({"arm": arm, "comparator": "random_base", "training_seeds": sorted(seed_set),
                                "metrics": metrics})
    report = {"status": "complete" if len(completed_ids) == len(manifest["runs"]) else "partial_completed_runs_only",
              "generated_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_sha,
              "planned_runs": len(manifest["runs"]), "completed_run_ids": completed_ids,
              "baseline": summarize_responses(tasks, baseline), "runs": run_rows, "arms": arms,
              "posthoc_paired_arm_comparisons": {"status": "secondary_read_only_comparison", "comparisons": comparisons,
                                                  "scope": "Added after three initial runs completed. Compares complete arms against matched-example random selection with the same optimization seeds. Family bootstrap conditions on these fitted seeds; no winner selection or multiple-comparison claim."},
              "posthoc_field_diagnostics": {"status": "secondary_posthoc_explanation", "primary_metrics_changed": False,
                                             "groups": aggregate_fields(fields),
                                             "scope": "Added after observing v2 zero exact-match outcomes. Invalid formats receive zero field credit; raw absolute-error means are undefined when required fields are unavailable. Equal family weights, no output repair or selection/training changes."},
              "analysis_code_sha256": sha(Path(__file__)),
              "interval_scope": "Paired percentile bootstrap of families within equal-weight domain/regime strata,2000 replicates,seed2026092401. Conditional on the observed optimization seeds; siblings remain together. No multiple-comparison adjustment or winner claim.",
              "claims_scope": "Controlled synthetic procedural tasks; no held public benchmark training, population-production generalization, real temporal drift, or validated semantic equivalence beyond exact oracle outputs."}
    output = BASE / ("analysis_complete.json" if report["status"] == "complete" else "analysis_partial.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "completed_runs": len(completed_ids), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
