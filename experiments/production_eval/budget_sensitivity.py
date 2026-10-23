"""Isolated output-budget sensitivity on the older balanced construction pool.

prepare freezes 100 existing families (250 descendants) without reading model
outcomes. run is dry by default; --execute is needed to use the local server.
Only the allowed output-token budget changes from the pinned 1024-token work
configuration. These are exploratory construction comparisons, not references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.analyze_outcomes_v2 import (
    domain, family_members, grade_response, load_jsonl, role, verify_frozen_source_hashes,
)

VERSION = "construction_output_budget_sensitivity_v1"
BUDGETS = (2048, 4096)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "output/research/production_eval/experiments_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_frozen(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def choose_pool(frozen, records):
    runs = [run for run in frozen["runs"] if run["dataset"] == "balanced_mixture"]
    if not runs:
        raise ValueError("No previously frozen balanced mixture")
    ids = runs[0]["construction_family_ids"]
    if len(ids) != 100 or len(set(ids)) != 100 or any(run["construction_family_ids"] != ids for run in runs):
        raise ValueError("Expected the same 100-family pool in all frozen mixture runs")
    selected = [row for row in records if row["family_id"] in set(ids)]
    if len(selected) != 250 or len({row["example_id"] for row in selected}) != 250:
        raise ValueError("Expected exactly 250 unique descendants of the frozen pool")
    if any(role(row) != "construction" for row in selected):
        raise ValueError("Sensitivity pool must contain only construction rows")
    families = family_members(selected)
    if set(families) != set(ids):
        raise ValueError("Missing frozen family")
    actual_domains = {family: domain(rows[0]) for family, rows in families.items()}
    if actual_domains != runs[0].get("family_domain", frozen.get("family_domain")):
        raise ValueError("Frozen family-domain map changed")
    if sorted(Counter(actual_domains.values()).values()) != [25, 25, 25, 25]:
        raise ValueError("Expected 25 families in each of four domains")
    for family, members in families.items():
        if len({domain(row) for row in members}) != 1:
            raise ValueError("Family crosses domains")
        if members[0]["answer_type"] == "json":
            variants = {row["metadata"]["variant"] for row in members}
            if len(members) != 4 or variants != {"base", "paraphrase", "counterfactual", "counterfactual_paraphrase"}:
                raise ValueError("Synthetic family lost a descendant")
        elif len(members) != 1:
            raise ValueError("Benchmark family should contain one example")
    return sorted(selected, key=lambda row: row["example_id"]), sorted(ids), actual_domains


def variant_config(baseline, budget):
    if budget not in BUDGETS or baseline["provider"] != "local" or baseline["profile"] != "reason_then_answer":
        raise ValueError("Only local work-profile budgets 2048/4096 are allowed")
    if baseline["max_output_tokens"] != 1024 or baseline["max_context_tokens"] != 8192 or baseline["temperature"] != 0:
        raise ValueError("Baseline must have the prespecified 1024/8192/temperature0 settings")
    return {**baseline, "config_id": f"{baseline['config_id']}_b{budget}",
            "max_output_tokens": budget, "budget_sensitivity_revision": VERSION}


def verify_same_task_protocol(row, baseline, variant):
    base_request, new_request = inference.prepare(row, baseline), inference.prepare(row, variant)
    for key in ("system_prompt", "input_text", "schema"):
        if base_request["request"][key] != new_request["request"][key]:
            raise ValueError("Task, schema or work prompt changed in budget variant")
    differences = {key for key in baseline.keys() | variant.keys() if baseline.get(key) != variant.get(key)}
    if differences != {"config_id", "max_output_tokens", "budget_sensitivity_revision"}:
        raise ValueError("Variant changes more than output budget and bookkeeping")
    return base_request, new_request


def prepare_experiment(base_dir, output_dir):
    frozen_path = base_dir / "frozen_selections.json"
    frozen = json.loads(frozen_path.read_text())
    source_paths = [ROOT / path if not Path(path).is_absolute() else Path(path) for path in frozen["sources_hash"]]
    source_hashes = {str(path): sha(path) for path in source_paths}
    source_check = verify_frozen_source_hashes(frozen, source_hashes)
    if source_check["status"] != "verified":
        raise ValueError("Source binding must be verified")
    records = [row for path in source_paths for row in load_jsonl(path)]
    rows, family_ids, family_domain = choose_pool(frozen, records)
    baseline_config_path = base_dir / "configs/qwen_work.json"
    baseline = inference.load_config(baseline_config_path)
    baseline_output = base_dir / "inference/qwen_work.jsonl"
    baseline_meta = json.loads(Path(str(baseline_output) + ".run.json").read_text())
    if baseline_meta["config"] != baseline or baseline_meta["config_sha256"] != inference.digest(inference.canonical(baseline)):
        raise ValueError("Expected the current pinned 1024 baseline configuration")
    variants = {budget: variant_config(baseline, budget) for budget in BUDGETS}
    requests = []
    for row in rows:
        baseline_request = inference.prepare(row, baseline)
        hashes = {}
        for budget, config in variants.items():
            checked_base, changed = verify_same_task_protocol(row, baseline, config)
            if checked_base["cache_key"] != baseline_request["cache_key"]:
                raise ValueError("Baseline cache key changed during preparation")
            hashes[str(budget)] = changed["cache_key"]
        requests.append({"example_id": row["example_id"], "input_hash": row["input_hash"],
                         "family_id": row["family_id"], "dataset": domain(row),
                         "baseline_request_sha256": baseline_request["cache_key"],
                         "variant_request_sha256": hashes})
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "inputs.jsonl"
    with input_path.open("x", encoding="utf-8") as stream:
        stream.write("".join(inference.canonical(row) + "\n" for row in rows))
    config_files = {}
    for budget, config in variants.items():
        path = output_dir / f"config_b{budget}.json"
        write_frozen(path, config)
        config_files[str(budget)] = {"path": path.name, "sha256": sha(path), "config": config}
    plan = {"version": VERSION, "created_at_utc": inference.utc_now(), "status": "prepared_no_variant_inference",
            "scope": "Exploratory construction follow-up motivated by v2 behavior; not an independent-reference or production-drift claim.",
            "selection_rule": "All descendants of the exact older frozen balanced_mixture pool; no correctness/format filtering.",
            "model_outcomes_read_for_preparation": False, "expected_families": len(family_ids), "expected_rows": len(rows),
            "family_ids": family_ids, "family_domain": family_domain,
            "rows_by_domain": dict(sorted(Counter(domain(row) for row in rows).items())),
            "source_hash_audit": source_check, "source_files": source_hashes,
            "original_selection_manifest": {"path": str(frozen_path), "sha256": sha(frozen_path)},
            "input_file": {"path": input_path.name, "sha256": sha(input_path)},
            "baseline": {"config": baseline, "config_source_path": str(baseline_config_path),
                         "config_source_sha256": sha(baseline_config_path), "output_path": str(baseline_output),
                         "run_metadata_path": str(baseline_output) + ".run.json",
                         "run_metadata_sha256": sha(str(baseline_output) + ".run.json")},
            "variants": config_files, "requests": requests,
            "protocol_checks": {"identical_system_prompt_all_rows": True, "identical_input_all_rows": True,
                                "identical_answer_blind_schema_all_rows": True,
                                "no_special_small_task_shortcuts": True, "baseline_cache_keys_preserved": True},
            "shared_inference_code_sha256": sha(inference.__file__), "wrapper_code_sha256": sha(__file__),
            "execution_rules": ["No cloud calls; local GPU server only.", "Every input retains the same profile, native JSON grammar, seed, temperature, model weights and 8192 context cap.",
                                "Context preflight uses actual tokenizer counts and may reject an oversized request without imputation.",
                                "Parser-format failures are completed model outcomes and never retried for quality.",
                                "Paired family comparisons require both configurations observed for every descendant; missing outcomes stay unknown.",
                                "Output limits are ceilings, not mandatory generation lengths. Latency sums include queueing and are not GPU runtime."]}
    write_frozen(output_dir / "plan.json", plan)
    return plan


def load_prepared(directory):
    plan = json.loads((directory / "plan.json").read_text())
    if plan["version"] != VERSION or sha(__file__) != plan["wrapper_code_sha256"] or sha(inference.__file__) != plan["shared_inference_code_sha256"]:
        raise ValueError("Prepared protocol code changed; explicitly prepare a separate revision")
    if sha(directory / plan["input_file"]["path"]) != plan["input_file"]["sha256"]:
        raise ValueError("Frozen sensitivity inputs changed")
    rows = load_jsonl(directory / plan["input_file"]["path"])
    for budget in BUDGETS:
        entry = plan["variants"][str(budget)]
        if sha(directory / entry["path"]) != entry["sha256"] or json.loads((directory / entry["path"]).read_text()) != variant_config(plan["baseline"]["config"], budget):
            raise ValueError("Variant config changed")
    expected = {request["example_id"]: request for request in plan["requests"]}
    if len(rows) != plan["expected_rows"] or {row["example_id"] for row in rows} != set(expected):
        raise ValueError("Frozen request membership changed")
    for row in rows:
        base = inference.prepare(row, plan["baseline"]["config"])
        if base["cache_key"] != expected[row["example_id"]]["baseline_request_sha256"]:
            raise ValueError("Baseline request hash no longer matches")
        for budget in BUDGETS:
            _, variant = verify_same_task_protocol(row, plan["baseline"]["config"], plan["variants"][str(budget)]["config"])
            if variant["cache_key"] != expected[row["example_id"]]["variant_request_sha256"][str(budget)]:
                raise ValueError("Variant request hash no longer matches")
    return plan, rows


def run_variant(directory, budget, execute=False, workers=4, limit=None):
    plan, rows = load_prepared(directory)
    if budget not in BUDGETS or not 1 <= workers <= 4:
        raise ValueError("Allowed budgets 2048/4096; 1–4 workers")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise ValueError("Invalid row limit")
        rows = rows[:limit]
    config = plan["variants"][str(budget)]["config"]
    items = [inference.prepare(row, config) for row in rows]
    output = directory / f"responses_b{budget}.jsonl"
    journal = inference.RunJournal(output, config, readonly=not execute)
    try:
        for item in items:
            journal.check_identity(item["example_id"], item["cache_key"])
        pending = [item for item in items if item["cache_key"] not in journal.completed]
        summary = {"mode": "execute" if execute else "dry_run", "budget": budget,
                   "expected_rows_full_experiment": plan["expected_rows"], "requested_rows": len(items),
                   "pending_rows": len(pending), **journal.summary()}
        print(inference.canonical(summary), flush=True)
        if not execute:
            return summary
        started, started_at = time.monotonic(), inference.utc_now()
        counts = Counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(inference.infer_one, item, config, journal) for item in pending]
            for number, future in enumerate(as_completed(futures), 1):
                counts[future.result()] += 1
                if number % 20 == 0:
                    print(inference.canonical({"completed_this_invocation": number, "statuses": dict(counts), **journal.summary()}), flush=True)
        terminal = sum(item["cache_key"] in journal.completed for item in items)
        result = {"started_at_utc": started_at, "completed_at_utc": inference.utc_now(),
                  "wall_seconds_this_invocation": time.monotonic() - started, "budget": budget,
                  "requested_rows": len(items), "terminal_rows": terminal,
                  "status": "complete" if terminal == len(items) else "partial", "statuses": dict(counts), **journal.summary()}
        inference.RunJournal.append(directory / f"invocations_b{budget}.jsonl", result)
        print(inference.canonical(result), flush=True)
        return result
    finally:
        journal.close()


def paired_comparison(rows, baseline_responses, variant_responses):
    """Conditional paired summaries without zero-imputing missing outcomes."""
    base_grades = {row["example_id"]: grade_response(row, baseline_responses.get(row["example_id"])) for row in rows}
    new_grades = {row["example_id"]: grade_response(row, variant_responses.get(row["example_id"])) for row in rows}
    result = []
    for group in sorted({domain(row) for row in rows}) + ["balanced_mixture"]:
        selected = [row for row in rows if group == "balanced_mixture" or domain(row) == group]
        families = family_members(selected)
        paired = []
        for family, members in families.items():
            ids = [row["example_id"] for row in members]
            if all(base_grades[i]["observed"] and new_grades[i]["observed"] for i in ids):
                a = mean(int(base_grades[i]["correct"]) for i in ids)
                b = mean(int(new_grades[i]["correct"]) for i in ids)
                paired.append((family, a, b))
        ids = [row["example_id"] for row in selected]
        a_observed, b_observed = sum(base_grades[i]["observed"] for i in ids), sum(new_grades[i]["observed"] for i in ids)
        result.append({"dataset": group, "expected_rows": len(ids), "expected_families": len(families),
                       "baseline_observed_rows": a_observed, "variant_observed_rows": b_observed,
                       "baseline_unknown_rows": len(ids) - a_observed, "variant_unknown_rows": len(ids) - b_observed,
                       "baseline_format_failures": sum(base_grades[i]["observed"] and not base_grades[i]["format_valid"] for i in ids),
                       "variant_format_failures": sum(new_grades[i]["observed"] and not new_grades[i]["format_valid"] for i in ids),
                       "paired_complete_families": len(paired), "paired_family_ids": [p[0] for p in paired],
                       "baseline_accuracy_on_paired_families": mean(p[1] for p in paired) if paired else None,
                       "variant_accuracy_on_paired_families": mean(p[2] for p in paired) if paired else None,
                       "paired_accuracy_delta": mean(p[2] - p[1] for p in paired) if paired else None,
                       "full_population_paired_delta": mean(p[2] - p[1] for p in paired) if paired and len(paired) == len(families) else None,
                       "families_improved": sum(p[2] > p[1] for p in paired), "families_regressed": sum(p[2] < p[1] for p in paired),
                       "interpretation": "Conditional paired family comparison on the fixed construction subset; no missing-outcome imputation or reference-validity claim."})
    return result


def usage_summary(attempts):
    usable = [row for row in attempts if isinstance(row.get("usage"), dict)]
    prompt = [row["usage"]["prompt_tokens"] for row in usable
              if type(row["usage"].get("prompt_tokens")) is int and row["usage"]["prompt_tokens"] >= 0]
    completion = [row["usage"]["completion_tokens"] for row in usable
                  if type(row["usage"].get("completion_tokens")) is int and row["usage"]["completion_tokens"] >= 0]
    return {"attempts": len(attempts), "attempts_with_usage": len(usable), "attempts_missing_usage": len(attempts) - len(usable),
            "reported_prompt_tokens": sum(prompt), "prompt_token_usage_missing_attempts": len(attempts) - len(prompt),
            "reported_completion_tokens": sum(completion), "completion_token_usage_missing_attempts": len(attempts) - len(completion),
            "sum_request_latency_seconds": sum(row.get("latency_seconds", 0) for row in attempts),
            "accounted_api_usd": sum(row.get("charged_nanodollars", 0) for row in attempts) / 1e9,
            "note": "Reported token sums exclude unknown usage; request latencies include queueing and must not be interpreted as GPU runtime or elapsed wall time. Zero API charge is not zero compute cost."}


def analyze_experiment(directory):
    plan, rows = load_prepared(directory)
    expected = {request["example_id"]: request for request in plan["requests"]}
    def read_outputs(path, config, budget=None):
        indexed = {}
        for response in inference.read_jsonl(path):
            identifier = response["example_id"]
            if identifier not in expected:
                if budget is not None:
                    raise ValueError("Variant response outside frozen construction subset")
                continue
            key = expected[identifier]["baseline_request_sha256"] if budget is None else expected[identifier]["variant_request_sha256"][str(budget)]
            if identifier in indexed or response["config_id"] != config["config_id"] or response["cache_key"] != key or response["input_hash"] != expected[identifier]["input_hash"]:
                raise ValueError("Response identity/config/protocol mismatch")
            indexed[identifier] = response
        attempts = [row for row in inference.read_jsonl(Path(str(path) + ".attempts.jsonl")) if row["example_id"] in expected]
        return indexed, usage_summary(attempts)
    baseline_path = Path(plan["baseline"]["output_path"])
    baseline, baseline_cost = read_outputs(baseline_path, plan["baseline"]["config"])
    variants = {}
    for budget in BUDGETS:
        path = directory / f"responses_b{budget}.jsonl"
        responses, cost = read_outputs(path, plan["variants"][str(budget)]["config"], budget)
        variants[str(budget)] = {"paired_comparisons": paired_comparison(rows, baseline, responses),
                                "inference_usage": cost, "response_sha256": sha(path) if path.exists() else None}
    result = {"version": VERSION, "created_at_utc": inference.utc_now(),
              "plan_sha256": sha(directory / "plan.json"), "baseline_response_sha256": sha(baseline_path) if baseline_path.exists() else None,
              "baseline_subset_inference_usage": baseline_cost, "variants": variants,
              "scientific_scope": plan["scope"], "confidence_intervals": None}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--directory", type=Path, default=DEFAULT_BASE / "budget_sensitivity")
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare_experiment(args.base_dir.resolve(), args.directory.resolve())
        print(inference.canonical({"status": result["status"], "families": result["expected_families"], "rows": result["expected_rows"], "directory": str(args.directory)}))
    elif args.mode == "run":
        if args.budget is None:
            parser.error("run requires --budget")
        run_variant(args.directory.resolve(), args.budget, args.execute, args.workers, args.limit)
    else:
        result = analyze_experiment(args.directory.resolve())
        print(inference.canonical({"status": "descriptive_construction_comparison", "variants": list(result["variants"])}))


if __name__ == "__main__":
    main()
