"""Isolated identical-budget control; keeps original and budget-study files intact."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.budget_sensitivity import DEFAULT_BASE, load_prepared, sha, write_frozen

VERSION = "identical_1024_construction_repeat_v1"


def repeat_config(baseline, repeat):
    if repeat not in (1, 2) or baseline["provider"] != "local" or baseline["profile"] != "reason_then_answer" or baseline["max_output_tokens"] != 1024:
        raise ValueError("Only identical local work-profile 1024 repeats 1/2")
    return {**baseline, "config_id": f"{baseline['config_id']}_repeat1024_{repeat}",
            "repeat_control_revision": VERSION}


def verify_protocol(row, baseline, control):
    differences = {key for key in baseline.keys() | control.keys() if baseline.get(key) != control.get(key)}
    if differences != {"config_id", "repeat_control_revision"}:
        raise ValueError("Identical-budget control changes only cache bookkeeping")
    a, b = inference.prepare(row, baseline), inference.prepare(row, control)
    for key in ("system_prompt", "input_text", "schema"):
        if a["request"][key] != b["request"][key]:
            raise ValueError("Control changed model-visible input")
    return a, b


def prepare_control(study, repeat, rationale):
    study_plan, rows = load_prepared(study)
    baseline = study_plan["baseline"]["config"]
    config = repeat_config(baseline, repeat)
    directory = study / f"repeat_control_{repeat}"
    directory.mkdir(exist_ok=True)
    expected = {item["example_id"]: item for item in study_plan["requests"]}
    bindings = []
    for row in rows:
        a, b = verify_protocol(row, baseline, config)
        if a["cache_key"] != expected[row["example_id"]]["baseline_request_sha256"]:
            raise ValueError("Original baseline hash changed")
        bindings.append({"example_id": row["example_id"], "input_hash": row["input_hash"],
                         "baseline_request_sha256": a["cache_key"], "repeat_request_sha256": b["cache_key"]})
    baseline_path = Path(study_plan["baseline"]["output_path"])
    plan = {"version": VERSION, "created_at_utc": inference.utc_now(), "repeat": repeat,
            "status": "frozen_before_control_calls", "scope": "Post hoc construction diagnostic, no independent-reference or production claim.",
            "rationale": rationale, "study_directory": str(study), "study_plan_sha256": sha(study / "plan.json"),
            "input_file_sha256": study_plan["input_file"]["sha256"], "expected_rows": len(rows),
            "expected_families": study_plan["expected_families"], "config": config, "workers": 4,
            "baseline_output_path": str(baseline_path), "baseline_output_sha256": sha(baseline_path),
            "baseline_attempts_sha256": sha(str(baseline_path) + ".attempts.jsonl"),
            "control_code_sha256": sha(__file__), "requests": bindings,
            "rules": ["Exactly the older frozen 100-family/250-descendant construction pool, without outcome-based exclusions.",
                      "Same 1024 ceiling, model, work prompt, answer-blind schema, seed, native grammar, temperature, workers and context limit.",
                      "Only config_id/repeat_control_revision cache bookkeeping differs; neither field is sent to the model endpoint.",
                      "Keep first completed model outcome, including malformed/parser-invalid results; only true transport errors may retry.",
                      "max_tokens is a generation parameter, not model-visible prompt text. Differences when neither ceiling binds do not establish a budget effect.",
                      "Compare correctness and canonical prediction agreement separately from raw-text and retokenized-output prefix hashes.",
                      "Unknown outcomes and unknown token usage remain explicit; both-stop/<1024 is a conditional diagnostic subset, not a primary evaluation filter.",
                      "A second repeat may be added adaptively after examining this control, with a separate frozen rationale before calls."]}
    write_frozen(directory / "amendment.json", plan)
    return plan


def load_control(directory):
    plan = json.loads((directory / "amendment.json").read_text())
    if plan["version"] != VERSION or sha(__file__) != plan["control_code_sha256"]:
        raise ValueError("Frozen control code changed")
    study = Path(plan["study_directory"])
    if sha(study / "plan.json") != plan["study_plan_sha256"]:
        raise ValueError("Original sensitivity plan changed")
    study_plan, rows = load_prepared(study)
    if study_plan["input_file"]["sha256"] != plan["input_file_sha256"] or len(rows) != plan["expected_rows"]:
        raise ValueError("Frozen inputs changed")
    if sha(plan["baseline_output_path"]) != plan["baseline_output_sha256"] or sha(plan["baseline_output_path"] + ".attempts.jsonl") != plan["baseline_attempts_sha256"]:
        raise ValueError("Original baseline or attempts changed")
    if plan["config"] != repeat_config(study_plan["baseline"]["config"], plan["repeat"]):
        raise ValueError("Control runtime config changed")
    expected = {item["example_id"]: item for item in plan["requests"]}
    if len(expected) != len(rows) or {row["example_id"] for row in rows} != set(expected):
        raise ValueError("Control membership changed")
    for row in rows:
        a, b = verify_protocol(row, study_plan["baseline"]["config"], plan["config"])
        binding = expected[row["example_id"]]
        if a["cache_key"] != binding["baseline_request_sha256"] or b["cache_key"] != binding["repeat_request_sha256"] or row["input_hash"] != binding["input_hash"]:
            raise ValueError("Control request binding changed")
    return plan, rows


def run_control(directory, execute=False):
    plan, rows = load_control(directory)
    config = plan["config"]
    items = [inference.prepare(row, config) for row in rows]
    journal = inference.RunJournal(directory / "responses.jsonl", config, readonly=not execute)
    try:
        for item in items:
            journal.check_identity(item["example_id"], item["cache_key"])
        pending = [item for item in items if item["cache_key"] not in journal.completed]
        summary = {"mode": "execute" if execute else "dry_run", "repeat": plan["repeat"],
                   "requested_rows": len(rows), "pending_rows": len(pending), **journal.summary()}
        print(inference.canonical(summary), flush=True)
        if not execute:
            return summary
        started, started_at, counts = time.monotonic(), inference.utc_now(), Counter()
        with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
            futures = [pool.submit(inference.infer_one, item, config, journal) for item in pending]
            for number, future in enumerate(as_completed(futures), 1):
                counts[future.result()] += 1
                if number % 25 == 0:
                    print(inference.canonical({"completed_this_invocation": number, "statuses": dict(counts), **journal.summary()}), flush=True)
        terminal = sum(item["cache_key"] in journal.completed for item in items)
        result = {"started_at_utc": started_at, "completed_at_utc": inference.utc_now(),
                  "wall_seconds_this_invocation": time.monotonic() - started,
                  "requested_rows": len(items), "terminal_rows": terminal,
                  "status": "complete" if terminal == len(items) else "partial", "statuses": dict(counts), **journal.summary()}
        inference.RunJournal.append(directory / "invocations.jsonl", result)
        print(inference.canonical(result), flush=True)
        return result
    finally:
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--study", type=Path, default=DEFAULT_BASE / "budget_sensitivity")
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rationale")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.rationale:
            parser.error("prepare requires an explicit rationale")
        plan = prepare_control(args.study.resolve(), args.repeat, args.rationale)
        print(inference.canonical({"status": plan["status"], "repeat": plan["repeat"], "expected_rows": plan["expected_rows"]}))
    else:
        run_control(args.study.resolve() / f"repeat_control_{args.repeat}", args.execute)
