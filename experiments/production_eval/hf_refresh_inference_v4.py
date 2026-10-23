"""Answer-blind, resumable HF inference for the frozen v4 refresh study.

Only inputs_<partition>.jsonl may enter prompt construction. The common contract
owns run/predictor freezes; this runner owns checkpoint loading, raw generations,
and an append-only attempt journal. Incorrect or format-invalid observed outputs
are retained permanently. Infrastructure failures remain unknown outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.structured_decoding_v3 import (
    DomainPrefixFilter, PARAMETERS, prompt_tokens, schema_for_domain, sha, tokenizer_cpu,
)
from experiments.production_eval.trained_structured_decoding_v3 import RUNS, FOLLOWUP, load_trained, verify_followup
from experiments.production_eval.synthetic_sft_v3 import write_json
from experiments.production_eval.synthetic_tasks import canonical, digest
from experiments.production_eval.analyze_sft_v3 import format_diagnostics

BASE = ROOT / "output/research/production_eval/experiments_v4"
INPUT_FIELDS = {"complexity", "domain", "example_id", "family_id", "group_id", "input_hash",
                "input_text", "observable_counts", "partition", "presentation"}
PROVENANCE_FIELDS = INPUT_FIELDS - {"input_text"}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_input(row, partition):
    if set(row) != INPUT_FIELDS:
        raise ValueError("Inference accepts only the exact input allowlist, never gold or programs")
    if row["partition"] != partition or partition not in {"fixture", "construction", "future"}:
        raise ValueError("Input is outside requested partition")
    if row["domain"] not in {"structured_records", "inventory_tracking"} or row["complexity"] not in {"simple", "complex"}:
        raise ValueError("Unknown input domain/complexity")
    if row["group_id"] != row["domain"] + "_" + row["complexity"] or row["presentation"] not in {"A", "B"}:
        raise ValueError("Input group or presentation mismatch")
    if not isinstance(row["input_text"], str) or digest(row["input_text"]) != row["input_hash"]:
        raise ValueError("Input text hash mismatch")
    for key in ("example_id", "family_id", "input_hash"):
        if not isinstance(row[key], str) or len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
            raise ValueError("Malformed input identifier/hash")
    return row


def encode_input(tokenizer, row):
    validate_input(row, row["partition"])
    # Explicitly translate input provenance into the existing frozen prompt API.
    # No answer, group statistics, or model outputs enter the chat text.
    reduced = {key: row[key] for key in ("example_id", "family_id", "input_hash", "input_text", "domain")}
    reduced.update(role=row["partition"], regime=row["complexity"], variant=row["presentation"])
    return prompt_tokens(tokenizer, reduced)


def append_json(path, record):
    value = (canonical(record) + "\n").encode()
    with path.open("ab", buffering=0) as stream:
        count = stream.write(value)
        if count != len(value):
            raise OSError("Incomplete journal write")
        os.fsync(stream.fileno())


def observed_record(row, config_id, run_hash, response, output_ids, prompt, prefix_calls, elapsed, eos_token):
    validate_input(row, row["partition"])
    if not output_ids or prefix_calls != len(output_ids):
        raise ValueError("Every observed output token must pass the schema filter")
    formatting = format_diagnostics(response, row["domain"])
    result = {key: row[key] for key in PROVENANCE_FIELDS}
    result.update(config_id=config_id, status="success", response=response, response_sha256=digest(response),
                  format_status="valid" if formatting["required_schema_valid"] else "invalid",
                  generated_tokens=len(output_ids), output_token_ids=output_ids, prefix_filter_calls=prefix_calls,
                  generation_limit_hit=len(output_ids) >= PARAMETERS["max_new_tokens"] and output_ids[-1] != eos_token,
                  elapsed_seconds=elapsed, prompt_token_sha256=digest(canonical(prompt)),
                  run_manifest_sha256=run_hash, completed_at=now(), **formatting)
    return result


def read_observed(path, expected, config_id, run_hash):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)  # A partial/corrupt line fails closed; never silently truncates a journal.
        identifier = row["example_id"]
        if identifier in rows or identifier not in expected:
            raise ValueError("Duplicate or unexpected observed output")
        if row["config_id"] != config_id or row["run_manifest_sha256"] != run_hash or row["status"] != "success":
            raise ValueError("Observed output belongs to another run or is not an observed generation")
        if any(row[key] != expected[identifier][key] for key in PROVENANCE_FIELDS):
            raise ValueError("Observed input provenance changed")
        formatting = format_diagnostics(row["response"], row["domain"])
        if row["response_sha256"] != digest(row["response"]) or any(row[key] != value for key, value in formatting.items()):
            raise ValueError("Observed response or stored formatting changed")
        if row["format_status"] != ("valid" if formatting["required_schema_valid"] else "invalid"):
            raise ValueError("Observed format status changed")
        if row["generated_tokens"] != len(row["output_token_ids"]) or row["prefix_filter_calls"] != row["generated_tokens"]:
            raise ValueError("Observed token provenance changed")
        rows[identifier] = row
    return rows


def run(run_manifest_path, config_id, partition, future_manifest_path, device):
    import torch
    from experiments.production_eval.refresh_run_contract_v4 import validate_stage
    # This common helper blocks all inference before the run freeze and future
    # inference until all construction outputs and predictors have been frozen.
    contract = validate_stage(run_manifest_path, config_id, partition, future_manifest_path)
    rows = contract["rows"]
    config = contract["config"]
    provenance = contract["provenance"]
    run_hash = provenance["run_manifest_sha256"]
    if not any(Path(entry["path"]).resolve() == Path(__file__).resolve() for entry in provenance["code_files"]):
        raise ValueError("This HF inference runner must itself be frozen")
    if contract["schemas"] != {domain: schema_for_domain(domain) for domain in ("structured_records", "inventory_tracking")}:
        raise ValueError("Dataset schemas differ from the unchanged frozen decoding grammar")
    if config_id not in RUNS or config["config_id"] != config_id:
        raise ValueError("Only the two prespecified trained HF checkpoints are available")
    verify_followup()
    if config["trained_followup_manifest_sha256"] != sha(FOLLOWUP / "followup_manifest.json"):
        raise ValueError("Run freeze points to a different trained-checkpoint manifest")
    for row in rows:
        validate_input(row, partition)
    expected = {row["example_id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate input ID")
    destination = BASE / "inference" / config_id / partition
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = read_observed(destination / "responses.jsonl", expected, config_id, run_hash)
        completion_path = destination / "completed.json"
        if completion_path.exists():
            completed = json.loads(completion_path.read_text())
            if completed.get("status") != "completed" or completed.get("config_id") != config_id or completed.get("partition") != partition or completed.get("evaluated") != len(rows) or completed.get("success_count") != len(rows) or completed["run_manifest_sha256"] != run_hash or completed["responses_sha256"] != sha(destination / "responses.jsonl") or len(observed) != len(rows):
                raise ValueError("Existing completion does not match frozen outputs")
            print(json.dumps({"status": "already_completed", "config_id": config_id, "partition": partition, "count": len(observed)}), flush=True)
            return
        execution_id = str(uuid.uuid4())
        started = {"execution_id": execution_id, "started_at": now(), "config_id": config_id,
                   "partition": partition, "device": device, "run_manifest_sha256": run_hash,
                   "future_predictor_manifest_sha256": provenance["future_gate"]["sha256"] if provenance["future_gate"] else None,
                   "input_file_sha256": provenance["input_sha256"],
                   "existing_observed_outputs": len(observed)}
        append_json(destination / "execution_attempts.jsonl", started)
        tokenizer, tokenizer_data = tokenizer_cpu()
        model, evidence = load_trained(config_id, device)
        append_json(destination / "execution_attempts.jsonl", {"execution_id": execution_id, "event": "model_loaded", "at": now(), "model_evidence": evidence})
        start_all = time.monotonic()
        with torch.inference_mode():
            for row in rows:
                if row["example_id"] in observed:
                    continue
                prompt = encode_input(tokenizer, row)
                ids = torch.tensor([prompt], device=device)
                prefix = DomainPrefixFilter(tokenizer_data, row["domain"], prompt)
                attempt_id = str(uuid.uuid4())
                attempt = {"attempt_id": attempt_id, "execution_id": execution_id, "config_id": config_id,
                           "example_id": row["example_id"], "input_hash": row["input_hash"], "at": now()}
                append_json(destination / "attempts.jsonl", {**attempt, "event": "started"})
                started_generation = time.monotonic()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                   max_new_tokens=PARAMETERS["max_new_tokens"], do_sample=False,
                                                   use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                                   prefix_allowed_tokens_fn=prefix)
                    output_ids = generated[0, len(prompt):].tolist()
                    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    record = observed_record(row, config_id, run_hash, output, output_ids, prompt, prefix.calls,
                                             time.monotonic() - started_generation, tokenizer.eos_token_id)
                    record["attempt_id"] = attempt_id
                    append_json(destination / "responses.jsonl", record)
                    observed[row["example_id"]] = record
                except Exception as error:
                    append_json(destination / "attempts.jsonl", {**attempt, "event": "infrastructure_or_runner_error", "finished_at": now(),
                                                                "error_type": type(error).__name__, "error_message": str(error), "outcome": "unknown"})
                    raise
                append_json(destination / "attempts.jsonl", {**attempt, "event": "observed", "finished_at": now(), "response_sha256": record["response_sha256"]})
                if len(observed) % 10 == 0:
                    print(json.dumps({"config_id": config_id, "partition": partition, "observed": len(observed), "of": len(rows)}), flush=True)
        complete = {"status": "completed", "config_id": config_id, "partition": partition,
                    "evaluated": len(observed), "success_count": len(observed), "completed_at": now(),
                    "run_manifest_sha256": run_hash, "responses_sha256": sha(destination / "responses.jsonl"),
                    "future_predictor_manifest_sha256": started["future_predictor_manifest_sha256"],
                    "model_evidence": evidence, "last_execution_elapsed_seconds": time.monotonic() - start_all,
                    "strict_json_valid": sum(r["strict_json_valid"] for r in observed.values()),
                    "required_schema_valid": sum(r["required_schema_valid"] for r in observed.values()),
                    "generation_limit_hit": sum(r["generation_limit_hit"] for r in observed.values()),
                    "gold_consulted": False, "grading_performed": False}
        write_json(completion_path, complete, exclusive=True)
        print(json.dumps({"status": "completed", "config_id": config_id, "partition": partition, "count": len(observed)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--config", choices=RUNS, required=True)
    parser.add_argument("--partition", choices=("fixture", "construction", "future"), required=True)
    parser.add_argument("--future-predictor-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.run_manifest, args.config, args.partition, args.future_predictor_manifest, args.device)


if __name__ == "__main__":
    main()
