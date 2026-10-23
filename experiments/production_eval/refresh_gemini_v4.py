"""Gold-free v4 Gemini stages with one durable USD1 ledger across all stages."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from experiments.production_eval.api_audit import actual_cost, canonical, digest, response_text, utc_now
from experiments.production_eval.experiment_inference import RunJournal, ContextLimitError, validate_shape
from experiments.production_eval.refresh_run_contract_v4 import BASE, ROOT, INPUT_FIELDS, validate_stage


def validate_config(config):
    expected = {"config_id": "gemini_direct", "provider": "gemini", "model": "gemini-3.1-flash-lite",
                "temperature": 0., "seed": 2026091504, "max_output_tokens": 512,
                "max_context_tokens": 8192, "thinking_level": "minimal", "api_cap_usd": 1.}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Gemini configuration differs from the prespecified v4 recipe")
    if not 10 <= config.get("timeout_seconds", 0) <= 600:
        raise ValueError("Bounded timeout required")


def prepare(row, config, schemas, run_hash):
    validate_config(config)
    if set(row) != INPUT_FIELDS or digest(row["input_text"]) != row["input_hash"]:
        raise ValueError("Input-only allowlist/hash violated")
    schema = schemas[row["domain"]]
    request = {"input_text": row["input_text"], "schema": schema}
    bound = len(row["input_text"].encode()) + len(canonical(schema).encode()) + 512
    reservation = bound * 250 + config["max_output_tokens"] * 1500
    return {"example_id": row["example_id"], "input_hash": row["input_hash"], "row": row,
            "run_manifest_sha256": run_hash,
            "request": request, "cache_key": digest(canonical({"config": config, "request": request, "run": run_hash})),
            "reservation_nanodollars": reservation, "input_token_reservation": bound}


def format_result(text, schema):
    if not isinstance(text, str) or not text.strip():
        return {"strict_json_valid": False, "required_schema_valid": False, "format_status": "no_text"}
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("Nonfinite constant")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, TypeError):
        return {"strict_json_valid": False, "required_schema_valid": False, "format_status": "invalid_json"}
    valid = validate_shape(value, schema)
    return {"strict_json_valid": True, "required_schema_valid": valid, "format_status": "valid" if valid else "schema_invalid"}


def transport(item, config, api_key):
    from google import genai
    from google.genai import types
    request = item["request"]
    options = types.HttpOptions(timeout=int(config["timeout_seconds"] * 1000),
                                retry_options=types.HttpRetryOptions(attempts=1))
    with genai.Client(api_key=api_key, vertexai=False, http_options=options) as client:
        counted = client.models.count_tokens(model=config["model"], contents=request["input_text"])
        slack = len(canonical(request["schema"]).encode()) + 64
        if counted.total_tokens + slack + config["max_output_tokens"] > config["max_context_tokens"]:
            raise ContextLimitError()
        response = client.models.generate_content(model=config["model"], contents=request["input_text"],
            config=types.GenerateContentConfig(temperature=0., seed=config["seed"], candidate_count=1,
                max_output_tokens=config["max_output_tokens"], response_mime_type="application/json",
                response_json_schema=request["schema"],
                thinking_config=types.ThinkingConfig(thinking_level="minimal", include_thoughts=False)))
    return {"raw_response": response.model_dump(mode="json", exclude={"sdk_http_response"}),
            "response": response_text(response),
            "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else None,
            "actual_model_version": response.model_version,
            "finish_reasons": [str(c.finish_reason.value if hasattr(c.finish_reason, "value") else c.finish_reason) for c in response.candidates or []],
            "preflight_prompt_tokens": counted.total_tokens, "preflight_schema_slack_tokens": slack}


def infer_one(item, config, journal, api_key=None, call=transport):
    if item["cache_key"] in journal.completed:
        return "cached"
    while True:
        reserved = journal.reserve(item)
        if reserved is None:
            return "budget_or_attempt_limit"
        call_id, attempt = reserved
        row = item["row"]
        result = {key: row[key] for key in INPUT_FIELDS if key != "input_text"}
        result.update(config_id=config["config_id"], cache_key=item["cache_key"], call_id=call_id, attempt=attempt,
                      run_manifest_sha256=item["run_manifest_sha256"],
                      started_at_utc=utc_now(), status="api_error", response=None, raw_response=None, usage=None,
                      response_schema_sha256=digest(canonical(item["request"]["schema"])),
                      reservation_nanodollars=item["reservation_nanodollars"])
        start, retryable = time.monotonic(), True
        try:
            result.update(call(item, config, api_key))
            result.update(format_result(result["response"], item["request"]["schema"]))
            result["response_sha256"] = digest(result["response"]) if isinstance(result["response"], str) else None
            result["raw_response_sha256"] = digest(canonical(result["raw_response"]))
            result["status"] = "success"
            retryable = False  # Invalid generated formats are definitive outcomes.
        except Exception as error:
            result["error_class"] = type(error).__name__
            code = getattr(error, "code", None) or getattr(getattr(error, "response", None), "status_code", None)
            if type(code) is int:
                result["http_status_code"] = code
                retryable = code in {408, 429} or code >= 500
            if isinstance(error, ContextLimitError):
                result["status"], retryable = "context_limit", False
        amount, basis = actual_cost(result["usage"], item["reservation_nanodollars"])
        if result["status"] == "context_limit":
            amount, basis = 0, "no_generation_context_preflight_rejected"
        result.update(completed_at_utc=utc_now(), latency_seconds=time.monotonic()-start,
                      charged_nanodollars=amount, accounting_basis=basis,
                      reservation_exceeded=amount > item["reservation_nanodollars"], terminal=not retryable or attempt >= 2)
        journal.finish(result)
        if result["terminal"] or result["reservation_exceeded"]:
            return result["status"]


def immutable_json(path, value):
    payload = canonical(value) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError("Existing stage summary differs; refusing overwrite")
    else:
        with path.open("x") as stream:
            stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("fixture", "construction", "future"))
    parser.add_argument("--run-manifest", type=Path, default=BASE / "run_manifest.json")
    parser.add_argument("--future-predictor-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers")
    checked = validate_stage(args.run_manifest, "gemini_direct", args.partition, args.future_predictor_manifest)
    if not any(Path(entry["path"]).resolve() == Path(__file__).resolve() for entry in checked["provenance"]["code_files"]):
        raise ValueError("The Gemini inference runner must itself be frozen")
    config = checked["config"]; validate_config(config)
    run_hash = checked["provenance"]["run_manifest_sha256"]
    items = [prepare(row, config, checked["schemas"], run_hash) for row in checked["rows"]]
    if args.dry_run:
        print(json.dumps({"mode": "dry_run", "partition": args.partition, "requests": len(items),
                          "maximum_single_attempt_reservation_usd": max(i["reservation_nanodollars"] for i in items)/1e9,
                          "sum_worst_case_reservations_usd": sum(i["reservation_nanodollars"] for i in items)/1e9,
                          "cap_usd_across_all_stages": 1., "network_calls": 0}))
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini credentials unavailable")
    directory = BASE / "inference/gemini_direct"
    journal_config = {**config, "run_manifest_sha256": run_hash, "runner_version": "refresh_gemini_v4"}
    journal = RunJournal(directory / "all_responses.jsonl", journal_config, budget_nanodollars=1_000_000_000)
    try:
        stage = directory / args.partition; stage.mkdir(exist_ok=True)
        pending = [item for item in items if item["cache_key"] not in journal.completed]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, status in enumerate(pool.map(lambda item: infer_one(item, config, journal, api_key), pending), 1):
                if index % 20 == 0:
                    print(json.dumps({"partition": args.partition, "processed_pending": index, "of": len(pending), **journal.summary()}), flush=True)
        missing = [item["example_id"] for item in items if item["cache_key"] not in journal.completed]
        if missing:
            raise RuntimeError(f"Stage remains incomplete: {len(missing)} requests; preserve ledger and inspect budget/attempt state")
        responses = [journal.completed[item["cache_key"]] for item in items]
        content = "".join(canonical(row)+"\n" for row in responses)
        path = stage / "responses.jsonl"
        if path.exists() and path.read_text() != content:
            raise ValueError("Stage snapshot differs from original completed responses")
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)
        unknown = [r["example_id"] for r in responses if r["status"] != "success"]
        if unknown:
            immutable_json(stage / "incomplete_outcomes.json", {"status": "incomplete_unknown_outcomes",
                           "partition": args.partition, "config_id": config["config_id"],
                           "terminal_requests": len(responses), "observed_generations": len(responses)-len(unknown),
                           "unknown_example_ids": unknown, "responses_sha256": digest(content),
                           "provenance": checked["provenance"]})
            raise RuntimeError(f"Stage retains {len(unknown)} unknown outcomes; no completion or future inference allowed")
        immutable_json(stage / "completed.json", {"status": "completed", "partition": args.partition,
                       "config_id": config["config_id"], "rows": len(responses),
                       "observed_generations": sum(r["status"] == "success" for r in responses),
                       "responses_sha256": digest(content), "provenance": checked["provenance"]})
        print(json.dumps({"partition": args.partition, "rows": len(responses), **journal.summary()}), flush=True)
    finally:
        journal.close()


if __name__ == "__main__":
    main()
