"""Bounded local/Gemini inference with answer-blind schemas and durable resume.

The primary JSONL has one terminal row per request; every attempt is retained in
`.attempts.jsonl` and reserved before transport in `.ledger.jsonl`. Invalid model
formats are completed outcomes and are never retried to obtain a better answer.
This runner does not grade predictions. --dry-run uses no network or credentials.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse
import uuid

try:
    from .api_audit import (actual_cost, canonical, digest, read_jsonl, response_text,
                            utc_now, MODEL, PRICING)
except ImportError:
    from api_audit import (actual_cost, canonical, digest, read_jsonl, response_text,
                           utc_now, MODEL, PRICING)

ROOT = Path(__file__).resolve().parents[2]
VERSION = "answer_blind_inference_transport_v2"
LOCAL_TRANSPORT_REVISION = "llama_native_json_schema_parser_outcomes_v4"
DEFAULTS = {"temperature": 0.0, "seed": 20260914, "max_output_tokens": 1024,
            "max_context_tokens": 8192, "timeout_seconds": 180,
            "profile": "direct", "base_url": "http://127.0.0.1:8123"}
FIXED_ANSWER_SCHEMAS = {
    "structured_records": {"type": "object", "additionalProperties": False,
                           "properties": {"selected_ids": {"type": "array", "items": {"type": "string"}},
                                          "total_units": {"type": "integer"}, "total_value": {"type": "integer"}},
                           "required": ["selected_ids", "total_units", "total_value"]},
    "inventory_tracking": {"type": "object", "additionalProperties": False,
                           "properties": {"stock": {"type": "array", "items": {
                               "type": "object", "additionalProperties": False,
                               "properties": {"item": {"type": "string"}, "east": {"type": "integer"}, "west": {"type": "integer"}},
                               "required": ["item", "east", "west"]}}, "total_units": {"type": "integer"}},
                           "required": ["stock", "total_units"]},
}


def load_config(path):
    source = json.loads(Path(path).read_text())
    if not isinstance(source, dict):
        raise ValueError("Configuration must be an object")
    unknown = set(source) - set(DEFAULTS) - {"config_id", "provider", "model", "model_revision"}
    if unknown:
        raise ValueError("Unknown configuration fields")
    config = {**DEFAULTS, **source, "runner_version": VERSION}
    for key in ("config_id", "model", "model_revision"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError("Configuration needs config_id, model and model_revision")
    if config.get("provider") not in {"local", "gemini"} or config["profile"] not in {"direct", "reason_then_answer"}:
        raise ValueError("Invalid provider/profile")
    if config["temperature"] != 0 or type(config["seed"]) is not int:
        raise ValueError("This experiment freezes temperature 0 and an integer seed")
    if config["max_output_tokens"] != 1024 or config["max_context_tokens"] != 8192:
        raise ValueError("This experiment freezes 1024 output and 8192 context tokens")
    if type(config["timeout_seconds"]) not in {int, float} or not 10 <= config["timeout_seconds"] <= 600:
        raise ValueError("Timeout must be 10–600 seconds")
    if config["provider"] == "gemini" and config["model"] != MODEL:
        raise ValueError("Verified prices are only for Gemini 3.1 Flash Lite")
    if config["provider"] == "local":
        parsed = urlparse(config["base_url"])
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != 8123 or parsed.path not in {"", "/"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Local endpoint must be http://127.0.0.1:8123")
        # Provider-specific revision preserves the Gemini v2 cache contract while
        # invalidating all local requests using the broken response_format route.
        config["local_transport_revision"] = LOCAL_TRANSPORT_REVISION
    return config


def schema_for(answer_type, family, profile):
    if answer_type == "json":
        if family not in FIXED_ANSWER_SCHEMAS:
            raise ValueError("JSON tasks need a known fixed schema")
        answer_schema = FIXED_ANSWER_SCHEMAS[family]
    elif answer_type in {"decimal", "boolean"}:
        answer_schema = {"type": "string"}
    else:
        raise ValueError("Unsupported answer type")
    properties = {"work": {"type": "string"}} if profile == "reason_then_answer" else {}
    properties["answer"] = answer_schema
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}


def prepare(row, config):
    # Explicit allowlist: never read answer, source metadata, or example IDs into
    # any model message, schema constraint or generation setting.
    text = row["input_text"]
    if not isinstance(text, str) or not text.strip() or row["input_hash"] != digest(text):
        raise ValueError("Input text/hash mismatch")
    schema = schema_for(row["answer_type"], row["task_family"], config["profile"])
    system = ("Solve the task in the user message using only the supplied information. "
              "Return exactly one JSON object matching the response schema, with no markdown. "
              "The outer object wraps your final result in the answer field. If the task itself "
              "requests a JSON object, put that complete object inside answer. For arithmetic "
              "return only the final number as a string in answer; for Boolean reasoning return "
              "the string True or False in answer. Do not call tools or execute code. ")
    system += ("Write your calculation or reasoning in the work string first, then provide answer."
               if config["profile"] == "reason_then_answer" else "Provide answer directly; do not add a work field.")
    request = {"config": config, "system_prompt": system, "input_text": text, "schema": schema}
    key = digest(canonical(request))
    input_bound = len(canonical(request).encode("utf-8")) + 1024
    reservation = input_bound * 250 + config["max_output_tokens"] * 1500 if config["provider"] == "gemini" else 0
    return {"example_id": row["example_id"], "input_hash": row["input_hash"],
            "answer_type": row["answer_type"], "task_family": row["task_family"],
            "cache_key": key, "request_sha256": key, "request": request,
            "input_token_reservation": input_bound, "reservation_nanodollars": reservation}


def validate_shape(value, schema):
    kind = schema["type"]
    if kind == "object":
        return (type(value) is dict and set(value) == set(schema["required"])
                and all(validate_shape(value[key], subschema) for key, subschema in schema["properties"].items()))
    if kind == "array":
        return type(value) is list and all(validate_shape(v, schema["items"]) for v in value)
    return {"string": str, "integer": int, "boolean": bool}.get(kind) is type(value)


def parse_output(text, schema):
    if not isinstance(text, str) or not text.strip():
        return None, None, "no_text"
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("non-finite number")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError):
        return None, None, "invalid_json"
    if not validate_shape(value, schema):
        return None, None, "schema_invalid"
    return value["answer"], value.get("work"), "valid"


class RunJournal:
    """One output/config, durable per-attempt money/compute reservations."""
    def __init__(self, output, config, budget_nanodollars=5_000_000_000, readonly=False):
        self.output, self.config, self.budget = Path(output), config, budget_nanodollars
        self.ledger = self.output.with_suffix(self.output.suffix + ".ledger.jsonl")
        self.attempt_file = self.output.with_suffix(self.output.suffix + ".attempts.jsonl")
        self.meta = self.output.with_suffix(self.output.suffix + ".run.json")
        self.mutex, self.lock_stream, self.readonly = threading.Lock(), None, readonly
        if not readonly:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.lock_stream = os.fdopen(os.open(str(self.output) + ".lock", os.O_RDWR | os.O_CREAT, 0o600), "a")
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_hash = digest(canonical(config))
        if self.meta.exists():
            if json.loads(self.meta.read_text())["config_sha256"] != config_hash:
                raise ValueError("Output belongs to a different configuration")
        elif any(path.exists() for path in (self.output, self.ledger, self.attempt_file)):
            raise ValueError("Existing output requires its original run metadata")
        elif not readonly:
            with os.fdopen(os.open(self.meta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
                stream.write(canonical({"config": config, "config_sha256": config_hash,
                                        "created_at_utc": utc_now(), "budget_nanodollars": budget_nanodollars}) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        self.attempts, self.charges, self.call_keys, self.completed = {}, {}, {}, {}
        self.identity_keys, self.key_identities = {}, {}
        self.settled, self.bound_exceeded = set(), False
        for event in read_jsonl(self.ledger):
            call, key = event["call_id"], event["cache_key"]
            if event["event"] == "reserve":
                amount = event["reservation_nanodollars"]
                if call in self.charges or type(amount) is not int or amount < 0:
                    raise ValueError("Invalid reservation ledger")
                self.charges[call], self.call_keys[call] = amount, key
                self.check_identity(event["example_id"], key)
                self.identity_keys[event["example_id"]] = key
                self.key_identities[key] = event["example_id"]
                self.attempts[key] = self.attempts.get(key, 0) + 1
            elif event["event"] == "settle":
                amount = event["charged_nanodollars"]
                if call not in self.charges or call in self.settled or self.call_keys[call] != key or type(amount) is not int or amount < 0:
                    raise ValueError("Invalid settlement ledger")
                self.charges[call] = amount
                self.settled.add(call)
            else:
                raise ValueError("Unknown ledger event")
        attempt_results = {}
        for result in read_jsonl(self.attempt_file):
            call = result["call_id"]
            if call not in self.charges or call in attempt_results or result["cache_key"] != self.call_keys[call]:
                raise ValueError("Invalid attempt provenance")
            if call in self.settled and result["charged_nanodollars"] != self.charges[call]:
                raise ValueError("Attempt charge disagrees with ledger")
            attempt_results[call] = result
            self.bound_exceeded |= bool(result.get("reservation_exceeded"))
        if self.settled - set(attempt_results):
            raise ValueError("Settlements missing attempt records")
        for result in read_jsonl(self.output):
            key = result["cache_key"]
            if key in self.completed or result["call_id"] not in attempt_results or result != attempt_results[result["call_id"]]:
                raise ValueError("Invalid or duplicate terminal output")
            self.completed[key] = result
        # Recover a crash after durable terminal attempt but before output append.
        # Unsettled reservations stay fully charged; no uncertain spend is erased.
        for result in attempt_results.values():
            key = result["cache_key"]
            if result["terminal"] and key not in self.completed:
                if not readonly:
                    self.append(self.output, result)
                self.completed[key] = result

    @staticmethod
    def append(path, value):
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush(); os.fsync(stream.fileno())

    def check_identity(self, identifier, key):
        if identifier in self.identity_keys and self.identity_keys[identifier] != key:
            raise ValueError("Example ID changed its full request within this output")
        if key in self.key_identities and self.key_identities[key] != identifier:
            raise ValueError("A cached full request cannot silently move to another example ID")

    def reserve(self, item):
        with self.mutex:
            key, amount = item["cache_key"], item["reservation_nanodollars"]
            self.check_identity(item["example_id"], key)
            if self.bound_exceeded or key in self.completed or self.attempts.get(key, 0) >= 2:
                return None
            if sum(self.charges.values()) + amount > self.budget:
                return None
            call, attempt = uuid.uuid4().hex, self.attempts.get(key, 0) + 1
            event = {"event": "reserve", "at_utc": utc_now(), "call_id": call,
                     "cache_key": key, "attempt": attempt, "reservation_nanodollars": amount,
                     "example_id": item["example_id"],
                     "compute_request_reservation": 1, "provider": self.config["provider"]}
            if self.config["provider"] == "gemini":
                event["pricing"] = PRICING
            self.append(self.ledger, event)
            self.attempts[key], self.charges[call], self.call_keys[call] = attempt, amount, key
            self.identity_keys[item["example_id"]], self.key_identities[key] = key, item["example_id"]
            return call, attempt

    def finish(self, result):
        with self.mutex:
            self.append(self.attempt_file, result)
            if result["terminal"]:
                self.append(self.output, result)
                self.completed[result["cache_key"]] = result
            self.append(self.ledger, {"event": "settle", "at_utc": utc_now(),
                                     "call_id": result["call_id"], "cache_key": result["cache_key"],
                                     "charged_nanodollars": result["charged_nanodollars"],
                                     "accounting_basis": result["accounting_basis"]})
            self.charges[result["call_id"]] = result["charged_nanodollars"]
            self.settled.add(result["call_id"])
            self.bound_exceeded |= result["reservation_exceeded"]

    def summary(self):
        with self.mutex:
            return {"ledger_accounted_usd": sum(self.charges.values()) / 1e9,
                    "budget_remaining_usd": max(0, self.budget - sum(self.charges.values())) / 1e9,
                    "reserved_compute_attempts": sum(self.attempts.values()),
                    "unsettled_attempts": len(set(self.charges) - self.settled),
                    "terminal_cached_requests": len(self.completed), "reservation_bound_exceeded": self.bound_exceeded}

    def close(self):
        if self.lock_stream:
            self.lock_stream.close()


class ContextLimitError(Exception):
    pass


def recover_local_output_parse_error(status_code, raw):
    """Identify the local server's post-generation parser failure, not HTTP500s.

    server-task.cpp passes generated_text to common_chat_parse after inference.
    The pinned runtime's error includes that text. Missing token/finish metadata
    remains unknown; a parser error alone does not prove token-limit truncation.
    """
    if status_code != 500 or not isinstance(raw, dict) or not isinstance(raw.get("error"), dict):
        return None
    error = raw["error"]
    if error.get("type") != "server_error" or not isinstance(error.get("message"), str):
        return None
    match = re.fullmatch(r"Failed to parse input at pos ([0-9]+): ([\s\S]*)", error["message"])
    if not match:
        return None
    return {"raw_response": raw, "raw_output_text": match.group(2), "usage": None,
            "finish_reason": None, "actual_model_version": None, "http_status_code": 500,
            "provider_output_parse_error": True, "parser_failure_position": int(match.group(1)),
            "response_recovery": "post_generation_server_parser_error_payload",
            "generation_occurred": True}


def local_call(item, config):
    import requests
    request = item["request"]
    messages = [{"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": request["input_text"]}]
    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        def post(endpoint, body):
            response = session.post(config["base_url"].rstrip("/") + endpoint, json=body,
                                    timeout=(10, config["timeout_seconds"]))
            if endpoint == "/v1/chat/completions" and response.status_code == 500:
                try:
                    recovered = recover_local_output_parse_error(response.status_code, response.json())
                except (ValueError, TypeError):
                    recovered = None
                if recovered is not None:
                    return recovered
            response.raise_for_status()
            return response.json()
        # Non-generating preflight prevents silent long-prompt truncation. The
        # extra 64 tokens cover small endpoint/template framing differences.
        template = post("/apply-template", {"messages": messages,
                                             "chat_template_kwargs": {"enable_thinking": False}})
        tokens = post("/tokenize", {"content": template["prompt"], "add_special": False,
                                     "parse_special": True})["tokens"]
        if len(tokens) + 64 + config["max_output_tokens"] > config["max_context_tokens"]:
            raise ContextLimitError()
        body = {"model": config["model"], "messages": messages, "temperature": 0.0,
                "seed": config["seed"], "max_tokens": config["max_output_tokens"], "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                # On pinned llama.cpp 2f2923f, response_format configures a chat
                # parser but does not activate the generation constraint. The
                # native json_schema field activates the schema/grammar handler.
                # Verified on all eight smoke requests plus a forced-const probe.
                "json_schema": request["schema"]}
        raw = post("/v1/chat/completions", body)
    if raw.get("provider_output_parse_error") is True:
        return {**raw, "context_preflight_prompt_tokens": len(tokens),
                "context_preflight_slack_tokens": 64}
    choice = (raw.get("choices") or [{}])[0]
    text = choice.get("message", {}).get("content")
    return {"raw_response": raw, "raw_output_text": text, "usage": raw.get("usage"),
            "finish_reason": choice.get("finish_reason"), "actual_model_version": raw.get("model"),
            "context_preflight_prompt_tokens": len(tokens), "context_preflight_slack_tokens": 64}


def gemini_call(item, config, api_key):
    from google import genai
    from google.genai import types
    request = item["request"]
    options = types.HttpOptions(timeout=int(config["timeout_seconds"] * 1000),
                                retry_options=types.HttpRetryOptions(attempts=1))
    with genai.Client(api_key=api_key, vertexai=False, http_options=options) as client:
        # Gemini Developer API rejects CountTokensConfig.system_instruction.
        # Count the combined system/user text as contents, then add framing and
        # schema slack. Generation still sends the actual separate system turn.
        count_contents = request["system_prompt"] + "\n\n" + request["input_text"]
        counted = client.models.count_tokens(model=config["model"], contents=count_contents)
        schema_slack = len(canonical(request["schema"]).encode("utf-8")) + 64
        if counted.total_tokens + schema_slack + config["max_output_tokens"] > config["max_context_tokens"]:
            raise ContextLimitError()
        response = client.models.generate_content(model=config["model"], contents=request["input_text"],
            config=types.GenerateContentConfig(system_instruction=request["system_prompt"], temperature=0.0,
                seed=config["seed"], candidate_count=1, max_output_tokens=config["max_output_tokens"],
                response_mime_type="application/json", response_json_schema=request["schema"],
                thinking_config=types.ThinkingConfig(thinking_level="minimal", include_thoughts=False)))
    raw = response.model_dump(mode="json", exclude={"sdk_http_response"})
    reasons = [str(c.finish_reason.value if hasattr(c.finish_reason, "value") else c.finish_reason)
               for c in response.candidates or []]
    return {"raw_response": raw, "raw_output_text": response_text(response),
            "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else None,
            "finish_reason": reasons, "actual_model_version": response.model_version,
            "context_preflight_prompt_tokens": counted.total_tokens,
            "context_preflight_slack_tokens": schema_slack}


def infer_one(item, config, journal, api_key=None, transport=None):
    if item["cache_key"] in journal.completed:
        return "cached"
    while True:
        reservation = journal.reserve(item)
        if reservation is None:
            return "budget_or_attempt_limit"
        call, attempt = reservation
        result = {"example_id": item["example_id"], "input_hash": item["input_hash"],
                  "answer_type": item["answer_type"], "task_family": item["task_family"],
                  "config_id": config["config_id"], "config_sha256": digest(canonical(config)),
                  "provider": config["provider"], "requested_model": config["model"],
                  "requested_model_revision": config["model_revision"],
                  "cache_key": item["cache_key"], "request_sha256": item["cache_key"],
                  "system_prompt_sha256": digest(item["request"]["system_prompt"]),
                  "response_schema_sha256": digest(canonical(item["request"]["schema"])),
                  "call_id": call, "attempt": attempt, "started_at_utc": utc_now(),
                  "answer": None, "work": None, "format_status": "not_received", "status": "api_error",
                  "raw_response": None, "raw_output_text": None, "response_sha256": None,
                  "usage": None, "finish_reason": None, "actual_model_version": None,
                  "reservation_nanodollars": item["reservation_nanodollars"]}
        start, retryable = time.monotonic(), True
        try:
            received = (transport(item, config) if transport else local_call(item, config)
                        if config["provider"] == "local" else gemini_call(item, config, api_key))
            result.update(received)
            result["response_sha256"] = digest(canonical(result["raw_response"]))
            if result.get("provider_output_parse_error") is True:
                result["answer"], result["work"], result["format_status"] = None, None, "server_parser_invalid"
            else:
                result["answer"], result["work"], result["format_status"] = parse_output(result["raw_output_text"], item["request"]["schema"])
            result["status"] = "success"
            retryable = False  # A completed malformed answer is an outcome.
        except Exception as error:
            result["error_class"] = type(error).__name__
            response = getattr(error, "response", None)
            code = getattr(error, "code", None) or getattr(response, "status_code", None)
            if type(code) is int:
                result["http_status_code"] = code
                retryable = code in {408, 429} or code >= 500
            if isinstance(error, ContextLimitError):
                result["status"], retryable = "context_limit", False
        result["completed_at_utc"] = utc_now()
        result["latency_seconds"] = round(time.monotonic() - start, 4)
        if config["provider"] == "gemini":
            amount, basis = actual_cost(result["usage"], item["reservation_nanodollars"])
            if result["status"] == "context_limit":
                amount, basis = 0, "no_generation_context_preflight_rejected"
        else:
            amount, basis = 0, "local_compute_no_api_charge"
        result.update(charged_nanodollars=amount, accounting_basis=basis,
                      reservation_exceeded=amount > item["reservation_nanodollars"],
                      terminal=not retryable or attempt >= 2)
        journal.finish(result)
        if result["terminal"] or result["reservation_exceeded"]:
            return result["status"]


def read_inputs(path, limit, roles=(), splits=()):
    result, seen = [], set()
    for row in read_jsonl(path):
        if roles and row.get("data_role") not in roles or splits and row.get("split") not in splits:
            continue
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Missing or duplicate example ID")
        seen.add(identifier); result.append(row)
        if len(result) >= limit:
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--roles", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100000 or not 1 <= args.workers <= 4 or not 0 < args.max_cost_usd <= 5:
        parser.error("Use 1–100000 rows, 1–4 workers and a positive cap no greater than USD 5")
    if args.output.resolve() in {args.input.resolve(), args.config.resolve()}:
        parser.error("Output must differ from input/config")
    logging.disable(logging.CRITICAL)
    journal = None
    try:
        config = load_config(args.config)
        rows = read_inputs(args.input, args.limit, args.roles, args.splits)
        items = [prepare(row, config) for row in rows]
        if len({i["cache_key"] for i in items}) != len(items):
            raise ValueError("Identical full requests under different IDs require deduplication before inference")
        journal = RunJournal(args.output, config, int(args.max_cost_usd * 1e9), readonly=args.dry_run)
        for item in items:
            journal.check_identity(item["example_id"], item["cache_key"])
        pending = [item for item in items if item["cache_key"] not in journal.completed]
        print(canonical({"mode": "dry_run" if args.dry_run else "inference", "config_id": config["config_id"],
                         "requested_rows": len(items), "pending_rows": len(pending),
                         "worst_remaining_reservation_usd": sum(i["reservation_nanodollars"] * max(0, 2 - journal.attempts.get(i["cache_key"], 0)) for i in pending) / 1e9,
                         **journal.summary()}), flush=True)
        if args.dry_run:
            return 0
        api_key = None
        if config["provider"] == "gemini":
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Missing provider key")
        counts = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(infer_one, item, config, journal, api_key) for item in pending]
            for number, future in enumerate(as_completed(futures), 1):
                status = future.result(); counts[status] = counts.get(status, 0) + 1
                if number % 20 == 0:
                    print(canonical({"completed_this_run": number, "statuses": counts, **journal.summary()}), flush=True)
        terminal = sum(i["cache_key"] in journal.completed for i in items)
        print(canonical({"status": "complete" if terminal == len(items) else "partial",
                         "requested_rows": len(items), "terminal_rows": terminal, "statuses": counts,
                         **journal.summary()}), flush=True)
        return 0 if terminal == len(items) and not journal.bound_exceeded else 3
    except Exception as error:
        print(canonical({"status": "stopped", "error_class": type(error).__name__, "details_suppressed": True}), flush=True)
        return 2
    finally:
        if journal:
            journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
