"""No-network tests for cost containment and provenance of the feasibility audit.

All fixtures and journal files live in temporary directories under /tmp. The
real Gemini response/config models are used, but its Client is always mocked.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import io
import json
import logging
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from google.genai import types

SPEC = importlib.util.spec_from_file_location("api_audit", Path(__file__).with_name("api_audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

SETTINGS = {
    "temperature": 0.0, "candidate_count": 1, "max_output_tokens": 1024,
    "thinking_level": "minimal", "include_thoughts": False,
    "response_mime_type": "application/json", "timeout_ms": 60000,
    "sdk_attempts": 1, "harness_max_attempts": 2,
    "rubric_version": "feasibility_input_only_v1",
}
ANNOTATION = {
    "task_family": "extraction", "language": "en", "source_present": True,
    "self_contained": True, "outcome_judgeable": True,
    "objective_checker_possible": True, "missing_dependency": "none",
    "reason": "The supplied source supports an objectively checkable extraction.",
}


def request(identifier="a", value="Extract the number from this source: count=7"):
    return {"example_id": identifier, "input_text": value, "input_hash": audit.digest(value)}


def response(annotation=ANNOTATION, usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(text=json.dumps(annotation))
        ]))],
        model_version=audit.MODEL,
        usage_metadata=(types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=20,
            thoughts_token_count=5, total_token_count=35,
        ) if usage else None),
    )


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="production_eval_audit_test_", dir="/tmp")
        self.directory = Path(self.temp.name)
        self.output = self.directory / "audit.jsonl"
        self.item = audit.prepare(request(), audit.MODEL, SETTINGS)
        self.journals = []
        # Any accidental real network attempt fails the test immediately.
        self.socket_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.socket_guard.start()

    def tearDown(self):
        for journal in self.journals:
            journal.close()
        self.socket_guard.stop()
        logging.disable(logging.NOTSET)
        self.temp.cleanup()

    def journal(self, budget=2_000_000_000):
        journal = audit.Journal(self.output, budget)
        self.journals.append(journal)
        return journal

    def run_mocked(self, outcomes, journal=None, item=None):
        journal = journal or self.journal()
        client = MagicMock()
        client.models.generate_content.side_effect = outcomes
        stop = threading.Event()
        with patch("google.genai.Client") as constructor, patch.object(audit.time, "sleep"):
            constructor.return_value.__enter__.return_value = client
            result = audit.audit_one(item or self.item, journal, "FAKE_TEST_KEY_ONLY", audit.MODEL, SETTINGS, stop)
        return result, client, constructor, stop

    def test_rubric_schema_and_strict_boolean_validation(self):
        self.assertTrue(audit.eligible(audit.validate_annotation(json.dumps(ANNOTATION))))
        invalid = [
            {**ANNOTATION, "source_present": "true"},
            {**ANNOTATION, "self_contained": 1},
            {**ANNOTATION, "task_family": "unknown"},
            {**ANNOTATION, "language": "zz"},
            {**ANNOTATION, "reason": "word " * 41},
            {**ANNOTATION, "extra": "field"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.validate_annotation(json.dumps(value))
        for difference in ({"language": "es"}, {"task_family": "coding"}, {"source_present": False}):
            self.assertFalse(audit.eligible({**ANNOTATION, **difference}))

    def test_request_hash_changes_with_text_rubric_model_and_settings(self):
        key = self.item["cache_key"]
        self.assertEqual(key, audit.prepare(request("different_id"), audit.MODEL, SETTINGS)["cache_key"])
        self.assertNotEqual(key, audit.prepare(request(value="Count=8"), audit.MODEL, SETTINGS)["cache_key"])
        self.assertNotEqual(key, audit.prepare(request(), "different-model", SETTINGS)["cache_key"])
        self.assertNotEqual(key, audit.prepare(request(), audit.MODEL, {**SETTINGS, "temperature": 1})["cache_key"])
        with patch.object(audit, "RUBRIC", audit.RUBRIC + "changed"):
            self.assertNotEqual(key, audit.prepare(request(), audit.MODEL, SETTINGS)["cache_key"])

    def test_input_validation_and_assistant_exclusion(self):
        path = self.directory / "input.jsonl"
        row = {**request(), "assistant_response": "SECRET_ASSISTANT_OUTPUT"}
        path.write_text(json.dumps(row) + "\n")
        normalized = audit.read_requests(path, 300)
        self.assertEqual(set(normalized[0]), {"example_id", "input_text", "input_hash"})
        self.assertNotIn("SECRET_ASSISTANT_OUTPUT", audit.prepare(normalized[0], audit.MODEL, SETTINGS)["contents"])
        path.write_text(json.dumps({**row, "input_hash": "wrong"}) + "\n")
        with self.assertRaises(ValueError):
            audit.read_requests(path, 300)

    def test_dry_run_deduplicates_without_credentials_or_writes(self):
        path = self.directory / "input.jsonl"
        path.write_text("\n".join(json.dumps(request(identifier)) for identifier in ("a", "b")) + "\n")
        capture = io.StringIO()
        with patch("sys.argv", ["api_audit", "--input", str(path), "--output", str(self.output), "--dry-run"]), \
                patch("dotenv.load_dotenv") as dotenv, patch("google.genai.Client") as constructor, redirect_stdout(capture):
            self.assertEqual(audit.main(), 0)
        report = json.loads(capture.getvalue())
        self.assertEqual(report["pending_unique_requests"], 1)
        self.assertEqual(report["input_rows"], 2)
        self.assertAlmostEqual(report["all_remaining_attempts_reservation_usd"], self.item["reservation_nanodollars"] * 2 / 1e9)
        self.assertEqual(list(self.directory.iterdir()), [path])
        dotenv.assert_not_called()
        constructor.assert_not_called()
        self.assertNotIn(request()["input_text"], capture.getvalue())

    def test_concurrent_reservations_cannot_exceed_cap(self):
        journal = self.journal(95)
        items = [{"cache_key": str(i), "reservation_nanodollars": 10} for i in range(20)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(journal.reserve, items))
        self.assertEqual(sum(result is not None for result in reservations), 9)
        self.assertEqual(sum(journal.charges.values()), 90)
        self.assertEqual(len(audit.read_jsonl(journal.ledger_path)), 9)

    def test_crashed_call_keeps_charge_and_attempt_count_after_restart(self):
        amount = self.item["reservation_nanodollars"]
        journal = self.journal(amount * 2)
        first = journal.reserve(self.item)
        self.assertEqual(first[1], 1)
        journal.close()
        journal = self.journal(amount * 2)
        self.assertEqual(sum(journal.charges.values()), amount)
        self.assertEqual(journal.reserve(self.item)[1], 2)
        self.assertIsNone(journal.reserve(self.item))
        self.assertEqual(sum(journal.charges.values()), amount * 2)

    def test_reasoning_and_unexplained_tokens_are_billed(self):
        usage = {"prompt_token_count": 10, "candidates_token_count": 20, "thoughts_token_count": 5, "total_token_count": 35}
        self.assertEqual(audit.actual_cost(usage, 999999)[0], 10 * 250 + 25 * 1500)
        self.assertEqual(audit.actual_cost({**usage, "total_token_count": 40}, 999999)[0], 10 * 250 + 30 * 1500)
        for incomplete in (None, {}, {**usage, "prompt_token_count": None}, {**usage, "candidates_token_count": -1}):
            self.assertEqual(audit.actual_cost(incomplete, 999999)[0], 999999)

    def test_success_persists_raw_response_usage_and_resumes_without_call(self):
        result, client, constructor, _ = self.run_mocked([response()])
        self.assertEqual(result, "success")
        client.models.generate_content.assert_called_once()
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.thinking_config.thinking_level.value, "MINIMAL")
        self.assertIsNone(config.tools)
        self.assertEqual(constructor.call_args.kwargs["http_options"].retry_options.attempts, 1)
        row = audit.read_jsonl(self.output)[0]
        self.assertEqual(row["actual_model_version"], audit.MODEL)
        self.assertEqual(row["charged_nanodollars"], 40000)
        self.assertEqual(row["response_sha256"], audit.digest(audit.canonical(row["raw_response"])))
        self.assertEqual(row["annotation_provenance"], "machine_generated_unvalidated")
        self.journals[-1].close()
        reopened = self.journal()
        status, client, _, _ = self.run_mocked([], journal=reopened)
        self.assertEqual(status, "cached_or_stopped")
        client.models.generate_content.assert_not_called()
        self.assertEqual(sum(reopened.charges.values()), 40000)

    def test_invalid_schema_retries_and_bills_both_attempts(self):
        result, client, _, _ = self.run_mocked([response({**ANNOTATION, "source_present": "true"}), response()])
        self.assertEqual(result, "success")
        self.assertEqual(client.models.generate_content.call_count, 2)
        rows = audit.read_jsonl(self.output)
        self.assertEqual([row["status"] for row in rows], ["schema_invalid", "success"])
        self.assertIsNone(rows[0]["annotation"])
        self.assertIsNone(rows[0]["eligible_machine_prediction"])
        self.assertEqual(sum(self.journals[-1].charges.values()), 80000)

    def test_failed_calls_keep_reservations_redact_details_and_stop_at_two(self):
        secret = "NeverExposeThisAuthOrRequestContent"
        result, client, _, _ = self.run_mocked([RuntimeError(secret), TimeoutError(secret)])
        self.assertEqual(result, "api_error")
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertNotIn(secret, self.output.read_text())
        self.assertEqual(sum(self.journals[-1].charges.values()), self.item["reservation_nanodollars"] * 2)
        self.journals[-1].close()
        status, client, _, _ = self.run_mocked([], journal=self.journal())
        self.assertEqual(status, "budget_or_attempt_limit")
        client.models.generate_content.assert_not_called()

    def test_permanent_http_error_is_not_retried_in_run(self):
        class PermanentError(Exception):
            code = 400
        status, client, _, _ = self.run_mocked([PermanentError("Sensitive details")])
        self.assertEqual(status, "api_error")
        client.models.generate_content.assert_called_once()
        self.assertEqual(audit.read_jsonl(self.output)[0]["http_status_code"], 400)

    def test_missing_usage_retains_reservation(self):
        result, _, _, _ = self.run_mocked([response(usage=False)])
        self.assertEqual(result, "success")
        row = audit.read_jsonl(self.output)[0]
        self.assertEqual(row["charged_nanodollars"], self.item["reservation_nanodollars"])
        self.assertEqual(row["accounting_basis"], "conservative_reservation_missing_usage")

    def test_actual_usage_exceeding_reservation_stops_future_calls(self):
        broken_bound = {**self.item, "reservation_nanodollars": 1}
        result, client, _, stop = self.run_mocked([response()], item=broken_bound)
        self.assertEqual(result, "reservation_exceeded_stopped")
        self.assertTrue(stop.is_set())
        client.models.generate_content.assert_called_once()
        row = audit.read_jsonl(self.output)[0]
        self.assertTrue(row["reservation_exceeded"])
        self.assertEqual(row["charged_nanodollars"], 40000)
        self.assertIsNone(self.journals[-1].reserve({"cache_key": "different", "reservation_nanodollars": 1}))
        self.journals[-1].close()
        reopened = self.journal()
        self.assertTrue(reopened.billing_bound_exceeded)
        self.assertIsNone(reopened.reserve({"cache_key": "different", "reservation_nanodollars": 1}))

    def test_full_cli_duplicate_inputs_share_single_call_and_resume(self):
        path = self.directory / "input.jsonl"
        path.write_text("\n".join(json.dumps(request(identifier)) for identifier in ("a", "b")) + "\n")
        client = MagicMock()
        client.models.generate_content.return_value = response()
        argv = ["api_audit", "--input", str(path), "--output", str(self.output)]
        with patch("sys.argv", argv), patch("dotenv.load_dotenv"), \
                patch.dict("os.environ", {"GOOGLE_API_KEY": "FAKE_TEST_ONLY"}, clear=True), \
                patch("google.genai.Client") as constructor:
            constructor.return_value.__enter__.return_value = client
            capture = io.StringIO()
            with redirect_stdout(capture):
                self.assertEqual(audit.main(), 0)
            self.assertEqual(client.models.generate_content.call_count, 1)
            final = json.loads(capture.getvalue().splitlines()[-1])
            self.assertEqual(final["valid_machine_annotations_for_requested_rows"], 2)
            self.assertEqual(len(audit.read_jsonl(self.output)), 1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), 0)
            self.assertEqual(client.models.generate_content.call_count, 1)

    def test_full_cli_billing_breach_is_nonzero_even_with_valid_annotation(self):
        path = self.directory / "input.jsonl"
        path.write_text(json.dumps(request()) + "\n")
        client = MagicMock()
        client.models.generate_content.return_value = response()
        broken_bound = {**self.item, "reservation_nanodollars": 1}
        capture = io.StringIO()
        with patch("sys.argv", ["api_audit", "--input", str(path), "--output", str(self.output)]), \
                patch("dotenv.load_dotenv"), patch.dict("os.environ", {"GOOGLE_API_KEY": "FAKE_TEST_ONLY"}, clear=True), \
                patch.object(audit, "prepare", return_value=broken_bound), \
                patch("google.genai.Client") as constructor, redirect_stdout(capture):
            constructor.return_value.__enter__.return_value = client
            self.assertEqual(audit.main(), 2)
        final = json.loads(capture.getvalue().splitlines()[-1])
        self.assertEqual(final["status"], "stopped_billing_bound_exceeded")
        self.assertTrue(final["billing_bound_exceeded"])

    def test_negative_charge_and_duplicate_settlement_are_rejected(self):
        journal = self.journal()
        call_id, _ = journal.reserve(self.item)
        audit.Journal.append(journal.ledger_path, {
            "event": "settle", "call_id": call_id, "cache_key": self.item["cache_key"],
            "charged_nanodollars": -1000,
        })
        journal.close()
        with self.assertRaisesRegex(ValueError, "Invalid settlement"):
            audit.Journal(self.output, 2_000_000_000, readonly=True)
        journal.ledger_path.unlink()
        _, _, _, _ = self.run_mocked([response()], journal=self.journal())
        ledger = self.journals[-1].ledger_path
        settlement = audit.read_jsonl(ledger)[-1]
        audit.Journal.append(ledger, settlement)
        with self.assertRaisesRegex(ValueError, "repeated settlement"):
            audit.Journal(self.output, 2_000_000_000, readonly=True)

    def test_insufficient_budget_never_constructs_client(self):
        journal = self.journal(self.item["reservation_nanodollars"] - 1)
        status, _, constructor, _ = self.run_mocked([], journal=journal)
        self.assertEqual(status, "budget_or_attempt_limit")
        constructor.assert_not_called()
        self.assertEqual(sum(journal.charges.values()), 0)

    def test_corrupted_journal_does_not_silently_drop_charges(self):
        journal = self.journal()
        journal.reserve(self.item)
        path = journal.ledger_path
        journal.close()
        with path.open("a") as stream:
            stream.write('{"event":"reserve"')
        with self.assertRaises(ValueError):
            audit.Journal(self.output, 2_000_000_000, readonly=True)


if __name__ == "__main__":
    unittest.main()
