"""Offline transport, cost and irreversible-first-output journal checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.production_eval.api_audit import digest
from experiments.production_eval.experiment_inference import ContextLimitError, RunJournal
from experiments.production_eval.refresh_gemini_v4 import format_result, infer_one, prepare
from experiments.production_eval import refresh_gemini_v4 as runner


class RefreshGeminiTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(config_id="gemini_direct", provider="gemini", model="gemini-3.1-flash-lite",
                           temperature=0., seed=2026091504, max_output_tokens=512, max_context_tokens=8192,
                           thinking_level="minimal", api_cap_usd=1., timeout_seconds=180)
        self.row = dict(example_id="a", family_id="f", partition="construction", domain="structured_records", group_id="g",
                        complexity="simple", presentation="A", input_text="arbitrary input", input_hash=digest("arbitrary input"),
                        observable_counts={"records": 8})
        self.schema = {"type": "object", "additionalProperties": False, "required": ["n"],
                       "properties": {"n": {"type": "integer"}}}
        self.item = prepare(self.row, self.config, {"structured_records": self.schema}, "frozen")

    def response(self, text='{"n":1}'):
        return {"response": text, "raw_response": {"text": text},
                "usage": {"prompt_token_count": 10, "candidates_token_count": 3, "thoughts_token_count": 2, "total_token_count": 15}}

    def test_request_is_gold_free_and_bound_accounts_for_thinking(self):
        self.assertEqual(set(self.item["request"]), {"input_text", "schema"})
        with self.assertRaises(ValueError): prepare({**self.row, "answer": "secret"}, self.config, {"structured_records": self.schema}, "frozen")
        with self.assertRaises(ValueError): prepare({**self.row, "input_text": "changed"}, self.config, {"structured_records": self.schema}, "frozen")
        with tempfile.TemporaryDirectory() as directory:
            journal = RunJournal(Path(directory)/"out.jsonl", self.config)
            try:
                infer_one(self.item, self.config, journal, call=lambda *args: self.response())
                saved = next(iter(journal.completed.values()))
                self.assertEqual(saved["charged_nanodollars"], 10*250+5*1500)
                self.assertEqual(saved["status"], "success")
                self.assertNotIn("exact_match", saved)
            finally: journal.close()

    def test_invalid_model_output_is_terminal_and_replay_does_not_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"out.jsonl"; calls = []
            def broken(*args): calls.append(1); return self.response("not json")
            journal = RunJournal(path, self.config)
            infer_one(self.item, self.config, journal, call=broken); journal.close()
            original = path.read_bytes()
            journal = RunJournal(path, self.config)
            try:
                self.assertEqual(infer_one(self.item, self.config, journal, call=broken), "cached")
                saved = next(iter(journal.completed.values()))
                self.assertEqual(saved["status"], "success")
                self.assertFalse(saved["required_schema_valid"])
            finally: journal.close()
            self.assertEqual(calls, [1]); self.assertEqual(path.read_bytes(), original)

    def test_retry_preserves_uncertain_attempt_charge_and_first_completed_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"out.jsonl"; calls=[]
            def once_fails(*args):
                calls.append(1)
                if len(calls)==1: raise TimeoutError("Do not print this exception or credentials")
                return self.response()
            journal = RunJournal(path, self.config)
            try:
                infer_one(self.item,self.config,journal,call=once_fails)
                self.assertEqual(len(calls),2)
                self.assertEqual(journal.summary()["reserved_compute_attempts"],2)
                self.assertAlmostEqual(journal.summary()["ledger_accounted_usd"],(self.item["reservation_nanodollars"]+10000)/1e9)
                attempts=[json.loads(l) for l in journal.attempt_file.read_text().splitlines()]
                self.assertEqual(attempts[0]["error_class"],"TimeoutError")
                self.assertNotIn("Do not print",journal.attempt_file.read_text())
            finally: journal.close()

    def test_missing_usage_is_reserved_and_budget_prevents_call(self):
        with tempfile.TemporaryDirectory() as directory:
            journal=RunJournal(Path(directory)/"small.jsonl",self.config,budget_nanodollars=self.item["reservation_nanodollars"]-1)
            try:
                self.assertEqual(infer_one(self.item,self.config,journal,call=lambda *args: self.fail("overbudget transport")),"budget_or_attempt_limit")
            finally: journal.close()
            journal=RunJournal(Path(directory)/"normal.jsonl",self.config)
            try:
                infer_one(self.item,self.config,journal,call=lambda *args: {**self.response(),"usage":None})
                self.assertEqual(next(iter(journal.completed.values()))["charged_nanodollars"],self.item["reservation_nanodollars"])
            finally: journal.close()

    def test_context_preflight_rejection_is_no_generation_and_zero_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            journal=RunJournal(Path(directory)/"out.jsonl",self.config)
            def rejected(*args): raise ContextLimitError()
            try:
                infer_one(self.item,self.config,journal,call=rejected)
                self.assertEqual(journal.summary()["ledger_accounted_usd"],0.)
                self.assertEqual(next(iter(journal.completed.values()))["status"],"context_limit")
            finally: journal.close()

    def test_schema_checks_duplicate_keys_booleans_nonfinite_and_negative_integers(self):
        for text in ('{"n":1,"n":2}','{"n":true}','{"n":NaN}','{"n":1.0}','{"n":1,"extra":2}'):
            self.assertFalse(format_result(text,self.schema)["required_schema_valid"])
        self.assertTrue(format_result('{"n":-3}',self.schema)["required_schema_valid"])

    def test_terminal_infrastructure_outcome_does_not_create_stage_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            contract = {"rows": [self.row], "config": self.config, "schemas": {"structured_records": self.schema},
                        "provenance": {"run_manifest_sha256": "frozen", "code_files": [{"path": runner.__file__}]}}
            def failed_transport(*args): raise ContextLimitError()
            def failed_infer(item, config, journal, key):
                return infer_one(item, config, journal, key, call=failed_transport)
            with patch.object(runner, "BASE", base), patch.object(runner, "validate_stage", return_value=contract), \
                 patch.object(runner, "infer_one", side_effect=failed_infer), patch.object(runner.os, "getenv", return_value="test-key"), \
                 patch("dotenv.load_dotenv"), patch("sys.argv", ["runner", "construction", "--workers", "1"]):
                with self.assertRaisesRegex(RuntimeError, "unknown outcomes"):
                    runner.main()
            stage = base / "inference/gemini_direct/construction"
            self.assertTrue((stage / "responses.jsonl").exists())
            self.assertEqual(json.loads((stage / "incomplete_outcomes.json").read_text())["observed_generations"], 0)
            self.assertFalse((stage / "completed.json").exists())


if __name__ == "__main__":
    unittest.main()
