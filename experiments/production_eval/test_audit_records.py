"""Local provenance tests; synthetic fixtures are created only under /tmp."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("audit_records", Path(__file__).with_name("audit_records.py"))
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)
audit = loader.api_audit


class AuditRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit_records_test_", dir="/tmp")
        self.path = Path(self.temp.name) / "audit.jsonl"
        value = "Extract the count: count=7"
        self.request = {"example_id": "fixture", "input_text": value, "input_hash": audit.digest(value)}
        self.settings = {"max_output_tokens": 1024, "temperature": 0.0, "thinking_level": "minimal"}
        prepared = audit.prepare(self.request, audit.MODEL, self.settings)
        annotation = {
            "task_family": "extraction", "language": "en", "source_present": True,
            "self_contained": True, "outcome_judgeable": True,
            "objective_checker_possible": True, "missing_dependency": "none", "reason": "Source is present.",
        }
        raw = {"candidates": [{"text": json.dumps(annotation)}]}
        self.row = {
            "example_id": self.request["example_id"], "input_hash": self.request["input_hash"],
            "requested_model": audit.MODEL, "actual_model_version": audit.MODEL,
            "rubric_sha256": audit.digest(audit.RUBRIC), "settings": self.settings,
            "cache_key": prepared["cache_key"], "request_sha256": prepared["cache_key"],
            "annotation_provenance": "machine_generated_unvalidated", "status": "success",
            "annotation": annotation, "eligible_machine_prediction": True,
            "raw_response": raw, "response_sha256": audit.digest(audit.canonical(raw)),
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self, records):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in records))

    def load(self):
        return loader.load_audits(self.path, [self.request])

    def test_last_success_wins_and_later_failures_do_not_erase_it(self):
        later = {**self.row, "attempt": 2}
        failed = {**self.row, "status": "api_error", "annotation": None,
                  "eligible_machine_prediction": None, "actual_model_version": None}
        self.write([self.row, later, failed])
        before = self.path.read_bytes()
        records = self.load()
        self.assertEqual(records[self.request["input_hash"]]["attempt"], 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_all_unsuccessful_attempts_are_unknown_not_negative(self):
        self.write([{**self.row, "status": "schema_invalid", "annotation": {"broken": True}},
                    {**self.row, "status": "api_error", "actual_model_version": None}])
        self.assertEqual(self.load(), {})

    def test_failed_attempt_cannot_hide_configuration_change(self):
        changed = copy.deepcopy(self.row)
        changed.update(status="api_error", actual_model_version=None)
        changed["settings"]["max_output_tokens"] = 2048
        self.write([self.row, changed])
        with self.assertRaisesRegex(ValueError, "mixes frozen"):
            self.load()

    def test_model_and_rubric_changes_are_rejected(self):
        for difference in ({"requested_model": "another-model"}, {"rubric_sha256": "stale"}):
            with self.subTest(difference=difference):
                self.write([self.row, {**self.row, **difference}])
                with self.assertRaises(ValueError):
                    self.load()

    def test_same_alias_with_different_actual_versions_is_rejected(self):
        self.write([self.row, {**self.row, "actual_model_version": "changed-backend-version"}])
        with self.assertRaisesRegex(ValueError, "actual model versions"):
            self.load()

    def test_full_request_hash_input_hash_and_identifier_are_verified(self):
        for difference in ({"cache_key": "wrong"}, {"request_sha256": "wrong"},
                           {"input_hash": "outside"}, {"example_id": "outside"}):
            with self.subTest(difference=difference):
                self.write([{**self.row, **difference}])
                with self.assertRaises(ValueError):
                    self.load()
        self.write([self.row])
        changed = {**self.request, "input_text": "Changed text with stale hash"}
        with self.assertRaisesRegex(ValueError, "normalized request provenance"):
            loader.load_audits(self.path, [changed])

    def test_invalid_success_schema_eligibility_and_response_hash_are_rejected(self):
        variants = [
            {**self.row, "annotation": {**self.row["annotation"], "source_present": "true"}},
            {**self.row, "eligible_machine_prediction": False},
            {**self.row, "response_sha256": "wrong"},
            {**self.row, "actual_model_version": None},
        ]
        for row in variants:
            with self.subTest(keys=row.keys()):
                self.write([row])
                with self.assertRaises(ValueError):
                    self.load()

    def test_duplicate_text_request_aliases_share_success(self):
        alias = {**self.request, "example_id": "alias"}
        self.write([{**self.row, "example_id": "alias"}])
        records = loader.load_audits(self.path, [self.request, alias])
        self.assertEqual(len(records), 1)

    def test_missing_empty_and_malformed_files_are_distinct(self):
        with self.assertRaisesRegex(ValueError, "missing or unreadable"):
            self.load()
        self.write([])
        self.assertEqual(self.load(), {})
        sensitive = "PRIVATE_REQUEST_CONTENT"
        self.path.write_text(sensitive)
        with self.assertRaisesRegex(ValueError, "Malformed audit JSON") as context:
            self.load()
        self.assertNotIn(sensitive, str(context.exception))


if __name__ == "__main__":
    unittest.main()
