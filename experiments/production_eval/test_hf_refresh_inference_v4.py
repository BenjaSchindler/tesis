import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.production_eval import hf_refresh_inference_v4 as hf


def fixture():
    text = "Input-only fixture"
    return {"example_id": "1" * 64, "family_id": "2" * 64, "domain": "structured_records", "complexity": "simple",
            "group_id": "structured_records_simple", "partition": "fixture", "presentation": "A",
            "input_text": text, "input_hash": hf.digest(text), "observable_counts": {"records": 8}}


def record(row=None, response="not JSON"):
    row = fixture() if row is None else row
    return hf.observed_record(row, "random_base_seed17", "frozen", response, [10, 11], [1, 2], 2, 1., 11)


class HFRefreshInferenceTests(unittest.TestCase):
    def test_gold_fields_hash_mismatch_and_wrong_partition_rejected(self):
        row = fixture()
        for modified in ({**row, "answer": "gold"}, {**row, "input_text": "changed"}, {**row, "partition": "future"}, {**row, "group_id": "wrong"}):
            with self.subTest(modified=modified):
                with self.assertRaises(ValueError):
                    hf.validate_input(modified, "fixture")

    def test_format_invalid_generation_is_observed_and_not_graded(self):
        observed = record()
        self.assertEqual(observed["status"], "success")
        self.assertEqual(observed["format_status"], "invalid")
        self.assertNotIn("exact_match", observed)
        self.assertNotIn("answer", observed)
        self.assertNotIn("input_text", observed)
        self.assertEqual(observed["generated_tokens"], 2)

    def test_schema_filter_must_cover_every_token(self):
        row = fixture()
        with self.assertRaises(ValueError):
            hf.observed_record(row, "random_base_seed17", "frozen", "output", [1, 2], [3], 1, 1., 2)

    def test_existing_invalid_output_preserved_and_corruption_rejected(self):
        row = fixture()
        original = record(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            hf.append_json(path, original)
            existing = hf.read_observed(path, {row["example_id"]: row}, "random_base_seed17", "frozen")
            self.assertEqual(existing[row["example_id"]]["response"], "not JSON")
            for key, value in (("status", "unknown"), ("run_manifest_sha256", "different"), ("response", "changed"), ("format_status", "valid"), ("prefix_filter_calls", 1)):
                changed = {**original, key: value}
                path.write_text(json.dumps(changed) + "\n")
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):
                        hf.read_observed(path, {row["example_id"]: row}, "random_base_seed17", "frozen")

    def test_duplicate_or_partial_journal_does_not_get_silently_repaired(self):
        row = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            hf.append_json(path, record(row))
            hf.append_json(path, record(row))
            with self.assertRaises(ValueError):
                hf.read_observed(path, {row["example_id"]: row}, "random_base_seed17", "frozen")
            before = (json.dumps(record(row)) + "\n").encode() + b'{"partial":'
            path.write_bytes(before)
            with self.assertRaises(ValueError):
                hf.read_observed(path, {row["example_id"]: row}, "random_base_seed17", "frozen")
            self.assertEqual(path.read_bytes(), before)

    def test_completed_resume_does_not_load_model_or_regenerate(self):
        row = fixture()
        config_id = "random_base_seed17"
        contract = {"rows": [row], "config": {"config_id": config_id, "trained_followup_manifest_sha256": hf.sha(hf.FOLLOWUP / "followup_manifest.json")},
                    "schemas": {domain: hf.schema_for_domain(domain) for domain in ("structured_records", "inventory_tracking")},
                    "provenance": {"run_manifest_sha256": "frozen", "code_files": [{"path": hf.__file__}], "future_gate": None, "input_sha256": "input"}}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "inference" / config_id / "fixture"
            destination.mkdir(parents=True)
            hf.append_json(destination / "responses.jsonl", record(row))
            completion = {"status": "completed", "config_id": config_id, "partition": "fixture", "evaluated": 1, "success_count": 1,
                          "run_manifest_sha256": "frozen", "responses_sha256": hf.sha(destination / "responses.jsonl")}
            (destination / "completed.json").write_text(json.dumps(completion))
            with patch.object(hf, "BASE", base), patch.object(hf, "verify_followup"), \
                 patch("experiments.production_eval.refresh_run_contract_v4.validate_stage", return_value=contract), \
                 patch.object(hf, "tokenizer_cpu", side_effect=AssertionError("Must not reload")), \
                 patch.object(hf, "load_trained", side_effect=AssertionError("Must not infer")):
                hf.run(Path("unused"), config_id, "fixture", None, "cuda:0")


if __name__ == "__main__":
    unittest.main()
