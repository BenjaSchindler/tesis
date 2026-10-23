"""Stage and artifact boundaries; no models, credentials, gold or real outputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.production_eval import refresh_run_contract_v4 as contract


class RefreshContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.patcher = patch.object(contract, "BASE", self.base)
        self.patcher.start()
        dataset = self.base / "draft_dataset"; dataset.mkdir()
        files = {}
        self.inputs = {}
        for partition, count in contract.STAGE_COUNTS.items():
            rows = [{"example_id": f"{partition}_{i}", "family_id": f"family_{partition}_{i}",
                     "partition": partition, "domain": "structured_records", "group_id": "a", "complexity": "simple",
                     "presentation": "A", "input_text": "input", "input_hash": __import__('hashlib').sha256(b"input").hexdigest(),
                     "observable_counts": {"records": 8}} for i in range(count)]
            path = dataset / f"inputs_{partition}.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            files[path.name] = {"sha256": contract.sha(path)}
            self.inputs[partition] = rows
        schemas = dataset / "schemas.json"; schemas.write_text("{}")
        files[schemas.name] = {"sha256": contract.sha(schemas)}
        manifest = dataset / "manifest.json"; manifest.write_text(json.dumps({"files": files}))
        analysis = self.base / "analysis_plan.json"; analysis.write_text("{}")
        selection = self.base / "selection/selection_manifest.json"; selection.parent.mkdir(); selection.write_text("{}")
        self.run = {"status": "frozen", "inference_enabled": True,
                    "configs": [{"config_id": c} for c in sorted(contract.CONFIG_IDS)],
                    "code_files": [{"path": contract.__file__, "sha256": contract.sha(contract.__file__)}],
                    "dataset_manifest_sha256": contract.sha(manifest), "analysis_plan_sha256": contract.sha(analysis),
                    "selection_manifest_sha256": contract.sha(selection)}
        self.run_path = self.base / "run_manifest.json"
        self.run_path.write_text(json.dumps(self.run))

    def tearDown(self):
        self.patcher.stop(); self.tmp.cleanup()

    def gate(self):
        outputs = []
        for config in sorted(contract.CONFIG_IDS):
            path = self.base / (config + ".jsonl")
            rows = [{"example_id": r["example_id"], "input_hash": r["input_hash"], "status": "success",
                     "run_manifest_sha256": contract.sha(self.run_path),
                     "partition": "construction", "config_id": config} for r in self.inputs["construction"]]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            outputs.append({"config_id": config, "path": str(path), "sha256": contract.sha(path), "count": 200})
        predictions = self.base / "predictions.json"; predictions.write_text("{}")
        gate = {"status": "frozen", "run_manifest_sha256": contract.sha(self.run_path),
                "selection_manifest_sha256": self.run["selection_manifest_sha256"], "labels_scope": "construction_only",
                "construction_responses": outputs, "predictor_files": [{"path": str(predictions), "sha256": contract.sha(predictions)}]}
        path = self.base / "future_predictor_manifest.json"
        path.write_text(json.dumps(gate))
        return path, gate

    def test_construction_needs_no_gold_or_program_files(self):
        result = contract.validate_stage(self.run_path, "gemini_direct", "construction")
        self.assertEqual(len(result["rows"]), 200)
        self.assertIsNone(result["provenance"]["future_gate"])
        self.assertFalse((self.base / "draft_dataset/gold.jsonl").exists())

    def test_draft_cannot_run_and_future_cannot_run_before_predictions(self):
        with self.assertRaisesRegex(ValueError, "Future inference requires"):
            contract.validate_stage(self.run_path, "gemini_direct", "future")
        self.run["inference_enabled"] = False; self.run_path.write_text(json.dumps(self.run))
        with self.assertRaisesRegex(ValueError, "not frozen and enabled"):
            contract.validate_stage(self.run_path, "gemini_direct", "construction")

    def test_future_accepts_only_all_complete_matching_construction_sources(self):
        path, gate = self.gate()
        self.assertEqual(len(contract.validate_stage(self.run_path, "random_base_seed17", "future")["rows"]), 800)
        for mutation in ("model", "scope", "wrong_run", "partial", "failed", "input_hash"):
            changed = copy.deepcopy(gate)
            if mutation == "model": changed["construction_responses"].pop()
            if mutation == "scope": changed["labels_scope"] = "future"
            if mutation == "wrong_run": changed["run_manifest_sha256"] = "other"
            if mutation in {"partial", "failed", "input_hash"}:
                entry = changed["construction_responses"][0]
                rows = [json.loads(l) for l in Path(entry["path"]).read_text().splitlines()]
                if mutation == "partial": rows.pop()
                if mutation == "failed": rows[0]["status"] = "api_error"
                if mutation == "input_hash": rows[0]["input_hash"] = "changed"
                temporary = self.base / (mutation + ".jsonl")
                temporary.write_text("".join(json.dumps(r)+"\n" for r in rows))
                entry.update(path=str(temporary), sha256=contract.sha(temporary))
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): contract.validate_stage(self.run_path, "gemini_direct", "future")

    def test_input_hash_schema_code_and_prediction_artifact_changes_fail(self):
        path, gate = self.gate()
        for target in (self.base / "draft_dataset/inputs_future.jsonl", self.base / "draft_dataset/schemas.json", self.base / "predictions.json"):
            prior = target.read_bytes(); target.write_bytes(prior+b" ")
            with self.assertRaises(ValueError): contract.validate_stage(self.run_path, "gemini_direct", "future")
            target.write_bytes(prior)
        self.run["code_files"][0]["sha256"] = "changed"
        self.run_path.write_text(json.dumps(self.run))
        with self.assertRaises(ValueError): contract.validate_stage(self.run_path, "gemini_direct", "construction")

    def test_even_rehashed_input_gold_field_is_rejected(self):
        path = self.base / "draft_dataset/inputs_fixture.jsonl"
        rows = copy.deepcopy(self.inputs["fixture"]); rows[0]["answer"] = "leaked"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "gold fields forbidden"):
            contract.read_inputs(path, "fixture")


if __name__ == "__main__":
    unittest.main()
