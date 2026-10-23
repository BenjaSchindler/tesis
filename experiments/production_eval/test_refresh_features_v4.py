import copy
import json
import unittest

import numpy as np

from experiments.production_eval import refresh_features_v4 as features
from experiments.production_eval.synthetic_refresh_v4 import BASE, load_jsonl


class RefreshFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(BASE / "draft_dataset/inputs.jsonl")
        cls.construction = features.sorted_scope(cls.rows, "construction", "A")
        rng = np.random.default_rng(73014)
        cls.input_matrix = rng.normal(size=(200, 12))
        cls.gold_matrix = rng.normal(size=(200, 12))

    def test_23_sets_exact_budgets_and_main_uniform_weights(self):
        runs = features.freeze_initial_sets(self.construction, self.input_matrix, self.gold_matrix)
        self.assertEqual(len(runs), 23)
        self.assertEqual([r["seed"] for r in runs[:20]], list(features.ACQUISITION_SEEDS))
        self.assertEqual(sum(r["uses_gold_in_selection"] for r in runs), 1)
        for run in runs:
            self.assertEqual(len(set(run["selected_family_ids"])), 80)
            self.assertEqual(run["sample_weights"], [1/80] * 80)
            self.assertAlmostEqual(sum(run["selector_native_alternative_weights"]), 1)
            self.assertFalse(run["alternative_weights_are_primary"])
            self.assertEqual(run["future_inputs_used_in_selection"], 0)

    def test_main_sets_do_not_depend_on_construction_gold_features(self):
        first = features.freeze_initial_sets(self.construction, self.input_matrix, self.gold_matrix)
        second = features.freeze_initial_sets(self.construction, self.input_matrix, -self.gold_matrix[:, ::-1])
        self.assertEqual(first[:22], second[:22])

    def test_selection_rejects_future_rows_and_unexpected_gold_fields(self):
        rows = copy.deepcopy(self.construction)
        rows[0]["partition"] = "future"
        with self.assertRaises(ValueError):
            features.freeze_initial_sets(rows, self.input_matrix, self.gold_matrix)
        rows = copy.deepcopy(self.construction)
        rows[0]["answer"] = "forbidden"
        with self.assertRaises(ValueError):
            features.freeze_initial_sets(rows, self.input_matrix, self.gold_matrix)

    def test_gold_scope_is_exactly_construction_and_hashes_match_encoder_text(self):
        ids = {r["example_id"] for r in self.construction}
        targets = features.scoped_gold(BASE / "draft_dataset/gold.jsonl", ids)
        self.assertEqual(set(targets), ids)
        rows = features.gold_feature_rows(self.construction, targets)
        for old, new in zip(self.construction, rows):
            self.assertEqual(new["input_text"], old["input_text"] + features.GOLD_SEPARATOR + targets[old["example_id"]])
            self.assertEqual(new["input_hash"], features.digest(new["input_text"]))
        targets["future"] = "not allowed"
        with self.assertRaises(ValueError):
            features.gold_feature_rows(self.construction, targets)

    def test_draft_cpu_seed_contract(self):
        protocol = json.loads((BASE / "draft_selection_protocol.json").read_text())
        features.validate_selection_protocol(protocol)
        protocol["device"] = "cuda:0"
        with self.assertRaises(ValueError):
            features.validate_selection_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
