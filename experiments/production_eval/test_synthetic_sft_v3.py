import unittest
from collections import Counter

import numpy as np

from experiments.production_eval.synthetic_sft_v3 import (
    DOMAINS, fixture_rows, freeze_arms, new_tasks,
)


class SyntheticSFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = new_tasks()

    def test_roles_and_new_family_counts(self):
        self.assertEqual(Counter(row["role"] for row in self.rows),
                         {"train_pool": 512, "validation": 40, "reference": 120})
        roles = {}
        for row in self.rows:
            previous = roles.setdefault(row["family_id"], row["role"])
            self.assertEqual(previous, row["role"])
        self.assertEqual(len(roles), 336)
        self.assertEqual(len({row["input_hash"] for row in self.rows}), len(self.rows))
        self.assertFalse({row["input_hash"] for row in self.rows} &
                         {row["input_hash"] for row in fixture_rows()})

    def test_arms_match_example_counts_and_control_wording(self):
        matrices = {domain: np.random.default_rng(42).normal(size=(128, 8)) for domain in DOMAINS}
        runs, _ = freeze_arms(self.rows, matrices)
        self.assertEqual(len(runs), 12)
        for run in runs:
            self.assertEqual(run["training_examples"], 64)
            self.assertEqual(run["domain_counts"], {domain: 32 for domain in DOMAINS})
            expected_families = 32 if run["arm"] == "coverage_paired_wording" else 64
            self.assertEqual(run["unique_training_families"], expected_families)
        by_arm = {run["arm"]: run for run in runs if run["training_seed"] == 17}
        by_id = {row["example_id"]: row for row in self.rows}
        base = {by_id[i]["family_id"] for i in by_arm["coverage_base"]["training_example_ids"]}
        mixed = {by_id[i]["family_id"] for i in by_arm["coverage_mixed_wording"]["training_example_ids"]}
        self.assertEqual(base, mixed)
        self.assertEqual(by_arm["coverage_mixed_wording"]["variant_counts"], {"base": 32, "paraphrase": 32})
        for arm in by_arm:
            first, second = [run for run in runs if run["arm"] == arm]
            self.assertEqual(first["training_example_ids"], second["training_example_ids"])


if __name__ == "__main__":
    unittest.main()
