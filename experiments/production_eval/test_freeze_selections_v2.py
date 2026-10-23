import unittest

import numpy as np

from experiments.production_eval.freeze_selections_v2 import input_features, make_selections


class FreezeTests(unittest.TestCase):
    def rows(self, n=25):
        return [{"family_id": f"family_{i:04d}", "input_text": "same length"} for i in range(n)]

    def test_tied_lengths_have_reproducible_rank_quartiles(self):
        rows = self.rows()
        strata, risk = input_features(rows)
        self.assertEqual([strata.count(f"length_quartile_{i}") for i in range(4)], [7, 6, 6, 6])
        self.assertEqual(int(risk.sum()), 5)
        reversed_strata, reversed_risk = input_features(rows[::-1])
        self.assertEqual(strata, reversed_strata[::-1])
        np.testing.assert_equal(risk, reversed_risk[::-1])
        self.assertEqual(len(set(zip(strata, risk))), 5)

    def test_exact_cartesian_run_count_and_input_only_variants(self):
        rows = self.rows()
        rng = np.random.default_rng(1)
        matrices = {name: rng.normal(size=(25, 8)) for name in
                    ["mpnet_head", "mpnet_head_tail", "mpnet_mean_chunks", "qwen3_document"]}
        runs = make_selections("structured_records", rows, matrices)
        self.assertEqual(len(runs), 117)
        self.assertEqual(len({run["run_id"] for run in runs}), 117)
        for run in runs:
            self.assertEqual(len(set(run["selected_family_ids"])), run["budget"])
            self.assertAlmostEqual(sum(run["sample_weights"]), 1)
            if run["policy"] in {"uniform_random", "stratified_random", "random_plus_targeted"}:
                self.assertIn(run["seed"], range(100, 110))
                self.assertIsNone(run["embedding_variant"])
                self.assertTrue(all(p > 0 for p in run["population_inclusion_probabilities"]))
            else:
                self.assertIsNone(run["seed"])
                self.assertIsNone(run["population_inclusion_probabilities"])


if __name__ == "__main__":
    unittest.main()
