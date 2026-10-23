import unittest
from itertools import combinations
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from experiments.production_eval.geometry_outcome_diagnostic import smoothness, uniform_mean_variance, load_frozen_feature
from experiments.production_eval.analyze_outcomes_v2 import family_scores


class GeometryDiagnosticTest(unittest.TestCase):
    def test_domain_composition_alone_explains_apparent_global_geometry_gain(self):
        matrix = [[-1, 0]] * 3 + [[1, 0]] * 3
        rows = smoothness(matrix, [0, 0, 0, 1, 1, 1], ks=(1, 2), domains=["a"] * 3 + ["b"] * 3)
        for row in rows:
            self.assertEqual(row["relative_mse_improvement"], 1)
            self.assertEqual(row["loo_domain_mean_mse"], 0)
            self.assertTrue(row["domain_baseline_perfect"])
            self.assertEqual(row["neighbor_minus_domain_mean_mse"], 0)
            self.assertIsNone(row["relative_mse_improvement_vs_domain_mean"])

    def test_single_domain_control_equals_global_leave_one_out_mean(self):
        row = smoothness(np.eye(6), [0, 0, 0, 1, 1, 1], domains=["same"] * 6)[0]
        self.assertEqual(row["loo_domain_mean_mse"], row["loo_global_mean_mse"])
        self.assertEqual(row["loo_domain_mean_mae"], row["loo_global_mean_mae"])
        self.assertEqual(row["relative_mse_improvement_vs_domain_mean"], row["relative_mse_improvement"])

    def test_domain_mean_excludes_own_label(self):
        # Within each domain y=[0,1,1]; leave-one-out predictions [1,.5,.5]
        # have MSE=.5. Including own labels would incorrectly give MSE=2/9.
        row = smoothness(np.eye(6), [0, 1, 1, 0, 1, 1], domains=["a"] * 3 + ["b"] * 3)[0]
        self.assertEqual(row["loo_domain_mean_mse"], .5)
        self.assertEqual(row["domain_family_counts"], {"a": 3, "b": 3})
        self.assertFalse(row["domain_baseline_perfect"])
        with self.assertRaises(ValueError):
            smoothness(np.eye(6), range(6), domains=["singleton"] + ["other"] * 5)

    def test_invalid_target_dimensions_and_neighbor_sizes_are_rejected(self):
        for ks in [(), (0,), (-1,), (True,), (1.5,), (1, 1), (6,)]:
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                smoothness(np.eye(6), range(6), ks=ks)
        with self.assertRaises(ValueError):
            smoothness(np.eye(6), np.zeros((6, 1)))
        for budget in (True, 1.5):
            with self.assertRaises(ValueError):
                uniform_mean_variance([0, 1], budget)

    def test_family_target_uses_all_four_descendants_without_partial_family(self):
        rows = [{"example_id": f"{family}-{i}", "family_id": family} for family in ("a", "b") for i in range(4)]
        grades = {("c", row["example_id"]): {"observed": True, "correct": i % 4 < 2} for i, row in enumerate(rows)}
        self.assertEqual(family_scores(rows, grades, "c"), {"a": .5, "b": .5})
        grades[("c", "b-3")]["observed"] = False
        self.assertEqual(family_scores(rows, grades, "c"), {"a": .5})

    def feature_fixture(self, folder, *, id_key="example_ids", partial=False, wrong_input=False):
        rows = [{"example_id": f"e{i}", "family_id": f"f{i}", "input_hash": f"h{i}",
                 "answer_type": "decimal", "dataset": "gsm8k"} for i in range(6)]
        declared = [{"family_id": row["family_id"], "representative_example_id": row["example_id"], "input_hash": row["input_hash"]} for row in rows]
        selected = rows[:-1] if partial else rows
        hashes = [row["input_hash"] for row in selected]
        if wrong_input: hashes[0] = "other_input"
        path = Path(folder) / "head.npz"
        np.savez(path, **{id_key: np.array([row["family_id"] for row in selected]), "input_hashes": np.array(hashes), "embeddings": np.eye(len(selected))})
        manifest = {"feature_evidence": [{"embedding_file": str(path), "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dataset": "gsm8k", "embedding_variant": "mpnet_head"}],
                    "input_manifest": {"gsm8k": declared}}
        return path, rows, manifest

    def test_mpnet_example_id_field_is_resolved_as_frozen_family_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path, rows, manifest = self.feature_fixture(folder)
            ids, matrix, recovered_rows, evidence = load_frozen_feature(path, rows, manifest)
            self.assertEqual(ids, [f"f{i}" for i in range(6)])
            self.assertEqual(len(recovered_rows), 6)
            self.assertEqual(evidence["embedding_variant"], "mpnet_head")
            np.testing.assert_array_equal(matrix, np.eye(6))

    def test_partial_feature_pool_and_wrong_base_input_hash_are_rejected(self):
        for settings in ({"partial": True}, {"wrong_input": True}):
            with tempfile.TemporaryDirectory() as folder:
                path, rows, manifest = self.feature_fixture(folder, **settings)
                with self.assertRaises(ValueError):
                    load_frozen_feature(path, rows, manifest)

    def test_feature_content_change_after_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, rows, manifest = self.feature_fixture(folder, id_key="family_ids")
            with path.open("ab") as stream: stream.write(b"altered")
            with self.assertRaises(ValueError):
                load_frozen_feature(path, rows, manifest)

    def test_two_clusters_predict_held_out_member_without_self(self):
        matrix = [[-1, 0]] * 3 + [[1, 0]] * 3
        results = smoothness(matrix, [0, 0, 0, 1, 1, 1], ks=(1, 2, 3))
        self.assertEqual(results[0]["loo_neighbor_mean_mse"], 0)
        self.assertEqual(results[1]["relative_mse_improvement"], 1)
        self.assertGreater(results[2]["loo_neighbor_mean_mse"], 0)
        self.assertAlmostEqual(results[0]["loo_global_mean_mse"], .36)

    def test_constant_outcomes_do_not_claim_geometry_gain(self):
        for value in (.25, .1):
            result = smoothness(np.eye(6), [value] * 6)[0]
            self.assertTrue(result["constant_target"])
            self.assertIsNone(result["relative_mse_improvement"])

    def test_orthogonal_vectors_cannot_copy_own_label(self):
        result = smoothness(np.eye(6), [0, 1, 2, 3, 4, 5])[0]
        self.assertGreater(result["loo_neighbor_mean_mse"], 0)

    def test_invalid_features_fail_closed(self):
        with self.assertRaises(ValueError):
            smoothness(np.zeros((6, 2)), range(6))
        with self.assertRaises(ValueError):
            smoothness(np.eye(6), [0, 1, 2, 3, 4, float("nan")])

    def test_finite_pool_variance_matches_every_possible_sample(self):
        values = np.array([-1., -.25, 0., .25, 1., 1.])
        population_mean = values.mean()
        for budget in range(1, 7):
            sample_errors = [(values[list(ids)].mean() - population_mean) ** 2
                             for ids in combinations(range(6), budget)]
            self.assertAlmostEqual(uniform_mean_variance(values, budget), np.mean(sample_errors))
        self.assertEqual(uniform_mean_variance(values, 6), 0.)


if __name__ == "__main__":
    unittest.main()
