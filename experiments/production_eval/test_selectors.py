"""Meaningful small-pool invariants; no downloads, model calls or GPU use.

Run from the repository root:
    python -m unittest experiments.production_eval.test_selectors -v

"""

import unittest
import numpy as np

from experiments.production_eval.selection_methods import POLICIES, select


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["f", "b", "e", "a", "d", "c"]
        self.embeddings = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.], [-1., -1.]])
        self.texts = ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma",
                      "unrelated words here", "delta epsilon zeta", "final lexical group"]
        self.strata = ["large", "large", "large", "large", "small", "small"]
        self.risk = [0., 1., 0., 1., 0., 1.]

    def call(self, policy, budget=4, seed=3, perm=None):
        p = np.arange(6) if perm is None else np.array(perm)
        return select(policy, [self.ids[i] for i in p], budget, seed=seed,
                      embeddings=self.embeddings[p], texts=[self.texts[i] for i in p],
                      strata=[self.strata[i] for i in p], risk=np.array(self.risk)[p])

    def test_exact_unique_budget_and_weight_semantics(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                r = self.call(policy)
                self.assertEqual(len(set(r.indices)), 4)
                self.assertTrue(np.all(r.sample_weights > 0))
                self.assertAlmostEqual(r.sample_weights.sum(), 1.)
                if r.inclusion_probabilities is not None:
                    p = r.inclusion_probabilities
                    self.assertTrue(np.all((p > 0) & (p <= 1)))
                    self.assertAlmostEqual(p.sum(), 4.)
                    np.testing.assert_allclose(r.sample_weights, 1. / (6 * p[r.indices]))
                else:
                    self.assertFalse(r.metadata["design_based_uncertainty_eligible"])
                    self.assertIn("potentially biased", r.metadata["estimator"])

    def test_permutation_invariant_ids_and_weights(self):
        perm = [3, 5, 1, 4, 0, 2]
        for policy in POLICIES:
            with self.subTest(policy=policy):
                first, second = self.call(policy), self.call(policy, perm=perm)
                self.assertEqual(first.metadata["selected_ids"], second.metadata["selected_ids"])
                np.testing.assert_allclose(first.sample_weights, second.sample_weights)
                if first.inclusion_probabilities is not None:
                    np.testing.assert_allclose(first.inclusion_probabilities[perm], second.inclusion_probabilities)

    def test_uniform_known_probabilities(self):
        r = self.call("uniform_random", budget=2)
        np.testing.assert_allclose(r.inclusion_probabilities, np.full(6, 1 / 3))
        np.testing.assert_allclose(r.sample_weights, [.5, .5])

    def test_stratum_weights_recover_constant_stratum_population_mean(self):
        r = self.call("stratified_random", budget=2)
        outcomes = np.array([1., 1., 1., 1., 0., 0.])
        np.testing.assert_allclose(r.inclusion_probabilities, [.25, .25, .25, .25, .5, .5])
        self.assertAlmostEqual(float(r.sample_weights @ outcomes[r.indices]), 2 / 3)
        self.assertNotAlmostEqual(float(outcomes[r.indices].mean()), 2 / 3)

    def test_targeted_random_known_unequal_probability_and_unbiased_mean(self):
        ids = [str(i) for i in range(10)]
        risk = [0.] * 8 + [1.] * 2
        r = select("random_plus_targeted", ids, 4, risk=risk, targeted_fraction=.5)
        np.testing.assert_allclose(r.inclusion_probabilities, [.25] * 8 + [1.] * 2)
        self.assertAlmostEqual(float(r.sample_weights @ np.array(risk)[r.indices]), .2)
        # A census of the targeted cell does not make the remaining population
        # unobservable; all ordinary rows still have positive probability.
        samples = np.zeros(10)
        estimates = []
        outcomes = np.array([0., 0., 0., 1., 1., 1., 1., 1., 0., 1.])
        for seed in range(2000):
            run = select("random_plus_targeted", ids, 4, risk=risk, seed=seed)
            samples[run.indices] += 1
            estimates.append(float(run.sample_weights @ outcomes[run.indices]))
        np.testing.assert_allclose(samples / 2000, r.inclusion_probabilities, atol=.035)
        self.assertAlmostEqual(float(np.mean(estimates)), outcomes.mean(), delta=.025)

    def test_no_high_risk_reduces_to_stratified_sampling(self):
        base = self.call("stratified_random")
        r = select("random_plus_targeted", self.ids, 4, strata=self.strata, risk=[0.] * 6, seed=3)
        np.testing.assert_array_equal(base.indices, r.indices)
        np.testing.assert_allclose(base.sample_weights, r.sample_weights)

    def test_centroid_euclidean_selection_and_stratum_mass(self):
        ids = ["left", "center", "right", "small"]
        r = select("centroid_proximity", ids, 2, strata=["a", "a", "a", "b"],
                   embeddings=np.array([[1.], [2.], [9.], [100.]]))
        self.assertEqual(r.metadata["selected_ids"], ["center", "small"])
        np.testing.assert_allclose(r.sample_weights, [.75, .25])
        self.assertIsNone(r.inclusion_probabilities)

    def test_facility_covers_distinct_clusters_and_represents_masses(self):
        ids = ["a", "b", "c", "d", "e"]
        x = np.array([[1., 0.]] * 3 + [[-1., 0.]] * 2)
        r = select("facility_location", ids, 2, embeddings=x)
        self.assertEqual(r.metadata["selection_order"], ["a", "d"])
        np.testing.assert_allclose(r.sample_weights, [.6, .4])
        self.assertIsNone(r.inclusion_probabilities)

    def test_facility_lazy_greedy_matches_explicit_objective(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(15, 5))
        ids = [f"{i:02d}" for i in range(15)]
        r = select("facility_location", ids, 7, embeddings=x)
        normalized = x / np.linalg.norm(x, axis=1)[:, None]
        sim = np.clip((normalized @ normalized.T + 1) / 2, 0, 1)
        np.fill_diagonal(sim, 1.)
        chosen, covered = [], np.zeros(15)
        for _ in range(7):
            gains = np.maximum(sim - covered[:, None], 0.).sum(axis=0)
            gains[chosen] = -np.inf
            picked = int(np.argmax(gains))
            chosen.append(picked)
            covered = np.maximum(covered, sim[:, picked])
        self.assertEqual(r.metadata["selection_order"], [ids[i] for i in chosen])

    def test_minhash_covers_distinct_text_clusters_and_mass(self):
        ids = ["a", "b", "c", "d", "e"]
        texts = ["red green blue"] * 3 + ["one two three"] * 2
        r = select("minhash_farthest_first", ids, 2, texts=texts)
        self.assertEqual(r.metadata["selection_order"], ["a", "d"])
        np.testing.assert_allclose(r.sample_weights, [.6, .4])
        self.assertIsNone(r.inclusion_probabilities)

    def test_all_identical_rows_ties_and_census(self):
        for policy in ["centroid_proximity", "facility_location", "minhash_farthest_first"]:
            with self.subTest(policy=policy):
                kwargs = {"embeddings": np.ones((3, 2)), "texts": ["same"] * 3}
                small = select(policy, ["c", "a", "b"], 2, **kwargs)
                self.assertEqual(small.metadata["selected_ids"], ["a", "b"])
                full = select(policy, ["c", "a", "b"], 3, **kwargs)
                np.testing.assert_allclose(full.sample_weights, np.full(3, 1 / 3))

    def test_deterministic_policies_ignore_acquisition_seed(self):
        for policy in ["centroid_proximity", "facility_location", "minhash_farthest_first"]:
            with self.subTest(policy=policy):
                np.testing.assert_array_equal(self.call(policy, seed=1).indices,
                                              self.call(policy, seed=999).indices)

    def test_rejects_impossible_budget_ids_and_malformed_inputs(self):
        for budget in [-1, 0, 7, 1.5, True]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.call("uniform_random", budget=budget)
        with self.assertRaises(ValueError):
            select("uniform_random", ["a", "a"], 1)
        for policy in ["stratified_random", "centroid_proximity", "random_plus_targeted"]:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.call(policy, budget=1)
        with self.assertRaises(ValueError):
            select("facility_location", ["a"], 1, embeddings=[[0., 0.]])
        with self.assertRaises(ValueError):
            select("centroid_proximity", ["a"], 1, embeddings=[[np.nan]])
        with self.assertRaises(ValueError):
            select("random_plus_targeted", ["a"], 1, risk=[np.nan])
        with self.assertRaises(ValueError):
            select("random_plus_targeted", ["a"], 1, risk=[1.], targeted_fraction=1.)
        with self.assertRaises(ValueError):
            select("minhash_farthest_first", ["a"], 1, texts=[None])


if __name__ == "__main__":
    unittest.main()
