"""Sampling identities, label boundaries and independent exhaustive checks."""
import itertools
import unittest

import numpy as np

from experiments.production_eval.refresh_estimators_v4 import (
    current_sample_mean, exact_conditional_risk, nearest_group_predictions,
    nested_stratified_samples, residual_estimate, resampling_diagnostics,
    weighted_population_mean,
)


class RefreshEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.groups = dict(a="A", b="A", c="B", d="B", e="B")
        self.weights = dict(A=.3, B=.7)
        self.predictions = dict(a=.1, b=.9, c=.4, d=.7, e=.2)

    def test_unbiasedness_and_variance_for_every_binary_population_exhaustively(self):
        ids = sorted(self.groups)
        samples = list(itertools.product(["a", "b"], ["c", "d", "e"]))
        for values in itertools.product((0., 1.), repeat=len(ids)):
            outcomes = dict(zip(ids, values))
            truth = .3*(outcomes["a"]+outcomes["b"])/2 + .7*sum(outcomes[i] for i in "cde")/3
            estimates = [residual_estimate(self.predictions, {i: outcomes[i] for i in sample}, sample,
                                          self.groups, self.weights) for sample in samples]
            risk = exact_conditional_risk(self.predictions, outcomes, self.groups, self.weights, {"A": 1, "B": 1})
            self.assertAlmostEqual(np.mean(estimates), truth)
            self.assertAlmostEqual(risk["variance"], np.var(estimates))
            self.assertAlmostEqual(risk["rmse"] ** 2, np.mean((np.array(estimates)-truth)**2))

    def test_group_constant_predictor_cancels_exactly_to_same_fresh_sample(self):
        predictions = {i: .17 if g == "A" else .83 for i, g in self.groups.items()}
        outcomes = dict(a=0., b=1., c=1., d=0., e=1.)
        for a, b in itertools.product(itertools.combinations("ab", 1), itertools.combinations("cde", 2)):
            sample = list(a+b); labels = {i: outcomes[i] for i in sample}
            self.assertAlmostEqual(residual_estimate(predictions, labels, sample, self.groups, self.weights),
                                   current_sample_mean(labels, sample, self.groups, self.weights))
        constant_risk = exact_conditional_risk(predictions, outcomes, self.groups, self.weights, {"A": 1, "B": 2})
        fresh_risk = exact_conditional_risk(dict.fromkeys(outcomes, 0.), outcomes, self.groups, self.weights, {"A": 1, "B": 2})
        self.assertAlmostEqual(constant_risk["variance"], fresh_risk["variance"])

    def test_no_clipping_and_paired_deltas_can_be_negative(self):
        groups, weights = dict(a="A", b="A"), dict(A=1.)
        self.assertEqual(residual_estimate(dict(a=1., b=0.), dict(b=1.), ["b"], groups, weights), 1.5)
        self.assertEqual(weighted_population_mean(dict(a=-1., b=0.), groups, weights), -.5)

    def test_rejects_unacquired_labels_unknown_sample_and_missing_group(self):
        sample = ["a", "c"]
        labels = dict(a=0., c=1.)
        bad = [({**labels, "d": 0.}, sample), ({"a": 0.}, sample),
               (labels, ["a", "a", "c"]), ({"a": 0.}, ["a"]),
               ({"a": 0., "unknown": 0.}, ["a", "unknown"])]
        for acquired, ids in bad:
            with self.assertRaises(ValueError): residual_estimate(self.predictions, acquired, ids, self.groups, self.weights)
        with self.assertRaises(ValueError): residual_estimate({**self.predictions, "d": float("nan")}, labels, sample, self.groups, self.weights)
        with self.assertRaises(ValueError): residual_estimate(self.predictions, labels, sample, self.groups, dict(A=.4, B=.7))

    def test_census_zero_variance_and_perfect_prediction(self):
        outcomes = dict(a=0., b=1., c=1., d=0., e=1.)
        full = residual_estimate(self.predictions, outcomes, sorted(outcomes), self.groups, self.weights)
        self.assertAlmostEqual(full, weighted_population_mean(outcomes, self.groups, self.weights))
        self.assertEqual(exact_conditional_risk(self.predictions, outcomes, self.groups, self.weights, dict(A=2, B=3))["variance"], 0.)
        self.assertEqual(exact_conditional_risk(outcomes, outcomes, self.groups, self.weights, dict(A=1, B=1))["variance"], 0.)

    def test_samples_nested_and_invariant_to_mapping_order(self):
        groups = {str(i): "A" if i < 10 else "B" for i in range(20)}
        sample = nested_stratified_samples(groups, [1, 3, 5], 192)
        self.assertEqual(sample, nested_stratified_samples(dict(reversed(list(groups.items()))), [1, 3, 5], 192))
        self.assertTrue(set(sample[2]) < set(sample[6]) < set(sample[10]))
        for budget, ids in sample.items():
            self.assertEqual(len(ids), budget)
            self.assertEqual(sum(groups[i] == "A" for i in ids), budget//2)
        for invalid in ([0], [11], [3, 1], [1, 1], [True]):
            with self.assertRaises(ValueError): nested_stratified_samples(groups, invalid, 192)

    def test_neighbor_is_within_group_stable_ties_and_inputs_unmodified(self):
        construction = dict(a="A", b="A", c="B")
        current = dict(x="A", y="B")
        source = np.array([[2., 0.], [4., 0.], [0., 3.]])
        target = np.array([[2., 0.], [2., 0.]])
        before = source.copy(); before_target = target.copy()
        predictions, neighbors = nearest_group_predictions(["a", "b", "c"], dict(a=0., b=1., c=.5), construction, current, source, target)
        self.assertEqual(neighbors, dict(x="a", y="c"))
        self.assertEqual(predictions, dict(x=0., y=.5))
        np.testing.assert_array_equal(source, before); np.testing.assert_array_equal(target, before_target)
        with self.assertRaises(ValueError):
            nearest_group_predictions(["a", "b", "c"], dict(a=0., b=1., c=.5, x=1.), construction, current, source, target)
        with self.assertRaises(ValueError):
            nearest_group_predictions(["a", "b", "c"], dict(a=0., b=1., c=.5), construction, dict(x="C", y="B"), source, target)

    def test_resampling_keeps_constant_proxy_control_and_budget_counts(self):
        predictions = {i: .17 if g == "A" else .83 for i, g in self.groups.items()}
        outcomes = dict(a=0., b=1., c=1., d=0., e=1.)
        rows = resampling_diagnostics(predictions, outcomes, self.groups, self.weights, [1, 2], list(range(50)))
        self.assertEqual(len(rows), 4)
        for budget in (2, 4):
            a, b = [r for r in rows if r["budget"] == budget]
            for key in ("bias", "rmse", "median_absolute_error", "p90_absolute_error"):
                self.assertAlmostEqual(a[key], b[key])
            self.assertEqual(a["draws"], 50)


if __name__ == "__main__":
    unittest.main()
