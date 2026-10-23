"""Independent exhaustive enumeration checks for exact finite-design moments."""
import copy
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.production_eval.matched_weight_control import (
    naive_copy, stratified_moments, validate_design, verify_published_sources,
)


class MatchedWeightTests(unittest.TestCase):
    def test_previously_published_selection_hash_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "outcome_aware").mkdir()
            files = []
            for name in ("selected_families.json", "selector_analysis.json"):
                (base / "outcome_aware" / name).write_text("{}")
                files.append({"path": "outcome_aware/" + name, "sha256": hashlib.sha256(b"{}").hexdigest()})
            (base / "completed_primary_manifest.json").write_text(json.dumps({"files": files}))
            self.assertEqual(len(verify_published_sources(base)["verified_files"]), 2)
            (base / "outcome_aware/selected_families.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "Previously completed source hash mismatch"):
                verify_published_sources(base)

    def test_moments_equal_exhaustive_enumeration_for_all_binary_populations(self):
        groups, quotas = [[0, 1], [2, 3, 4, 5]], [2, 1]
        samples = [(0, 1, low) for low in range(2, 6)]
        # Fixed target differs from the finite-pool mean, testing the additional
        # squared-bias term without relying on the estimator implementation.
        for values in itertools.product((0., 1.), repeat=6):
            for weighted in (True, False):
                estimates = [sum(values[i] * (1/6 if i < 2 else 4/6) for i in sample)
                             if weighted else sum(values[i] for i in sample) / 3 for sample in samples]
                for target in (sum(values) / 6, .7):
                    got = stratified_moments(values, groups, quotas, target, weighted)
                    self.assertAlmostEqual(got["expectation"], np.mean(estimates))
                    self.assertAlmostEqual(got["variance"], np.var(estimates))
                    self.assertAlmostEqual(got["mse_to_target"], np.mean((np.array(estimates) - target) ** 2))
                    if weighted and target == sum(values) / 6:
                        self.assertAlmostEqual(got["bias_to_target"], 0.)

    def test_negative_fractional_deltas_and_multiple_partial_cells(self):
        values = [-1., -.25, .5, 0., .75, 1., -.5]
        groups, quotas = [[0, 1, 2], [3, 4, 5, 6]], [2, 2]
        samples = list(itertools.product(itertools.combinations(groups[0], 2), itertools.combinations(groups[1], 2)))
        for weighted in (True, False):
            expected = []
            for a, b in samples:
                if weighted:
                    estimate = (3/7)*sum(values[i] for i in a)/2 + (4/7)*sum(values[i] for i in b)/2
                else:
                    estimate = sum(values[i] for i in a+b)/4
                expected.append(estimate)
            got = stratified_moments(values, groups, quotas, .4, weighted)
            self.assertAlmostEqual(got["expectation"], np.mean(expected))
            self.assertAlmostEqual(got["variance"], np.var(expected))

    def test_census_singletons_and_constant_targets(self):
        for weighted in (True, False):
            got = stratified_moments([.5]*5, [[0], [1, 2, 3, 4]], [1, 2], -.5, weighted)
            self.assertEqual(got["variance"], 0.)
            self.assertEqual(got["mse_to_target"], 1.)
            got = stratified_moments([-1, 0, 1], [[0], [1, 2]], [1, 2], 0, weighted)
            self.assertEqual(got["variance"], 0.)
            self.assertEqual(got["mse_to_target"], 0.)

    def test_rejects_unknown_values_nonpartitions_and_zero_coverage(self):
        for values, groups, quotas, target in [
            ([0, float("nan")], [[0, 1]], [1], 0), ([0, 1], [[0], [0]], [1, 1], 0),
            ([0, 1], [[0]], [1], 0), ([0, 1], [[0], [1]], [0, 1], 0),
            ([0, 1], [[0, 1]], [3], 0), ([0, 1], [[0, 1]], [True], 0),
            ([0, 1], [[0, 1]], [1], float("nan")),
        ]:
            with self.assertRaises(ValueError):
                stratified_moments(values, groups, quotas, target, True)

    def fixture(self):
        return {"policy": "error_hybrid_ht", "budget": 3,
                "construction_family_ids": ["a", "b", "c", "d", "e", "f"],
                "selected_family_ids": ["a", "b", "d"], "sample_weights": [1/6, 1/6, 4/6],
                "population_inclusion_probabilities": [1, 1, .25, .25, .25, .25],
                "selection_metadata": {"cells": [
                    {"key": ["all", True], "population_count": 2, "sample_count": 2},
                    {"key": ["all", False], "population_count": 4, "sample_count": 1}]}}

    def test_frozen_design_and_same_sample_copy(self):
        run = self.fixture()
        flags = {family: family in "ab" for family in run["construction_family_ids"]}
        self.assertEqual(validate_design(run, flags), ([[0, 1], [2, 3, 4, 5]], [2, 1]))
        naive = naive_copy(run)
        self.assertEqual(naive["selected_family_ids"], run["selected_family_ids"])
        self.assertEqual(naive["sample_weights"], [1/3]*3)
        self.assertIsNone(naive["population_inclusion_probabilities"])
        self.assertEqual(run["sample_weights"], [1/6, 1/6, 4/6])
        for mutation in ("probability", "weight", "quota", "selection", "extra_flag"):
            changed = copy.deepcopy(run); risk = dict(flags)
            if mutation == "probability": changed["population_inclusion_probabilities"][2] = .5
            if mutation == "weight": changed["sample_weights"] = [1/3]*3
            if mutation == "quota": changed["selection_metadata"]["cells"][0]["sample_count"] = 1
            if mutation == "selection": changed["selected_family_ids"] = ["a", "c", "d"]
            if mutation == "extra_flag": risk["reference"] = False
            with self.assertRaises(ValueError): validate_design(changed, risk)


if __name__ == "__main__":
    unittest.main()
