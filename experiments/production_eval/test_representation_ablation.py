"""Semantic invariants for retaining long-input payloads and fitting prefixes."""
import unittest

import numpy as np

from experiments.production_eval.representation_ablation import (
    VARIANTS, aggregate, fit_prefixes, make_plan, matching_prefix,
)


class RepresentationTests(unittest.TestCase):
    def test_chunking_preserves_every_position_exactly_once(self):
        for length in [1, 7, 8, 9, 100, 383, 1000]:
            for variant in ["chunk_mean", "prefix_payload_mix"]:
                segments, weights = make_plan(length, 8, variant, min(43, length - 1))
                self.assertEqual([i for segment in segments for i in segment], list(range(length)))
                self.assertTrue(all(0 < len(segment) <= 8 for segment in segments))
                self.assertAlmostEqual(sum(weights), 1.0)

    def test_balanced_context_uses_both_ends_without_overlap(self):
        segments, weights = make_plan(30, 9, "head_tail")
        self.assertEqual(segments, [[0, 1, 2, 3, 4, 26, 27, 28, 29]])
        self.assertEqual(weights, [1.0])

    def test_short_input_unchanged_in_single_window_variants(self):
        for variant in ["head", "tail", "head_tail"]:
            self.assertEqual(make_plan(7, 8, variant), ([list(range(7))], [1.0]))

    def test_prefix_model_never_uses_an_unseen_pair_or_full_duplicates(self):
        prefix = list(range(80))
        left, right = prefix + [100] * 20, prefix + [200] * 20
        self.assertEqual(fit_prefixes([left, left]), [])
        self.assertEqual(fit_prefixes([prefix, left]), [])
        fitted = fit_prefixes([left, right])
        self.assertEqual(fitted, [tuple(prefix)])
        self.assertEqual(matching_prefix([400] * 100, fitted), 0)
        self.assertEqual(matching_prefix(left, fitted), 80)
        # Adding a future example is not implicit in the fitted model.
        self.assertEqual(fit_prefixes([left]), [])

    def test_prefix_and_payload_have_equal_total_mass(self):
        segments, weights = make_plan(100, 8, "prefix_payload_mix", 80)
        self.assertAlmostEqual(sum(w for s, w in zip(segments, weights) if s[0] < 80), 0.5)
        self.assertAlmostEqual(sum(w for s, w in zip(segments, weights) if s[0] >= 80), 0.5)
        self.assertEqual(make_plan(100, 8, "prefix_payload_mix", 0), make_plan(100, 8, "chunk_mean"))

    def test_chunk_aggregate_matches_weighted_direction(self):
        embeddings = np.array([[1., 0.], [0., 1.]])
        result = aggregate(embeddings, [[0, 1]], [[0.75, 0.25]])
        expected = np.array([3., 1.]) / np.sqrt(10)
        np.testing.assert_allclose(result[0], expected, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(result[0])), 1.0)

    def test_invalid_plans_fail(self):
        for values in [(0, 8, "head", 0), (10, 1, "tail", 0), (10, 8, "missing", 0), (10, 8, "chunk_mean", 10)]:
            with self.assertRaises(ValueError):
                make_plan(*values)


if __name__ == "__main__":
    unittest.main()
