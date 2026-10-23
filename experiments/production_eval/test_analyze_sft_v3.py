import unittest

from experiments.production_eval.analyze_sft_v3 import bootstrap_families, paired_family_rows, paired_arm_rows, format_diagnostics


class AnalysisTests(unittest.TestCase):
    def test_schema_diagnostics_reject_duplicates_and_boolean_integers(self):
        self.assertFalse(format_diagnostics('{"stock":[],"total_units":0,"total_units":1}', "inventory_tracking")["strict_json_valid"])
        self.assertFalse(format_diagnostics('{"stock":[],"total_units":true}', "inventory_tracking")["required_schema_valid"])
        self.assertTrue(format_diagnostics('{"stock":[],"total_units":0}', "inventory_tracking")["required_schema_valid"])
        self.assertFalse(format_diagnostics('[]', "structured_records")["required_schema_valid"])

    def test_identical_model_has_zero_paired_delta_in_every_resample(self):
        rows = [{"family_id": str(i), "stratum": "a" if i < 5 else "b",
                 "baseline": i % 2, "trained": i % 2, "delta": 0} for i in range(10)]
        result = bootstrap_families(rows, repetitions=100)
        self.assertEqual(result["delta"], {"estimate": 0., "ci95_conditional_family_bootstrap": [0., 0.]})

    def test_strata_weight_equal_even_when_family_counts_differ(self):
        rows = [{"family_id": str(i), "stratum": "large", "baseline": 0, "trained": 0, "delta": 0} for i in range(20)]
        rows.append({"family_id": "last", "stratum": "small", "baseline": 0, "trained": 1, "delta": 1})
        self.assertEqual(bootstrap_families(rows, repetitions=100)["trained"]["estimate"], 0.5)

    def test_sibling_wordings_are_aggregated_into_one_sampling_unit(self):
        tasks = {"a": {"family_id": "one", "role": "reference", "domain": "d", "regime": "r", "variant": "base"},
                 "b": {"family_id": "one", "role": "reference", "domain": "d", "regime": "r", "variant": "paraphrase"}}
        baseline = {"a": {"exact_match": False}, "b": {"exact_match": True}}
        trained = {"a": {"exact_match": True}, "b": {"exact_match": True}}
        rows = paired_family_rows(tasks, [trained], baseline)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta"], 0.5)
        self.assertEqual(rows[0]["baseline"], 0.5)

    def test_arm_comparison_averages_seeds_without_changing_family_sample_size(self):
        tasks = {"a": {"family_id": "one", "role": "reference", "domain": "d", "regime": "r", "variant": "base"},
                 "b": {"family_id": "one", "role": "reference", "domain": "d", "regime": "r", "variant": "paraphrase"}}
        zero = {i: {"exact_match": False} for i in tasks}
        one = {i: {"exact_match": True} for i in tasks}
        rows = paired_arm_rows(tasks, {17: zero, 29: one}, {17: zero, 29: zero})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta"], 0.5)
        identical = paired_arm_rows(tasks, {17: zero, 29: one}, {17: zero, 29: one})
        self.assertEqual(bootstrap_families(identical, repetitions=50)["delta"]["ci95_conditional_family_bootstrap"], [0., 0.])
        with self.assertRaises(ValueError):
            paired_arm_rows(tasks, {17: zero}, {17: zero, 29: one})


if __name__ == "__main__":
    unittest.main()
