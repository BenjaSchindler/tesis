import unittest

from experiments.production_eval.field_diagnostics import diagnose, set_scores, aggregate, strict_json


class FieldDiagnosticTests(unittest.TestCase):
    def test_correct_set_wrong_order_is_separate(self):
        value = set_scores(["b", "a"], ["a", "b"], True)
        self.assertEqual(value["f1"], 1)
        self.assertFalse(value["order_compliant"])
        duplicate = set_scores(["a", "a", "b"], ["a", "b"], True)
        self.assertEqual(duplicate["f1"], 1)
        self.assertEqual(duplicate["duplicate_count"], 1)

    def test_missing_response_fails_fields_without_zero_error_imputation(self):
        expected = {"selected_ids": ["a"], "total_units": 5, "total_value": 20}
        result = diagnose(None, expected, "structured_records", available=False)
        self.assertEqual(result["scores"]["selected_id_set_f1"], 0)
        self.assertEqual(result["scores"]["total_units_exact"], 0)
        self.assertEqual(result["scores"]["total_units_capped_relative_error"], 1)
        self.assertIsNone(result["absolute_errors"]["total_units_absolute_error"])

    def test_duplicate_inventory_items_cannot_cherry_pick_correct_row(self):
        expected = {"stock": [{"item": "a", "east": 5, "west": 4}], "total_units": 9}
        actual = {"stock": [{"item": "a", "east": 5, "west": 4}, {"item": "a", "east": 0, "west": 0}], "total_units": 9}
        result = diagnose(actual, expected, "inventory_tracking")
        self.assertEqual(result["scores"]["item_set_f1"], 1)
        self.assertEqual(result["scores"]["item_duplicate_flag"], 1)
        self.assertEqual(result["scores"]["expected_item_location_count_accuracy"], 0)
        self.assertEqual(result["scores"]["reported_total_equals_sum_of_reported_stock"], 1)
        self.assertEqual(result["scores"]["primary_strict_exact_match_recomputed"], 0)

    def test_true_is_not_integer_answer(self):
        expected = {"selected_ids": ["a"], "total_units": 1, "total_value": 2}
        actual = {"selected_ids": ["a"], "total_units": True, "total_value": 2}
        result = diagnose(actual, expected, "structured_records")
        self.assertEqual(result["scores"]["total_units_exact"], 0)

    def test_family_weight_and_missing_rows_are_preserved(self):
        truth = {"selected_ids": ["a"], "total_units": 1, "total_value": 2}
        good = diagnose(truth, truth, "structured_records")
        bad = diagnose(None, truth, "structured_records", available=False)
        common = {"config_id": "c", "domain": "structured_records", "role": "reference", "regime": "stationary"}
        rows = [{**common, "family_id": "one", "availability": "available", **good} for _ in range(4)]
        rows.append({**common, "family_id": "two", "availability": "missing_or_transport_failure", **bad})
        group = aggregate(rows)[0]
        self.assertEqual(group["family_mean_metrics"]["total_units_exact"], 0.5)
        self.assertIsNone(group["absolute_error_diagnostics"]["total_units_absolute_error"]["mean_over_all_expected_families"])

    def test_strict_parser_never_removes_fences_or_duplicate_keys(self):
        self.assertIsNone(strict_json('```json\n{"a":1}\n```'))
        self.assertIsNone(strict_json('{"a":1,"a":2}'))


if __name__ == "__main__":
    unittest.main()
