import unittest

from experiments.production_eval.repeat_pair_metrics import pair_metrics
from experiments.production_eval.test_budget_sensitivity import record, response


class PairMetricsTest(unittest.TestCase):
    def test_prediction_changes_not_validity_or_raw_text_changes(self):
        rows = [record(str(i), str(i)) for i in range(4)]
        a = {r["example_id"]: {**response(r), "raw_output_text": "text a", "finish_reason": "stop", "usage": {"completion_tokens": 10}} for r in rows}
        b = {key: {**value, "raw_output_text": "text b"} for key, value in a.items()}
        b["1"]["answer"] = "False"
        b["2"].update(answer=None, format_status="server_parser_invalid", usage=None, finish_reason=None)
        a["3"].update(answer=None, format_status="server_parser_invalid", usage=None, finish_reason=None)
        b["3"].update(answer=None, format_status="server_parser_invalid", usage=None, finish_reason=None)
        result = pair_metrics(rows, a, b)["summaries"][0]
        self.assertEqual(result["all_pairs"]["both_valid_prediction_pairs"], 2)
        self.assertEqual(result["all_pairs"]["changed_canonical_predictions_both_valid"], 1)
        self.assertEqual(result["all_pairs"]["correctness_disagreements"], 2)
        self.assertEqual(result["all_pairs"]["raw_text_changed"], 4)
        self.assertEqual(result["conditional_both_stop_under_1024"]["rows"], 2)
        self.assertEqual(result["all_pairs"]["format_transitions"], {"both_valid": 2, "valid_to_invalid": 1, "both_invalid": 1})

    def test_unchanged_nonbinding_pairs_in_denominator_and_whole_family_missing(self):
        rows = [record("a", "f"), record("b", "f"), record("c", "g")]
        a = {r["example_id"]: {**response(r), "raw_output_text": "same", "finish_reason": "stop", "usage": {"completion_tokens": 1}} for r in rows}
        b = {key: value for key, value in a.items() if key != "b"}
        result = pair_metrics(rows, a, b)["summaries"][0]
        self.assertEqual(result["conditional_both_stop_under_1024"]["rows"], 2)
        self.assertEqual(result["conditional_both_stop_under_1024"]["raw_text_changed"], 0)
        self.assertEqual(result["complete_families"], 1)
        self.assertIsNone(result["full_family_delta"])
        self.assertEqual(result["all_pairs"]["unknown_b"], 1)


if __name__ == "__main__":
    unittest.main()
