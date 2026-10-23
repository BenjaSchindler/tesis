import hashlib
import unittest

from experiments.production_eval.analyze_structured_decoding_v3 import replay_comparison


def record(text, *, exact=False, valid=True, input_hash="input"):
    return {"response": text, "domain": "structured_records", "role": "reference",
            "input_hash": input_hash, "generated_tokens": len(text), "exact_match": exact,
            "required_schema_valid": valid, "response_sha256": hashlib.sha256(text.encode()).hexdigest()}


class ReplayComparisonTests(unittest.TestCase):
    def test_text_format_and_exactness_agreement_are_distinct(self):
        original = {"a": record("bad", valid=False), "b": record("one", exact=True), "c": record("two")}
        replay = {"a": record("new", valid=True), "b": record("two"), "c": record("two")}
        result = replay_comparison(replay, original)
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["counts"], {"response_identical": 1, "generated_token_count_identical": 3,
                                           "exact_match_identical": 2, "required_schema_valid_identical": 2})
        self.assertEqual(result["original_exact_match"], 1)
        self.assertEqual(result["replay_exact_match"], 0)

    def test_missing_empty_and_mismatched_inputs_rejected(self):
        original = {"a": record("same")}
        for replay in ({}, {"b": record("same")}, {"a": record("same", input_hash="changed")}):
            with self.subTest(replay=replay):
                with self.assertRaises(ValueError):
                    replay_comparison(replay, original)

    def test_replay_may_be_subset_but_keeps_all_replayed_rows(self):
        original = {"a": record("one"), "b": record("two")}
        result = replay_comparison({"b": original["b"]}, original)
        self.assertEqual(result["examples"], 1)
        self.assertEqual(result["rows"][0]["example_id"], "b")
        self.assertTrue(all(value == 1 for value in result["counts"].values()))


if __name__ == "__main__":
    unittest.main()
