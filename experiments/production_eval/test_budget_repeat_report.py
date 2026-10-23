import copy
from pathlib import Path
import tempfile
import unittest

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.budget_repeat_report import augment_hashes, common_prefix, read_bound, usage
from experiments.production_eval.repeat_pair_metrics import pair_metrics
from experiments.production_eval.test_budget_sensitivity import baseline_config, record, response


class RepeatReportTest(unittest.TestCase):
    def test_first_completed_model_output_is_required_even_when_later_is_valid(self):
        config = baseline_config(); row = record(); item = inference.prepare(row, config)
        first = {"example_id": row["example_id"], "input_hash": row["input_hash"], "config_id": config["config_id"],
                 "cache_key": item["cache_key"], "call_id": "first", "status": "success",
                 "format_status": "server_parser_invalid", "raw_response": None}
        later = {**first, "call_id": "second", "format_status": "valid"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "responses.jsonl"
            inference.RunJournal.append(Path(str(path) + ".attempts.jsonl"), first)
            inference.RunJournal.append(Path(str(path) + ".attempts.jsonl"), later)
            inference.RunJournal.append(path, later)
            with self.assertRaisesRegex(ValueError, "First completed"):
                read_bound(path, config, {row["example_id"]: item})
            path.write_text(inference.canonical(first) + "\n")
            observed, attempts = read_bound(path, config, {row["example_id"]: item})
            self.assertEqual(observed[row["example_id"]], first)
            self.assertEqual(len(attempts), 2)

    def test_returned_token_hashes_prefixes_use_complete_nonbinding_denominator(self):
        row = record(); a = {**response(row), "raw_output_text": "same", "finish_reason": "stop", "usage": {"completion_tokens": 1}}
        result = pair_metrics([row], {"a": a}, {"a": copy.deepcopy(a)})
        text_hash = inference.digest("same")
        cache = {text_hash: {"raw_text_sha256": text_hash, "token_sha256": inference.digest("[1,2]"), "token_ids": [1, 2]}}
        augment_hashes(result, {"a": a}, {"a": a}, cache)
        hashes = result["summaries"][0]["conditional_both_stop_under_1024"]["returned_text_hash_comparison"]
        self.assertEqual(hashes["comparable_pairs"], 1)
        self.assertEqual(hashes["identical_retokenized_sequences"], 1)
        self.assertEqual(hashes["both_at_least_32_retokens"], 0)
        self.assertEqual(hashes["median_common_prefix_retokens"], 2)
        self.assertEqual(common_prefix([1, 2, 4], [1, 2, 3]), 2)

    def test_usage_unknowns_not_imputed_to_zero_and_booleans_not_tokens(self):
        result = usage([{"charged_nanodollars": 0, "usage": {"completion_tokens": 7, "prompt_tokens": 12}},
                        {"charged_nanodollars": 0, "usage": None},
                        {"charged_nanodollars": 0, "usage": {"completion_tokens": True}}])
        self.assertEqual(result["completion_tokens"]["known"], 1)
        self.assertEqual(result["completion_tokens"]["unknown"], 2)
        self.assertEqual(result["completion_tokens"]["mean_known"], 7)


if __name__ == "__main__":
    unittest.main()
