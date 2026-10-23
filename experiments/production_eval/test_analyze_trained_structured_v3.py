import unittest

from experiments.production_eval.analyze_trained_structured_v3 import paired


class TrainedControlAnalysisTests(unittest.TestCase):
    def test_both_seeds_and_sibling_wordings_keep_equal_weight(self):
        tasks, baseline, first, second = {}, {}, {}, {}
        for family in range(4):
            for variant in ("base", "paraphrase"):
                identifier = f"{family}/{variant}"
                tasks[identifier] = {"role": "reference", "family_id": str(family), "domain": "structured_records",
                                     "regime": "stationary", "variant": variant}
                baseline[identifier] = {"exact_match": False}
                first[identifier] = {"exact_match": variant == "base"}
                second[identifier] = {"exact_match": variant == "paraphrase"}
        result = paired(tasks, [first, second], baseline)
        self.assertEqual(result["comparator"]["estimate"], 0.)
        self.assertEqual(result["treatment"]["estimate"], .5)
        self.assertEqual(result["delta"]["estimate"], .5)
        self.assertEqual(result["delta"]["ci95_conditional_family_bootstrap"], [.5, .5])


if __name__ == "__main__":
    unittest.main()
