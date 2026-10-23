from collections import Counter
import copy
import unittest
from unittest.mock import patch

from experiments.production_eval import refresh_initial_selection_v4 as revision
from experiments.production_eval.synthetic_refresh_v4 import BASE, load_jsonl


class InitialSelectionRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = revision.sorted_scope(load_jsonl(BASE / "draft_dataset/inputs_construction.jsonl"), "construction", "A")

    def test_one_rng_call_per_seed_exact_marginals_and_all_groups(self):
        with patch.object(revision, "select", wraps=revision.select) as sampler:
            runs = revision.revised_random_sets(self.rows)
        self.assertEqual(sampler.call_count, 20)
        for call in sampler.call_args_list:
            self.assertEqual(call.args[0], "stratified_random")
            self.assertEqual(len(call.args[1]), 200)
            self.assertEqual(call.args[2], 80)
            self.assertEqual(len(call.kwargs["strata"]), 200)
        for run in runs:
            self.assertEqual(run["sample_weights"], [1/80] * 80)
            self.assertEqual(run["population_inclusion_probabilities"], [.4] * 200)
            self.assertEqual(len(set(run["selected_family_ids"])), 80)
            self.assertEqual([len(c["selected_family_ids"]) for c in run["cells"]], [20] * 4)

    def test_group_ordinal_selections_are_not_identically_reset(self):
        runs = revision.revised_random_sets(self.rows)
        for run in runs:
            ordinals = [tuple(c["construction_families"].index(i) for i in c["selected_family_ids"]) for c in run["cells"]]
            self.assertGreater(len(set(ordinals)), 1)

    def test_reproducible_and_future_labels_forbidden(self):
        self.assertEqual(revision.revised_random_sets(self.rows), revision.revised_random_sets(copy.deepcopy(self.rows)))
        changed = copy.deepcopy(self.rows)
        changed[0]["answer"] = "hidden answer"
        with self.assertRaises(ValueError):
            revision.revised_random_sets(changed)
        changed = copy.deepcopy(self.rows)
        changed[0]["partition"] = "future"
        with self.assertRaises(ValueError):
            revision.revised_random_sets(changed)


if __name__ == "__main__":
    unittest.main()
