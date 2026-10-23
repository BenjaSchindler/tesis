"""Oracle, metamorphic, split-isolation and strict-grading checks."""
import copy
import json
import unittest

from experiments.production_eval.synthetic_tasks import (
    DOMAINS, canonical, generate, inventory_oracle, inventory_sql_oracle,
    json_exact_grade, records_oracle, records_sql_oracle, validate,
)


class ControlledTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate()

    def test_independent_oracles_and_family_metamorphisms(self):
        result = validate(self.rows)
        self.assertEqual(result["rows"], 400)
        self.assertEqual(result["families"], 100)
        self.assertEqual(result["independent_oracle_agreements"], 400)
        self.assertEqual(result["answer_changing_counterfactual_pairs"], 100)
        self.assertEqual(result["answer_preserving_paraphrase_pairs"], 200)

    def test_exact_split_counts_and_entity_seed_disjointness(self):
        for domain in DOMAINS:
            by_split = {split: [r for r in self.rows if r["task_family"] == domain and r["split"] == split]
                        for split in ("construction", "reference")}
            for rows in by_split.values():
                self.assertEqual(len(rows), 100)
            for key in ("data", "entities", "template_family"):
                left = {r["metadata"]["substream_seeds"][key] for r in by_split["construction"]}
                right = {r["metadata"]["substream_seeds"][key] for r in by_split["reference"]}
                self.assertFalse(left & right)
            entity_sets = []
            for rows in by_split.values():
                entity_sets.append({entry.get("id", entry.get("item")) for row in rows
                                    for entry in row["metadata"]["source_problem"].get(
                                        "records", row["metadata"]["source_problem"].get("initial", []))})
            self.assertFalse(entity_sets[0] & entity_sets[1])

    def test_generator_determinism_and_seed_sensitivity(self):
        self.assertEqual(self.rows, generate())
        changed = generate(123)
        self.assertFalse({r["example_id"] for r in self.rows} & {r["example_id"] for r in changed})

    def test_record_predicate_edges_and_empty_result(self):
        problem = {"target_team": "A", "records": [
            {"id": "keep", "team": "A", "active": True, "void": False, "units": 3, "unit_price": 7},
            {"id": "inactive", "team": "A", "active": False, "void": False, "units": 99, "unit_price": 99},
            {"id": "void", "team": "A", "active": True, "void": True, "units": 99, "unit_price": 99},
            {"id": "other", "team": "B", "active": True, "void": False, "units": 99, "unit_price": 99},
        ]}
        expected = {"selected_ids": ["keep"], "total_units": 3, "total_value": 21}
        self.assertEqual(records_oracle(problem), expected)
        self.assertEqual(records_sql_oracle(problem), expected)
        problem["target_team"] = "absent"
        empty = {"selected_ids": [], "total_units": 0, "total_value": 0}
        self.assertEqual(records_oracle(problem), empty)
        self.assertEqual(records_sql_oracle(problem), empty)

    def test_record_permutation_and_irrelevant_distractor_invariance(self):
        row = next(r for r in self.rows if r["task_family"] == "structured_records")
        problem = copy.deepcopy(row["metadata"]["source_problem"])
        expected = records_oracle(problem)
        problem["records"].reverse()
        problem["records"].append({"id": "extra", "team": "never-selected", "active": True,
                                   "void": False, "units": 100000, "unit_price": 100000})
        self.assertEqual(records_oracle(problem), expected)
        self.assertEqual(records_sql_oracle(problem), expected)

    def test_inventory_move_cancellation_and_conservation(self):
        problem = {"initial": [{"item": "A", "east": 10, "west": 5}], "operations": [
            {"step": 1, "kind": "move", "item": "A", "location": "east", "destination": "west", "quantity": 4, "cancelled": False},
            {"step": 2, "kind": "ship", "item": "A", "location": "west", "quantity": 99, "cancelled": True},
            {"step": 3, "kind": "receive", "item": "A", "location": "east", "quantity": 2, "cancelled": False},
            {"step": 4, "kind": "ship", "item": "A", "location": "west", "quantity": 3, "cancelled": False},
        ]}
        expected = {"stock": [{"item": "A", "east": 8, "west": 6}], "total_units": 14}
        self.assertEqual(inventory_oracle(problem), expected)
        self.assertEqual(inventory_sql_oracle(problem), expected)

    def test_counterfactual_changes_exactly_one_source_scalar(self):
        def differences(a, b):
            if isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                return sum(differences(a[key], b[key]) for key in a)
            if isinstance(a, list):
                self.assertEqual(len(a), len(b))
                return sum(differences(x, y) for x, y in zip(a, b))
            return int(a != b)
        bases = {r["family_id"]: r for r in self.rows if r["metadata"]["variant"] == "base"}
        for row in self.rows:
            if row["metadata"]["variant"] != "counterfactual":
                continue
            base = bases[row["family_id"]]
            self.assertEqual(differences(base["metadata"]["source_problem"], row["metadata"]["source_problem"]), 1)
            a, b = json.loads(base["answer"]), json.loads(row["answer"])
            self.assertEqual(b["total_units"] - a["total_units"], row["metadata"]["mutation"]["expected_total_units_delta"])

    def test_strict_grader_rejects_schema_and_type_errors(self):
        expected = canonical({"n": 1, "ids": ["a", "b"]})
        self.assertTrue(json_exact_grade('{"ids": ["a", "b"], "n": 1}', expected))
        for bad in ('{"n":true,"ids":["a","b"]}', '{"n":1.0,"ids":["a","b"]}',
                    '{"n":1,"ids":["b","a"]}', '{"n":1,"ids":["a","b"],"extra":0}',
                    '{"n":0,"n":1,"ids":["a","b"]}', '{"n":NaN,"ids":["a","b"]}',
                    '```json\n' + expected + '\n```', expected + ' trailing'):
            self.assertFalse(json_exact_grade(bad, expected), bad)


if __name__ == "__main__":
    unittest.main()
