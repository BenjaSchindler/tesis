"""Reference-label, parsing, split-integrity, and strict-scoring invariants."""
import json
import unittest

from experiments.production_eval.benchmark_data import (
    SOURCES, assign_roles, canonical_answer, check_answer, normalized_row,
    parse_source, sha256, verify_boolean_expression,
)


class AnswerCheckerTests(unittest.TestCase):
    def test_equivalent_exact_decimal_notations(self):
        for value in ["42", " 42.000 ", "+42", "0042", "00042.0"]:
            self.assertEqual(check_answer(value, "42", "decimal")["correct"], True)
        self.assertTrue(check_answer("1,234.50", "1234.5", "decimal")["correct"])
        self.assertTrue(check_answer("-0.000", "0", "decimal")["correct"])
        self.assertTrue(check_answer("-12.50", "-12.5", "decimal")["correct"])

    def test_exactness_without_rounding_or_tolerance(self):
        # Longer than default Decimal precision: normalization must not round.
        number = "1234567890123456789012345678901234567890.123456789"
        self.assertEqual(canonical_answer(number, "decimal"), number)
        self.assertFalse(check_answer("42.000000000000001", "42", "decimal")["correct"])
        self.assertFalse(check_answer("-42", "42", "decimal")["correct"])

    def test_invalid_and_ambiguous_numbers_never_match_substring(self):
        invalid = ["The answer is 42", "42 dollars", "$42", "42 or 43", "41\n42", "#### 42",
                   "\\boxed{42}", "42.", ".42", "4,2", "1,23,456", "42%", "NaN", "Infinity",
                   "4.2e1", "84/2", "0x2a", "４２", "4_2", "", " ", 42, 42.0, True, None,
                   {"answer": "42"}, ["42"], "9" * 257]
        for value in invalid:
            with self.subTest(value=value):
                result = check_answer(value, "42", "decimal")
                self.assertFalse(result["correct"])
                self.assertFalse(result["valid_format"])
                self.assertEqual(result["error"], "invalid_final_answer_format")

    def test_wrong_answer_is_not_format_error(self):
        result = check_answer("43", "42", "decimal")
        self.assertFalse(result["correct"])
        self.assertTrue(result["valid_format"])
        self.assertIsNone(result["error"])

    def test_boolean_requires_one_boolean_string(self):
        for value in ["True", "TRUE", " true "]:
            self.assertTrue(check_answer(value, "True", "boolean")["correct"])
        self.assertFalse(check_answer("False", "True", "boolean")["correct"])
        for value in ["True or False", "not False", "True.", "yes", "1", True, 1, None]:
            self.assertFalse(check_answer(value, "True", "boolean")["valid_format"])

    def test_invalid_reference_or_type_raises(self):
        with self.assertRaises(ValueError):
            check_answer("42", "unknown", "decimal")
        with self.assertRaises(ValueError):
            check_answer("42", "42", "approximate")


class SourceAndSplitTests(unittest.TestCase):
    def make_rows(self, count=20):
        return [normalized_row(SOURCES[0], "a" * 40, i, f"How much is {i} plus 1?", str(i + 1), "decimal")
                for i in range(count)]

    def test_split_counts_disjointness_order_independence(self):
        rows = self.make_rows()
        counts = {"development": 3, "construction": 8, "reference": 7}
        result, summary = assign_roles(rows, counts, 123)
        reversed_result, _ = assign_roles(list(reversed(rows)), counts, 123)
        self.assertEqual(result, reversed_result)
        self.assertEqual(summary["role_counts"], counts)
        self.assertEqual(summary["unused_unique_families"], 2)
        sets = [{row["family_id"] for row in result if row["data_role"] == role} for role in counts]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertTrue(all(row["data_role"] == "unassigned" for row in rows))

    def test_split_is_independent_of_labels_and_changes_with_seed(self):
        rows = self.make_rows(50)
        counts = {"development": 5, "construction": 10, "reference": 10}
        selected, _ = assign_roles(rows, counts, 2)
        altered, _ = assign_roles([{**row, "answer": "999"} for row in rows], counts, 2)
        self.assertEqual([row["example_id"] for row in selected], [row["example_id"] for row in altered])
        other, _ = assign_roles(rows, counts, 3)
        self.assertNotEqual([row["example_id"] for row in selected], [row["example_id"] for row in other])

    def test_duplicate_inputs_removed_before_roles_and_conflicts_raise(self):
        rows = self.make_rows(10)
        duplicate = normalized_row(SOURCES[0], "a" * 40, 99, "  HOW much is 0  plus 1? ", "1", "decimal")
        result, summary = assign_roles(rows + [duplicate], {"development": 3, "reference": 7}, 123)
        self.assertEqual(len(result), 10)
        self.assertEqual(summary["duplicates_removed"], 1)
        with self.assertRaises(ValueError):
            assign_roles(rows + [{**duplicate, "answer": "999"}], {"reference": 1}, 123)
        with self.assertRaises(ValueError):
            assign_roles(rows, {"reference": 11}, 123)

    def test_gsm8k_final_label_extraction_ignores_reasoning_numbers(self):
        payload = json.dumps({"question": "What is 2 plus 3?", "answer": "2 plus 3 = 5.\n#### 5"}).encode()
        rows, _ = parse_source(SOURCES[0], "a" * 40, payload)
        self.assertEqual(rows[0]["answer"], "5")
        self.assertEqual(rows[0]["input_hash"], sha256("What is 2 plus 3?"))
        self.assertFalse(rows[0]["label_independently_verified"])
        for answer in ["5", "#### 5 or 6", "#### 5\n#### 5"]:
            with self.assertRaises(ValueError):
                parse_source(SOURCES[0], "a" * 40, json.dumps({"question": "2+3?", "answer": answer}).encode())

    def test_boolean_evaluator_precedence_nested_not_and_no_execution(self):
        cases = {"True or False and False is": "True", "( True or False ) and False is": "False",
                 "not not not True is": "False", "not ( False or not True ) is": "True"}
        for expression, expected in cases.items():
            self.assertEqual(verify_boolean_expression(expression), expected)
        for bad in ["True or __import__('os').system('id') is", "False and 1 is", "1 is", "True == True is",
                    "True + False is", "[True] is", "True"]:
            with self.assertRaises((ValueError, SyntaxError)):
                verify_boolean_expression(bad)

    def test_boolean_published_label_disagreement_aborts(self):
        payload = {"canary": "test", "examples": [{"input": "True and False is", "target": "True"}]}
        with self.assertRaises(ValueError):
            parse_source(SOURCES[1], "a" * 40, json.dumps(payload).encode())
        payload["examples"][0]["target"] = "False"
        rows, summary = parse_source(SOURCES[1], "a" * 40, json.dumps(payload).encode())
        self.assertTrue(rows[0]["label_independently_verified"])
        self.assertEqual(summary["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
