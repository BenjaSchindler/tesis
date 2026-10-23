"""Integration checks for the answer-blind schema adapter, without model inference."""
import json
import unittest

import torch

from experiments.production_eval.structured_decoding_v3 import (
    DomainPrefixFilter, cpu_checks, input_only, mechanical_fixtures,
    prompt_tokens, schema_for_domain, tokenizer_cpu,
)


class StructuredDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.data = tokenizer_cpu()

    def permits(self, domain, output):
        row = next(r for r in mechanical_fixtures() if r["domain"] == domain)
        prompt = prompt_tokens(self.tokenizer, row)
        prefix = DomainPrefixFilter(self.data, domain, prompt)
        sequence = list(prompt)
        allowed = prefix(0, torch.tensor(sequence))
        for token in self.tokenizer(output, add_special_tokens=False)["input_ids"]:
            if token not in allowed:
                return False
            sequence.append(token)
            allowed = prefix(0, torch.tensor(sequence))
        return self.tokenizer.eos_token_id in allowed

    def test_arbitrary_values_and_twenty_eight_array_items(self):
        reports = cpu_checks(self.tokenizer, self.data)
        self.assertTrue(all(r["all_arbitrary_fixture_tokens_allowed"] for r in reports))

    def test_empty_arrays_and_reordered_fields_are_legal(self):
        self.assertTrue(self.permits("structured_records", '{"total_value":-3,"total_units":0,"selected_ids":[]}'))
        self.assertTrue(self.permits("inventory_tracking", '{"total_units":-4,"stock":[]}'))

    def test_unknown_identifiers_and_twenty_eight_inventory_items(self):
        value = {"stock": [{"item": f"not_in_input_{i}", "east": -i, "west": 99} for i in range(28)], "total_units": -87}
        self.assertTrue(self.permits("inventory_tracking", json.dumps(value)))

    def test_wrong_types_missing_fields_extra_fields_and_fences_rejected(self):
        examples = [
            '{"selected_ids":[],"total_units":true,"total_value":1}',
            '{"selected_ids":[],"total_units":1}',
            '{"selected_ids":[],"total_units":1,"total_value":1,"bonus":1}',
            '```json\n{"selected_ids":[],"total_units":1,"total_value":1}\n```',
            '{"selected_ids":[],"total_units":1+1,"total_value":1}',
        ]
        for output in examples:
            with self.subTest(output=output):
                self.assertFalse(self.permits("structured_records", output))

    def test_prompt_boundary_and_batch_are_enforced(self):
        row = mechanical_fixtures()[0]
        prompt = prompt_tokens(self.tokenizer, row)
        for batch, sequence in ((1, prompt), (0, prompt + [1]), (0, [1] + prompt[1:])):
            with self.subTest(batch=batch, extra=len(sequence)-len(prompt)):
                prefix = DomainPrefixFilter(self.data, row["domain"], prompt)
                with self.assertRaises(ValueError):
                    prefix(batch, torch.tensor(sequence))

    def test_answers_cannot_enter_prompt_and_hash_mismatch_rejected(self):
        row = mechanical_fixtures()[0]
        self.assertNotIn("answer", input_only({**row, "answer": "SECRET"}))
        with self.assertRaises(ValueError):
            prompt_tokens(self.tokenizer, {**row, "answer": "SECRET"})
        with self.assertRaises(ValueError):
            prompt_tokens(self.tokenizer, {**row, "input_text": row["input_text"] + " modified"})

    def test_schemas_have_no_answer_dependent_constraints(self):
        for domain in ("structured_records", "inventory_tracking"):
            encoded = json.dumps(schema_for_domain(domain))
            for key in ("enum", "const", "minItems", "maxItems", "minimum", "maximum", "pattern"):
                self.assertNotIn('"' + key + '"', encoded)


if __name__ == "__main__":
    unittest.main()
