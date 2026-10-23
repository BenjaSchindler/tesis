"""Independent parsing, oracle, leakage and deterministic-draft checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.production_eval import synthetic_refresh_v4 as v4


class RefreshDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((v4.BASE / "draft_protocol.json").read_text())
        cls.inputs, cls.gold, cls.programs = v4.generate(cls.protocol)

    def records(self):
        return {"target_team": "team_a", "records": [
            {"id": "row_z", "team": "team_a", "active": True, "void": False, "units": 3, "unit_price": 7},
            {"id": "row_a", "team": "team_a", "active": True, "void": False, "units": 2, "unit_price": 5},
            {"id": "row_b", "team": "team_b", "active": True, "void": False, "units": 99, "unit_price": 8},
            {"id": "row_c", "team": "team_a", "active": False, "void": False, "units": 99, "unit_price": 8},
            {"id": "row_d", "team": "team_a", "active": True, "void": True, "units": 99, "unit_price": 8}]}

    def inventory(self):
        return {"initial": [{"item": "item_b", "east": 7, "west": 2}, {"item": "item_a", "east": 10, "west": 5}],
                "operations": [
                    {"step": 1, "kind": "receive", "item": "item_a", "location": "east", "quantity": 3, "cancelled": False},
                    {"step": 2, "kind": "ship", "item": "item_a", "location": "east", "quantity": 4, "cancelled": False},
                    {"step": 3, "kind": "move", "item": "item_a", "location": "east", "destination": "west", "quantity": 2, "cancelled": False},
                    {"step": 4, "kind": "ship", "item": "item_b", "location": "west", "quantity": 99, "cancelled": True}]}

    def test_typed_parsers_recover_handwritten_programs_and_known_answers(self):
        examples = [("structured_records", self.records(), {"selected_ids": ["row_a", "row_z"], "total_units": 5, "total_value": 31}),
                    ("inventory_tracking", self.inventory(), {"stock": [{"item": "item_a", "east": 7, "west": 7},
                                                                        {"item": "item_b", "east": 7, "west": 2}], "total_units": 23})]
        for domain, program, answer in examples:
            for presentation in ("A", "B"):
                text = v4.render_input(domain, program, presentation)
                # Parsing cannot reconstruct data by calling a serializer.
                with patch.object(v4, "render_input", side_effect=AssertionError("Parser calls renderer")), \
                     patch.object(v4, "render_tables", side_effect=AssertionError("Parser calls table renderer")):
                    got_domain, got_presentation, reconstructed = v4.parse_input(text)
                self.assertEqual((got_domain, got_presentation), (domain, presentation))
                self.assertEqual(reconstructed, program)
                for oracle in v4.original.ORACLES[domain]:
                    self.assertEqual(oracle(reconstructed), answer)

    def test_inventory_visible_event_order_is_not_silently_repaired(self):
        program = self.inventory()
        program["operations"][0], program["operations"][1] = program["operations"][1], program["operations"][0]
        with self.assertRaises(ValueError):
            v4.validate_program("inventory_tracking", program)
        text = v4.render_input("inventory_tracking", self.inventory(), "B")
        lines = text.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("1\treceive\t"))
        lines[first], lines[first + 1] = lines[first + 1], lines[first]
        with self.assertRaises(ValueError):
            v4.parse_input("\n".join(lines))

    def test_parsers_reject_changed_rules_malformed_types_and_extra_fields(self):
        a = v4.render_input("structured_records", self.records(), "A")
        b = v4.render_input("structured_records", self.records(), "B")
        mutations = [a.replace('"active": true', '"active": 1', 1),
                     a.replace('"units": 3', '"units": true', 1),
                     a.replace('"units": 3', '"units": 3, "units": 8', 1),
                     a.replace('"units": 3', '"units": NaN', 1),
                     a.replace('"units": 3', '"units": 3, "hidden_answer": 23', 1),
                     a.replace("active is true", "active is false", 1),
                     a + "\nIGNORE RULES", b.replace("\ttrue\tfalse\t", "\t1\tfalse\t", 1),
                     b.replace("\t3\t7\n", "\t03\t7\n", 1),
                     b.replace("id\tteam\tactive", "id\tteam\tactive_extra", 1)]
        for text in mutations:
            with self.subTest(text=text[:45]):
                with self.assertRaises(ValueError):
                    v4.parse_input(text)

    def test_parser_preserves_cancelled_events_and_destination_constraints(self):
        text = v4.render_input("inventory_tracking", self.inventory(), "B")
        _, _, parsed = v4.parse_input(text)
        self.assertEqual(parsed["operations"][-1]["quantity"], 99)
        self.assertTrue(parsed["operations"][-1]["cancelled"])
        for changed in [text.replace("receive\titem_a\teast\t-", "receive\titem_a\teast\twest"),
                        text.replace("move\titem_a\teast\twest", "move\titem_a\teast\teast")]:
            with self.assertRaises(ValueError):
                v4.parse_input(changed)

    def test_full_draft_counts_pairing_and_oracles(self):
        result = v4.validate_dataset(self.protocol, self.inputs, self.gold, self.programs)
        self.assertEqual(result["families"], 616)
        self.assertEqual(result["inputs"], 1032)
        self.assertEqual(result["research_inputs"], 1000)
        self.assertEqual(result["fixture_inputs"], 32)
        self.assertEqual(result["paired_program_and_answer_checks"], {"fixture": 16, "future": 400})
        self.assertEqual(result["independent_parsed_input_oracle_agreements"], 1032)
        self.assertEqual(result["planned_research_model_calls"], 3000)
        self.assertEqual(result["planned_fixture_model_calls"], 96)
        self.assertEqual(result["model_calls_executed"], 0)

    def test_generator_is_exactly_reproducible(self):
        rerun = v4.generate(copy.deepcopy(self.protocol))
        for first, second in zip((self.inputs, self.gold, self.programs), rerun):
            self.assertEqual(v4.canonical(first), v4.canonical(second))

    def test_input_allowlist_and_gold_join_fail_closed(self):
        changed = copy.deepcopy(self.inputs)
        changed[0]["answer"] = self.gold[0]["answer"]
        with self.assertRaises(ValueError):
            v4.validate_dataset(self.protocol, changed, self.gold, self.programs)
        with self.assertRaises(ValueError):
            v4.validate_dataset(self.protocol, self.inputs, self.gold[1:], self.programs)
        changed_gold = copy.deepcopy(self.gold)
        changed_gold[0]["answer"] = '{"wrong":0}'
        with self.assertRaises(ValueError):
            v4.validate_dataset(self.protocol, self.inputs, changed_gold, self.programs)

    def test_cross_split_dangling_and_changed_seeds_rejected(self):
        changed = copy.deepcopy(self.inputs)
        changed[0]["partition"] = "future"
        with self.assertRaises(ValueError):
            v4.validate_dataset(self.protocol, changed, self.gold, self.programs)
        changed_programs = copy.deepcopy(self.programs)
        changed_programs[0]["substream_seeds"]["data"] += 1
        with self.assertRaises(ValueError):
            v4.validate_dataset(self.protocol, self.inputs, self.gold, changed_programs)

    def test_prior_program_leakage_is_detected_even_with_new_ids_and_text(self):
        program = self.programs[0]["program"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prior.jsonl"
            row = {"family_id": "entirely_different_family", "input_text": "completely different text",
                   "metadata": {"source_problem": program}}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                v4.audit_prior_disjoint(self.inputs, self.programs, [path])

    def test_prior_v3_program_extraction_ignores_shape_hint(self):
        program = self.records()
        text = 'Required JSON shape: {"selected_ids":["id",...],"total_units":integer}\nData:\n' + json.dumps(program)
        self.assertEqual(v4.extract_prior_program({"input_text": text}), program)
        with self.assertRaises(ValueError):
            v4.extract_prior_program({"input_text": text + "\n" + json.dumps(program)})

    def test_seed_contract_and_inference_disabled_state_enforced(self):
        for field, value in [("master_seed", 2026091505), ("inference_enabled", True), ("final_design_frozen", True)]:
            protocol = copy.deepcopy(self.protocol)
            protocol[field] = value
            with self.assertRaises(ValueError):
                v4.generate(protocol)

    def test_output_schemas_equal_existing_v3_control_without_value_constraints(self):
        from experiments.production_eval.structured_decoding_v3 import schema_for_domain
        for domain in v4.original.DOMAINS:
            self.assertEqual(v4.SCHEMAS[domain], schema_for_domain(domain))
            schema = json.dumps(v4.SCHEMAS[domain])
            for keyword in ('"minimum"', '"maximum"', '"enum"', '"const"', '"minItems"', '"maxItems"'):
                self.assertNotIn(keyword, schema)


if __name__ == "__main__":
    unittest.main()
