import copy
import unittest

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.budget_sensitivity import (
    BUDGETS, choose_pool, paired_comparison, usage_summary, variant_config, verify_same_task_protocol,
)


def baseline_config():
    return {**inference.DEFAULTS, "provider": "local", "profile": "reason_then_answer",
            "config_id": "qwen_work", "model": "pinned-model", "model_revision": "pinned-revision",
            "runner_version": inference.VERSION, "local_transport_revision": inference.LOCAL_TRANSPORT_REVISION}


def record(identifier="a", family="f", text="True and True"):
    return {"example_id": identifier, "family_id": family, "dataset": "toy", "task_family": "boolean_reasoning",
            "answer_type": "boolean", "answer": "True", "input_text": text,
            "input_hash": inference.digest(text), "data_role": "construction"}


def response(row, answer="True", status="success", format_status="valid"):
    return {"example_id": row["example_id"], "input_hash": row["input_hash"], "status": status,
            "format_status": format_status, "answer": answer}


def pool_fixture():
    rows, domains = [], {}
    for domain_index in range(4):
        for family_index in range(25):
            family = f"d{domain_index}f{family_index:02d}"; name = f"domain{domain_index}"
            domains[family] = name
            variants = ("base", "paraphrase", "counterfactual", "counterfactual_paraphrase") if domain_index >= 2 else (None,)
            for variant in variants:
                row = record(family + str(variant), family)
                row["dataset"] = name
                if variant:
                    row.update(answer_type="json", task_family=name, metadata={"variant": variant})
                rows.append(row)
    frozen = {"runs": [{"dataset": "balanced_mixture", "construction_family_ids": sorted(domains),
                        "family_domain": domains} for _ in range(2)]}
    return frozen, rows


class BudgetSensitivityTest(unittest.TestCase):
    def test_variants_change_only_output_budget_and_bookkeeping(self):
        baseline = baseline_config()
        original = copy.deepcopy(baseline)
        for budget in BUDGETS:
            config = variant_config(baseline, budget)
            before, after = verify_same_task_protocol(record(), baseline, config)
            self.assertEqual(before["request"]["system_prompt"], after["request"]["system_prompt"])
            self.assertEqual(before["request"]["schema"], after["request"]["schema"])
            self.assertEqual(config["max_context_tokens"], 8192)
            self.assertEqual(config["max_output_tokens"], budget)
            self.assertNotEqual(before["cache_key"], after["cache_key"])
        self.assertEqual(baseline, original)

    def test_baseline_hash_is_unchanged_after_variant_preparation(self):
        baseline = baseline_config(); row = record()
        original_hash = inference.prepare(row, baseline)["cache_key"]
        for budget in BUDGETS:
            verify_same_task_protocol(row, baseline, variant_config(baseline, budget))
        self.assertEqual(inference.prepare(row, baseline)["cache_key"], original_hash)

    def test_no_small_task_shortcut_or_gold_schema_leak(self):
        baseline = baseline_config(); variant = variant_config(baseline, 4096)
        short, long = record(text="True"), record(text="True and " * 1000 + "True")
        short_request = inference.prepare(short, variant)["request"]
        long_request = inference.prepare(long, variant)["request"]
        self.assertEqual(short_request["system_prompt"], long_request["system_prompt"])
        self.assertIn("work string first", short_request["system_prompt"])
        altered = {**short, "answer": "SECRET_GOLD_REPLACEMENT", "metadata": {"gold": "SECRET"}}
        self.assertEqual(inference.prepare(short, variant)["request"], inference.prepare(altered, variant)["request"])
        self.assertEqual(set(short_request["schema"]["required"]), {"work", "answer"})

    def test_unsupported_budget_profile_or_parameter_change_rejected(self):
        for budget in (1024, 8192):
            with self.assertRaises(ValueError):
                variant_config(baseline_config(), budget)
        baseline = baseline_config(); baseline["profile"] = "direct"
        with self.assertRaises(ValueError):
            variant_config(baseline, 2048)
        baseline = baseline_config(); changed = variant_config(baseline, 2048); changed["seed"] += 1
        with self.assertRaises(ValueError):
            verify_same_task_protocol(record(), baseline, changed)

    def test_exact_older_pool_all_250_descendants_preserved(self):
        frozen, rows = pool_fixture()
        selected, ids, domains = choose_pool(frozen, list(reversed(rows)))
        self.assertEqual(len(selected), 250)
        self.assertEqual(len(ids), 100)
        self.assertEqual({r["example_id"] for r in selected}, {r["example_id"] for r in rows})
        self.assertEqual(selected, sorted(rows, key=lambda row: row["example_id"]))
        self.assertEqual(domains, frozen["runs"][0]["family_domain"])

    def test_reference_rows_missing_descendant_or_changed_old_pool_rejected(self):
        frozen, rows = pool_fixture()
        bad = copy.deepcopy(rows); bad[0]["data_role"] = "reference"
        with self.assertRaises(ValueError):
            choose_pool(frozen, bad)
        with self.assertRaises(ValueError):
            choose_pool(frozen, rows[:-1])
        bad_frozen = copy.deepcopy(frozen); bad_frozen["runs"][1]["construction_family_ids"] = list(reversed(bad_frozen["runs"][1]["construction_family_ids"]))
        with self.assertRaises(ValueError):
            choose_pool(bad_frozen, rows)

    def test_conditional_pairing_excludes_whole_incomplete_family_without_imputation(self):
        rows = [record("a0", "a"), record("a1", "a"), record("b0", "b"), record("b1", "b")]
        baseline = {r["example_id"]: response(r, "False" if r["example_id"] == "a1" else "True") for r in rows}
        variant = {r["example_id"]: response(r) for r in rows[:-1]}
        summary = next(s for s in paired_comparison(rows, baseline, variant) if s["dataset"] == "toy")
        self.assertEqual(summary["expected_families"], 2)
        self.assertEqual(summary["paired_complete_families"], 1)
        self.assertEqual(summary["variant_unknown_rows"], 1)
        self.assertEqual(summary["paired_accuracy_delta"], .5)
        self.assertIsNone(summary["full_population_paired_delta"])
        variant["b1"] = response(rows[-1], None, format_status="server_parser_invalid")
        complete = next(s for s in paired_comparison(rows, baseline, variant) if s["dataset"] == "toy")
        self.assertEqual(complete["paired_complete_families"], 2)
        self.assertEqual(complete["variant_unknown_rows"], 0)
        self.assertEqual(complete["variant_format_failures"], 1)
        self.assertEqual(complete["full_population_paired_delta"], 0)

    def test_usage_missingness_remains_visible(self):
        attempts = [{"usage": {"prompt_tokens": 10, "completion_tokens": 20}, "latency_seconds": 1, "charged_nanodollars": 0},
                    {"usage": None, "latency_seconds": 2, "charged_nanodollars": 0},
                    {"usage": {"prompt_tokens": 7}, "latency_seconds": 3, "charged_nanodollars": 0}]
        result = usage_summary(attempts)
        self.assertEqual(result["reported_prompt_tokens"], 17)
        self.assertEqual(result["reported_completion_tokens"], 20)
        self.assertEqual(result["completion_token_usage_missing_attempts"], 2)
        self.assertEqual(result["prompt_token_usage_missing_attempts"], 1)
        self.assertEqual(result["sum_request_latency_seconds"], 6)
        self.assertEqual(result["accounted_api_usd"], 0)


if __name__ == "__main__":
    unittest.main()
