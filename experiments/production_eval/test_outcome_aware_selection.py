"""Independent finite-population mathematics and selection-boundary tests."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.production_eval.outcome_aware_selection import (
    SIGNAL_CONFIGS, UnknownConstructionOutcomes, make_manifest, risk_flags,
    targeted_naive, main, sha,
)


def construction(domain="toy", prefix="f", count=6):
    return [{"example_id": f"{prefix}{i}_row", "family_id": f"{prefix}{i}",
             "data_role": "construction", "dataset": domain, "answer_type": "boolean"}
            for i in range(count)]


def grades_for(rows, error_families=()):
    return {(config, row["example_id"]): {
        "observed": True, "correct": not(config == "qwen_direct" and row["family_id"] in error_families),
        "format_valid": True,
        "canonical_prediction": "False" if config == "qwen_direct" and row["family_id"] in error_families else "True"}
        for row in rows for config in SIGNAL_CONFIGS}


def amendment():
    return {"construction_signal_configs": list(SIGNAL_CONFIGS), "construction_selection_only": True,
            "independent_reference_not_used_for_selection": True, "frozen_at_utc": "2026-09-14T18:00:00+00:00",
            "seeds": [100], "policies": {f"{signal}_{kind}": {} for signal in ("error", "disagreement")
                                         for kind in ("targeted_naive", "hybrid_ht")}}


def base_for(rows, budget=3):
    domains = sorted({row["dataset"] for row in rows})
    return {"runs": [{"dataset": domain, "budget": budget,
                      "construction_family_ids": sorted({r["family_id"] for r in rows if r["dataset"] == domain})}
                     for domain in domains]}


class OutcomeAwareTest(unittest.TestCase):
    def test_ht_unbiased_by_exact_enumeration_all_64_binary_populations(self):
        rows = construction(); grades = grades_for(rows, {"f0", "f1"})
        run = next(r for r in make_manifest(base_for(rows), rows, grades, amendment())["runs"]
                   if r["policy"] == "error_hybrid_ht")
        # Independently enumerate this fixed design: both high-risk families and
        # exactly one of the other four, each equally likely. There is no Monte
        # Carlo tolerance or reuse of the selector's weight implementation.
        possible_samples = [(0, 1, low) for low in range(2, 6)]
        inclusion = np.array([sum(i in sample for sample in possible_samples) / 4 for i in range(6)])
        np.testing.assert_array_equal(inclusion, [1, 1, .25, .25, .25, .25])
        np.testing.assert_allclose(run["population_inclusion_probabilities"], inclusion)
        selected = [run["construction_family_ids"].index(f) for f in run["selected_family_ids"]]
        np.testing.assert_allclose(run["sample_weights"], [1 / (6 * inclusion[i]) for i in selected])
        for values in itertools.product((0, 1), repeat=6):
            estimates = [sum(values[i] / inclusion[i] for i in sample) / 6 for sample in possible_samples]
            self.assertAlmostEqual(sum(estimates) / len(estimates), sum(values) / 6)
        self.assertEqual(run["selection_metadata"]["targeted_fraction"], .5)
        self.assertEqual(run["selection_metadata"]["risk_threshold"], .5)
        self.assertEqual(run["risk_family_count"], 2)

    def test_naive_error_targeting_has_demonstrable_upward_failure_bias(self):
        ids = [f"f{i}" for i in range(6)]; risk = np.array([True, True, False, False, False, False])
        # Population failure rate 2/6; enriched selection necessarily 2/3.
        estimates = []
        for seed in range(40):
            selected = targeted_naive(ids, risk, budget=3, seed=seed)
            estimates.append(float(risk[selected].mean()))
        self.assertAlmostEqual(float(np.mean(estimates)), 2 / 3)
        self.assertAlmostEqual(float(np.mean(estimates)) - float(risk.mean()), 1 / 3)
        selected = targeted_naive(ids, risk, budget=2, seed=100)
        self.assertEqual(set(selected), {0, 1})  # Other families have zero inclusion.

    def test_hybrid_uniform_when_risk_constant_and_census_exact(self):
        rows = construction()
        for error_families in (set(), {f"f{i}" for i in range(6)}):
            grades = grades_for(rows, error_families)
            for budget in (3, 6):
                manifest = make_manifest(base_for(rows, budget), rows, grades, amendment())
                hybrid = next(r for r in manifest["runs"] if r["policy"] == "error_hybrid_ht")
                np.testing.assert_allclose(hybrid["population_inclusion_probabilities"], budget / 6)
                np.testing.assert_allclose(hybrid["sample_weights"], 1 / budget)
                self.assertAlmostEqual(sum(hybrid["sample_weights"]), 1)

    def test_one_failed_descendant_marks_family_and_two_invalids_do_not_disagree(self):
        rows = construction(count=2)
        sibling = {**rows[0], "example_id": "f0_other_descendant"}; rows.append(sibling)
        grades = grades_for(rows)
        grades[("qwen_direct", sibling["example_id"])].update(correct=False, format_valid=False, canonical_prediction="invalid-A")
        grades[("qwen_work", sibling["example_id"])].update(correct=False, format_valid=False, canonical_prediction="invalid-B")
        flags = risk_flags(rows, grades)
        self.assertTrue(flags["error"]["f0"])
        self.assertFalse(flags["disagreement"]["f0"])
        grades[("qwen_work", sibling["example_id"])].update(correct=True, format_valid=True, canonical_prediction="True")
        self.assertTrue(risk_flags(rows, grades)["disagreement"]["f0"])

    def test_reference_or_third_config_grade_cannot_cross_selection_boundary(self):
        rows = construction(); grades = grades_for(rows)
        for extra_key in (("qwen_direct", "reference_id"), ("gemini_direct", rows[0]["example_id"])):
            extra = {**grades, extra_key: {"secret_reference_outcome": True}}
            with self.assertRaises(ValueError):
                risk_flags(rows, extra)
            with self.assertRaises(ValueError):
                make_manifest(base_for(rows), rows, extra, amendment())
        reference = {**rows[0], "data_role": "reference"}
        with self.assertRaises(ValueError):
            risk_flags([reference], grades_for([reference]))

    def test_unknown_family_aborts_affected_domain_without_imputation(self):
        rows = construction("A", "a", 3) + construction("B", "b", 3)
        grades = grades_for(rows)
        grades[("qwen_work", "a0_row")] = {"observed": False, "correct": None, "format_valid": None, "canonical_prediction": None}
        with self.assertRaises(UnknownConstructionOutcomes):
            risk_flags(rows, grades)
        manifest = make_manifest(base_for(rows, budget=2), rows, grades, amendment())
        self.assertEqual({run["dataset"] for run in manifest["runs"]}, {"B"})
        self.assertEqual(manifest["skipped_pools"], [{"dataset": "A", "budget": 2,
                          "reason": "unknown_construction_outcomes", "unknown_family_ids": ["a0"]}])

    def test_mixture_metadata_preserved_and_target_alignment_checked(self):
        rows = construction("A", "a", 2) + construction("B", "b", 2)
        ids = sorted(row["family_id"] for row in rows)
        domains = {row["family_id"]: row["dataset"] for row in rows}
        base = {"runs": [{"dataset": "balanced_mixture", "budget": 2,
                          "construction_family_ids": ids, "family_domain": domains,
                          "reference_domain_weights": {"A": .5, "B": .5}}]}
        manifest = make_manifest(base, rows, grades_for(rows, {"a0"}), amendment())
        for run in manifest["runs"]:
            self.assertEqual(run["family_domain"], domains)
            self.assertEqual(run["reference_domain_weights"], {"A": .5, "B": .5})
        bad = copy.deepcopy(base); bad["runs"][0]["family_domain"]["a0"] = "B"
        with self.assertRaises(ValueError):
            make_manifest(bad, rows, grades_for(rows), amendment())
        bad = copy.deepcopy(base); bad["runs"][0]["reference_domain_weights"] = {"A": .75, "B": .25}
        with self.assertRaises(ValueError):
            make_manifest(bad, rows, grades_for(rows), amendment())

    def test_unknown_or_wrong_domain_family_rejected(self):
        rows = construction(); base = base_for(rows)
        base["runs"][0]["construction_family_ids"][0] = "reference-family"
        base["runs"][0]["construction_family_ids"].sort()
        with self.assertRaises(ValueError):
            make_manifest(base, rows, grades_for(rows), amendment())
        base = base_for(rows); base["runs"][0]["dataset"] = "different-domain"
        with self.assertRaises(ValueError):
            make_manifest(base, rows, grades_for(rows), amendment())

    def test_invalid_risk_arrays_and_budgets_rejected(self):
        ids = ["a", "b"]
        for risk, budget in (([True], 1), ([float("nan"), 1.0], 1), ([[True], [False]], 1), ([True, False], 0)):
            with self.assertRaises(ValueError):
                targeted_naive(ids, risk, budget, 100)
        with self.assertRaises(ValueError):
            targeted_naive(ids[::-1], [True, False], 1, 100)

    def test_full_record_hash_checked_before_responses_or_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.jsonl"; source.write_text('{"answer":"original"}\n')
            base = directory / "base.json"; base.write_text(json.dumps({"sources_hash": {"original_path": sha(source)}, "runs": []}))
            frozen = directory / "amendment.json"; frozen.write_text(json.dumps({"base_selection_manifest_sha256": sha(base)}))
            # Changing only the gold answer must fail before even opening outputs.
            source.write_text('{"answer":"tampered"}\n')
            args = ["selector", "--records", str(source), "--responses", str(directory / "missing-response.jsonl"),
                    "--frozen-selections", str(base), "--amendment", str(frozen), "--output", str(directory / "out")]
            with patch("sys.argv", args), patch("experiments.production_eval.outcome_aware_selection.make_manifest") as select_mock:
                with self.assertRaisesRegex(ValueError, "source contents differ"):
                    main()
                select_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
