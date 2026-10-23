"""Analysis invariants: missing outcomes, family units, frozen weights and grading."""
import copy
import json
import unittest

from experiments.production_eval.analyze_outcomes_v2 import analyze, grade_response, sha256, verify_frozen_source_hashes


def benchmark(i, *, role="construction", dataset="gsm8k", answer="1"):
    text = f"Task {dataset} {role} {i}"
    return {"example_id": f"{dataset}/{role}/{i}", "input_text": text, "input_hash": sha256(text),
            "family_id": f"family/{dataset}/{role}/{i}", "dataset": dataset, "split": "test",
            "task_family": "arithmetic", "data_role": role, "answer": answer, "answer_type": "decimal"}


def synthetic_family(i=0, *, role="construction"):
    rows = []
    for variant in ("base", "paraphrase", "counterfactual", "counterfactual_paraphrase"):
        text = f"Task {i} {variant} {role}"
        rows.append({"example_id": f"synthetic/{role}/{i}/{variant}", "input_text": text, "input_hash": sha256(text),
                     "family_id": f"synthetic_family/{role}/{i}", "dataset": "controlled_tasks_v1", "split": role,
                     "task_family": "structured_records", "data_role": "development_construction" if role == "construction" else "controlled_reference",
                     "answer": json.dumps({"count": 2 if variant.startswith("counterfactual") else 1}), "answer_type": "json",
                     "metadata": {"variant": variant, "shift_regime": "stationary"}})
    return rows


def response(row, config="a", answer=None, *, status="success", format_status="valid"):
    if answer is None and status == "success" and format_status == "valid":
        answer = json.loads(row["answer"]) if row["answer_type"] == "json" else row["answer"]
    return {"config_id": config, "example_id": row["example_id"], "input_hash": row["input_hash"],
            "status": status, "format_status": format_status, "answer": answer,
            "started_at_utc": "2026-09-14T11:00:00+00:00", "completed_at_utc": "2026-09-14T11:00:01+00:00"}


def frozen(rows, *, selected=None, weights=None, probabilities=None):
    pool = [row["family_id"] for row in rows if row["data_role"] == "construction"]
    selected = pool if selected is None else selected
    return {"created_at": "2026-09-14T10:00:00+00:00", "runs": [{"dataset": "gsm8k", "policy": "uniform_random",
            "budget": len(selected), "seed": 100, "selected_family_ids": selected,
            "sample_weights": weights if weights is not None else [1 / len(selected)] * len(selected),
            "population_inclusion_probabilities": probabilities,
            "construction_family_ids": pool, "embedding_variant": "input_only", "estimator": "Horvitz-Thompson"}]}


class OutcomeAnalysisTests(unittest.TestCase):
    def test_targeted_discovery_yield_differs_from_unbiased_population_estimate(self):
        # Two known high-risk failures and four low-risk successes. Review both
        # failures plus one uniformly drawn success: raw yield=2/3, truth=1/3.
        # HT weights are derived analytically, without calling a selector.
        rows = [benchmark(i) for i in range(6)]
        outputs = [response(row, answer="2" if i < 2 else "1") for i, row in enumerate(rows)]
        manifest = frozen(rows, selected=[row["family_id"] for row in rows[:3]],
                          weights=[1 / 6, 1 / 6, 2 / 3], probabilities=[1, 1, .25, .25, .25, .25])
        manifest["runs"][0]["policy"] = "random_plus_targeted"
        result = analyze(rows, outputs, ("a",), manifest)
        discovery = result["selector_estimation"]["failure_discovery"][0]
        self.assertAlmostEqual(discovery["full_selection_failing_family_yield_per_100"], 200 / 3)
        self.assertAlmostEqual(discovery["estimated_construction_family_failure_rate"], 1 / 3)
        self.assertAlmostEqual(discovery["full_construction_family_failure_rate"], 1 / 3)
        self.assertEqual(discovery["family_failure_estimation_absolute_error"], 0)

    def test_failure_yield_unknowns_do_not_become_failures(self):
        rows = [benchmark(i) for i in range(2)]
        outputs = [response(rows[0], answer="2")]
        discovery = analyze(rows, outputs, ("a",), frozen(rows))["selector_estimation"]["failure_discovery"][0]
        self.assertEqual(discovery["failing_complete_selected_families"], 1)
        self.assertEqual(discovery["complete_selected_families"], 1)
        self.assertIsNone(discovery["full_selection_failing_family_yield_per_100"])
        self.assertIsNone(discovery["estimated_construction_family_failure_rate"])
        self.assertIsNone(discovery["full_construction_family_failure_rate"])

    def test_missingness_bounds_cover_every_possible_unknown_completion(self):
        rows = [benchmark(i) for i in range(2)]
        outputs = [response(rows[0], "a"), response(rows[1], "a"), response(rows[0], "b")]
        result = analyze(rows, outputs, ("a", "b"))
        candidate = next(row for row in result["group_summary"] if row["config_id"] == "b")
        self.assertEqual(candidate["full_family_accuracy_bounds_from_unknown_outcomes"], [.5, 1])
        self.assertEqual(result["paired_comparisons"][0]["full_family_delta_bounds_from_unknown_outcomes"], [-.5, 0])
        for final in ("1", "2"):
            completed = analyze(rows, outputs + [response(rows[1], "b", final)], ("a", "b"))
            delta = completed["paired_comparisons"][0]["full_population_family_mean_delta"]
            self.assertLessEqual(-.5, delta)
            self.assertLessEqual(delta, 0)

    def test_frozen_file_hashes_bind_answers_and_roles_not_only_prompt_ids(self):
        manifest = {"sources_hash": {"old/path/a": "hash1", "old/path/b": "hash2"}}
        self.assertEqual(verify_frozen_source_hashes(manifest, {"new/path/a": "hash1", "new/path/b": "hash2"})["status"], "verified")
        with self.assertRaises(ValueError):
            verify_frozen_source_hashes(manifest, {"a": "hash1", "b": "changed_answer_or_role"})
        with self.assertRaises(ValueError):
            verify_frozen_source_hashes(manifest, {"a": "hash1"})

    def test_infrastructure_unknown_is_distinct_from_completed_format_error(self):
        rows = [benchmark(i) for i in range(4)]
        outputs = [response(rows[0]), response(rows[1], format_status="invalid_json"),
                   response(rows[2], status="api_error", format_status="not_received")]
        result = analyze(rows, outputs, ("a",))
        summary = result["group_summary"][0]
        self.assertEqual(summary["observed_examples"], 2)
        self.assertEqual(summary["unknown_examples"], 2)
        self.assertEqual(summary["format_failures"], 1)
        self.assertEqual(summary["accuracy_among_observed"], .5)
        self.assertIsNone(summary["full_population_family_mean_accuracy"])

    def test_ground_truth_is_independent_of_reported_score(self):
        row = benchmark(1)
        output = {**response(row, answer="2"), "correct": True, "score": 1}
        self.assertFalse(grade_response(row, output)["correct"])

    def test_synthetic_json_checks_exact_types_and_ignores_key_order(self):
        row = synthetic_family()[0]
        self.assertTrue(grade_response(row, response(row, answer={"count": 1}))["correct"])
        self.assertFalse(grade_response(row, response(row, answer={"count": True}))["correct"])
        self.assertFalse(grade_response(row, response(row, answer={"count": 1.0}))["correct"])
        self.assertFalse(grade_response(row, response(row, answer={"count": 1, "extra": 0}))["correct"])
        self.assertFalse(grade_response(row, response(row, answer='{"count":1}'))["correct"])

    def test_hash_mismatch_and_duplicate_response_abort(self):
        row = benchmark(1)
        with self.assertRaises(ValueError):
            analyze([row], [{**response(row), "input_hash": "bad"}], ("a",))
        with self.assertRaises(ValueError):
            analyze([row], [response(row), response(row)], ("a",))

    def test_partial_synthetic_family_never_becomes_complete_family_mean(self):
        rows = synthetic_family(0) + synthetic_family(1)
        outputs = [response(row) for row in rows[:4]] + [response(rows[4], answer={"count": 9})]
        result = analyze(rows, outputs, ("a",))
        summary = result["group_summary"][0]
        self.assertEqual(summary["observed_examples"], 5)
        self.assertEqual(summary["accuracy_among_observed"], .8)
        self.assertEqual(summary["complete_families"], 1)
        self.assertEqual(summary["family_mean_accuracy_among_complete"], 1)
        self.assertIsNone(summary["full_population_family_mean_accuracy"])
        self.assertEqual(result["synthetic_consistency"][0]["incomplete_families"], 1)

    def test_paired_delta_direction_and_correctness_disagreement(self):
        rows = [benchmark(i) for i in range(3)]
        outputs = [response(rows[0], "a"), response(rows[1], "a", "2"), response(rows[2], "a", "2"),
                   response(rows[0], "b", "2"), response(rows[1], "b"), response(rows[2], "b")]
        paired = analyze(rows, outputs, ("a", "b"))["paired_comparisons"][0]
        self.assertAlmostEqual(paired["family_mean_delta"], 1 / 3)
        self.assertEqual(paired["correctness_disagreement"], 1)
        self.assertEqual(paired["candidate_wins"], 2)
        self.assertEqual(paired["anchor_wins"], 1)

    def test_invalid_pair_does_not_count_as_paraphrase_consistency(self):
        rows = synthetic_family()
        outputs = [response(row, format_status="invalid_json") for row in rows]
        metrics = analyze(rows, outputs, ("a",))["synthetic_consistency"][0]
        self.assertEqual(metrics["complete_families"], 1)
        self.assertEqual(metrics["paraphrase_consistency"], 0)
        self.assertEqual(metrics["counterfactual_prediction_changed"], 0)
        self.assertEqual(metrics["valid_paraphrase_pairs"], 0)

    def test_consistent_wrong_answer_is_not_counterfactual_success(self):
        rows = synthetic_family()
        outputs = [response(row, answer={"count": 9}) for row in rows]
        metrics = analyze(rows, outputs, ("a",))["synthetic_consistency"][0]
        self.assertEqual(metrics["paraphrase_consistency"], 1)
        self.assertEqual(metrics["paraphrase_both_correct"], 0)
        self.assertEqual(metrics["counterfactual_prediction_changed"], 0)
        self.assertEqual(metrics["counterfactual_both_correct"], 0)

    def test_changed_wrong_predictions_do_not_imply_correct_counterfactuals(self):
        rows = synthetic_family()
        outputs = [response(row, answer={"count": 9 if row["metadata"]["variant"].startswith("counterfactual") else 8}) for row in rows]
        metrics = analyze(rows, outputs, ("a",))["synthetic_consistency"][0]
        self.assertEqual(metrics["counterfactual_prediction_changed"], 1)
        self.assertEqual(metrics["counterfactual_both_correct"], 0)

    def test_family_crossing_roles_and_missing_variant_abort(self):
        rows = synthetic_family()
        altered = copy.deepcopy(rows)
        altered[0]["data_role"], altered[0]["split"] = "controlled_reference", "reference"
        with self.assertRaises(ValueError):
            analyze(altered, [], ("a",))
        with self.assertRaises(ValueError):
            analyze(rows[:-1], [], ("a",))

    def test_frozen_weighted_estimate_and_independent_reference_error(self):
        rows = [benchmark(i) for i in range(4)] + [benchmark(0, role="reference")]
        outputs = [response(row, "a", "2") for row in rows]
        outputs += [response(row, "b", "1" if i < 3 else "2") for i, row in enumerate(rows)]
        manifest = frozen(rows, selected=[rows[0]["family_id"], rows[3]["family_id"]], weights=[.75, .25], probabilities=[1 / 3, 1 / 3, 1 / 3, 1])
        result = analyze(rows, outputs, ("a", "b"), manifest)
        selection = result["selector_estimation"]["paired_run_results"][0]
        self.assertEqual(selection["selected_estimated_delta"], .75)
        self.assertEqual(selection["full_construction_delta"], .75)
        self.assertEqual(selection["construction_absolute_error"], 0)
        self.assertEqual(selection["references"][0]["full_reference_delta"], 0)
        self.assertEqual(selection["references"][0]["reference_absolute_error"], .75)
        self.assertIsNone(selection["confidence_interval"])
        self.assertFalse(result["selector_estimation"]["selection_performed_during_analysis"])

    def test_missing_selected_or_reference_outcomes_remain_unknown(self):
        rows = [benchmark(i) for i in range(2)] + [benchmark(0, role="reference")]
        outputs = [response(row, "a") for row in rows] + [response(rows[0], "b")]
        result = analyze(rows, outputs, ("a", "b"), frozen(rows))
        selection = result["selector_estimation"]["paired_run_results"][0]
        self.assertIsNone(selection["selected_estimated_delta"])
        self.assertIsNone(selection["full_construction_delta"])
        self.assertIsNone(selection["references"][0]["full_reference_delta"])

    def test_invalid_frozen_weights_pool_or_inclusion_probabilities_abort(self):
        rows = [benchmark(i) for i in range(2)]
        for mutation in ("weights", "pool", "probabilities"):
            manifest = frozen(rows)
            if mutation == "weights": manifest["runs"][0]["sample_weights"] = [.2, .2]
            if mutation == "pool": manifest["runs"][0]["construction_family_ids"] = [rows[0]["family_id"]]
            if mutation == "probabilities": manifest["runs"][0]["population_inclusion_probabilities"] = [.5, 1.0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze(rows, [], ("a", "b"), manifest)

    def test_primary_outcomes_before_freeze_abort_development_is_excluded(self):
        rows = [benchmark(0), benchmark(1, role="development")]
        outputs = [response(row) for row in rows]
        outputs[1]["started_at_utc"] = "2026-09-14T09:00:00+00:00"
        result = analyze(rows, outputs, ("a",), frozen(rows))
        self.assertEqual(result["freeze_timing_audit"]["primary_completed_responses_checked"], 1)
        outputs[0]["started_at_utc"] = "2026-09-14T09:00:00+00:00"
        with self.assertRaises(ValueError):
            analyze(rows, outputs, ("a",), frozen(rows))

    def test_balanced_reference_uses_domain_means_not_pooled_examples(self):
        domains = ("a", "b", "c", "d")
        rows = [benchmark(i, dataset=domain) for domain in domains for i in range(25)]
        rows += [benchmark(i, dataset=domain, role="reference") for index, domain in enumerate(domains) for i in range(index + 1)]
        outputs = [response(row, "a", "2") for row in rows]
        outputs += [response(row, "b", "1" if row["dataset"] in {"a", "b"} else "2") for row in rows]
        pool = [row["family_id"] for row in rows if row["data_role"] == "construction"]
        manifest = {"created_at": "2026-09-14T10:00:00+00:00", "runs": [{"dataset": "balanced_mixture", "policy": "uniform_random", "budget": 100,
            "seed": 100, "selected_family_ids": pool, "sample_weights": [.01] * 100, "construction_family_ids": pool,
            "embedding_variant": "input_only", "estimator": "Horvitz-Thompson", "population_inclusion_probabilities": [1] * 100,
            "family_domain": {row["family_id"]: row["dataset"] for row in rows if row["data_role"] == "construction"},
            "reference_domain_weights": {domain: .25 for domain in domains}}]}
        selection = analyze(rows, outputs, ("a", "b"), manifest)["selector_estimation"]["paired_run_results"][0]
        self.assertAlmostEqual(selection["full_construction_delta"], .5)
        self.assertAlmostEqual(selection["references"][0]["full_reference_delta"], .5)
        self.assertNotAlmostEqual(selection["references"][0]["full_reference_delta"], .3)


if __name__ == "__main__":
    unittest.main()
