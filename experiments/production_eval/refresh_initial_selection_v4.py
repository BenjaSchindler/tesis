"""Revise only draft initial random sets, using one RNG across all strata.

Preserves the first feature/selection artifacts unchanged. No re-embedding,
inference, training, outcomes, future labels or final experiment freeze.
Run after refresh_features_v4.py completes, before root final protocol review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_refresh_v4 import BASE, INPUT_FIELDS, GROUPS, load_jsonl, sha
from experiments.production_eval.refresh_features_v4 import ACQUISITION_SEEDS, sorted_scope
from experiments.production_eval.selection_methods import select

DETERMINISTIC_POLICIES = ("facility_input_only", "centroid_input_only", "facility_with_construction_gold")


def revised_random_sets(construction):
    ids = [r["family_id"] for r in construction]
    if ids != sorted(set(ids)) or len(ids) != 200 or any(set(r) != INPUT_FIELDS or r["partition"] != "construction"
                                                       or r["presentation"] != "A" for r in construction):
        raise ValueError("Expected 200 input-only construction A families")
    labels = [r["group_id"] for r in construction]
    if Counter(labels) != {group: 50 for group, _, _, _ in GROUPS}:
        raise ValueError("Incorrect construction groups")
    runs = []
    for seed in ACQUISITION_SEEDS:
        # One sampler call creates one RNG and advances it across every group.
        result = select("stratified_random", ids, 80, strata=labels, seed=seed)
        picked = [ids[i] for i in result.indices]
        chosen_rows = [construction[i] for i in result.indices]
        if len(picked) != 80 or Counter(r["group_id"] for r in chosen_rows) != {g: 20 for g in set(labels)}:
            raise ValueError("Revised selection violates fixed allocation")
        cells = []
        for group, _, _, _ in GROUPS:
            cells.append({"group_id": group, "construction_families": [r["family_id"] for r in construction if r["group_id"] == group],
                          "selected_family_ids": [r["family_id"] for r in chosen_rows if r["group_id"] == group],
                          "population_count": 50, "sample_count": 20})
        runs.append({"selection_id": f"stratified_random_seed{seed}", "policy": "stratified_random", "seed": seed,
                     "selected_family_ids": picked, "selected_example_ids": [r["example_id"] for r in chosen_rows],
                     "construction_family_ids": ids, "budget_families": 80, "per_group": 20,
                     "sample_weights": [float(w) for w in result.sample_weights],
                     "estimator": "uniform_mean_within_group_then_scenario_group_weight",
                     "population_inclusion_probabilities": result.inclusion_probabilities.tolist(),
                     "selector_native_alternative_weights": [float(w) for w in result.sample_weights],
                     "alternative_weights_are_primary": False, "cells": cells,
                     "raw_joint_sampler_metadata": result.metadata,
                     "uses_gold_in_selection": False, "gold_family_answers_revealed_for_selection": 0,
                     "model_outcomes_used": 0, "future_inputs_used_in_selection": 0,
                     "information_scope": "200 construction inputs and observable groups",
                     "randomization": "Single stratified_random call and RNG per acquisition seed; RNG advances across groups"})
    return runs


def combine_with_deterministic(original_runs, revised):
    deterministic = [copy.deepcopy(r) for r in original_runs if r["policy"] in DETERMINISTIC_POLICIES]
    if len(deterministic) != 3 or {r["policy"] for r in deterministic} != set(DETERMINISTIC_POLICIES):
        raise ValueError("Original deterministic arms missing/duplicated")
    if len(revised) != 20 or [r["seed"] for r in revised] != list(ACQUISITION_SEEDS):
        raise ValueError("Revised random seed cohort changed")
    if any(r["sample_weights"] != [1/80] * 80 or r["model_outcomes_used"] != 0 for r in deterministic):
        raise ValueError("Deterministic principal estimator or information scope changed")
    return revised + deterministic


def prepare(base=BASE):
    base = Path(base)
    features = base / "draft_selection_features"
    destination = base / "draft_selection_revision"
    if destination.exists():
        raise FileExistsError("Selection revision exists; refusing overwrite")
    feature_manifest = json.loads((features / "manifest.json").read_text())
    original_path = features / "selected_families.json"
    if sha(original_path) != feature_manifest["selected_families_sha256"]:
        raise ValueError("Original draft selections changed")
    data_manifest = base / "draft_dataset/manifest.json"
    if sha(data_manifest) != feature_manifest["dataset_manifest_sha256"]:
        raise ValueError("Original reviewed dataset manifest changed")
    input_path = base / "draft_dataset/inputs_construction.jsonl"
    expected_hash = feature_manifest["dataset_files"]["inputs_construction.jsonl"]["sha256"]
    if sha(input_path) != expected_hash:
        raise ValueError("Construction inputs changed")
    construction = sorted_scope(load_jsonl(input_path), "construction", "A")
    original_runs = json.loads(original_path.read_text())["runs"]
    revised = revised_random_sets(construction)
    runs = combine_with_deterministic(original_runs, revised)
    changed = sum(a["selected_family_ids"] != b["selected_family_ids"] for a, b in zip(original_runs[:20], revised))
    destination.mkdir()
    path = destination / "selected_families.json"
    path.write_text(json.dumps({"status": "draft_initial_sets_rng_revision_pending_final_review", "runs": runs,
                               "selection_performed_before_model_outcomes": True, "new_model_outcomes_observed": 0}, indent=2) + "\n")
    manifest = {"status": "draft_selection_revision_complete_pending_final_review", "final_experiment_frozen": False,
                "inference_enabled": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "revision_reason": "Initial draft reset an identical RNG in each same-sized group, coupling selected ordinal positions. This revision uses one global stratified sampler call per seed, with unchanged 20-per-group quotas and all information constraints.",
                "initial_sets": len(runs), "random_sets": 20, "random_sets_changed": changed,
                "deterministic_sets_preserved_exactly": 3, "no_features_recomputed": True,
                "selected_families_file": str(path.resolve()), "selected_families_sha256": sha(path),
                "original_selected_families_sha256": sha(original_path), "original_feature_manifest_sha256": sha(features / "manifest.json"),
                "dataset_manifest_sha256": sha(data_manifest), "selection_protocol_sha256": sha(base / "draft_selection_protocol.json"),
                "source_files": [{"path": str(p.resolve()), "sha256": sha(p)} for p in
                                 (Path(__file__), Path(__file__).with_name("test_refresh_initial_selection_v4.py"),
                                  Path(__file__).with_name("selection_methods.py"))],
                "model_outcomes_used": 0, "future_labels_used": False,
                "promotion_note": "Use these 23 sets for any later final selection manifest. Preserve the previous draft feature manifest and selections as history."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "initial_sets": 23, "changed_random_sets": changed,
                      "destination": str(destination.resolve())}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    prepare(parser.parse_args().base)
