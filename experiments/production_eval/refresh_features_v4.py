"""CPU-only draft embeddings and 23 initial sets for gold-refresh v4.

Main selectors receive construction inputs only. The separate oracle-information
arm receives gold for exactly 200 construction inputs. No model outcomes, future
gold, model inference, or final experiment freeze is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_refresh_v4 import BASE, INPUT_FIELDS, GROUPS, canonical, load_jsonl, sha
from experiments.production_eval.synthetic_tasks import digest
from experiments.production_eval.representation_ablation import run_variant
from experiments.production_eval.selection_methods import select

ACQUISITION_SEEDS = tuple(range(202609152001, 202609152021))
GOLD_SEPARATOR = "\n\nCanonical expected JSON answer:\n"


def sorted_scope(rows, partition, presentation):
    result = sorted([r for r in rows if r["partition"] == partition and r["presentation"] == presentation],
                    key=lambda r: r["family_id"])
    if any(set(r) != INPUT_FIELDS for r in result) or len({r["family_id"] for r in result}) != len(result):
        raise ValueError("Feature inputs violate the input-only scope or duplicate families")
    return result


def scoped_gold(gold_path, construction_ids):
    """Extract answer values only for construction IDs; reject missing targets."""
    targets = {}
    for line in Path(gold_path).read_text().splitlines():
        row = json.loads(line)
        if row["example_id"] not in construction_ids:
            continue
        if row["example_id"] in targets:
            raise ValueError("Duplicate construction gold")
        targets[row["example_id"]] = row["answer"]
    if set(targets) != set(construction_ids):
        raise ValueError("Construction gold scope is incomplete")
    return targets


def gold_feature_rows(construction, targets):
    ids = {r["example_id"] for r in construction}
    if set(targets) != ids or any(r["partition"] != "construction" or r["presentation"] != "A" for r in construction):
        raise ValueError("Gold features accept exactly construction A")
    result = []
    for row in construction:
        text = row["input_text"] + GOLD_SEPARATOR + targets[row["example_id"]]
        result.append({**row, "input_text": text, "input_hash": digest(text)})
    return result


def freeze_initial_sets(construction, input_matrix, gold_matrix):
    """Selectors only consume this construction scope; future arrays forbidden."""
    if len(construction) != 200 or [r["family_id"] for r in construction] != sorted({r["family_id"] for r in construction}):
        raise ValueError("Expected 200 construction families in stable order")
    if any(r["partition"] != "construction" or r["presentation"] != "A" or set(r) != INPUT_FIELDS for r in construction):
        raise ValueError("Selection requires input-only construction A records")
    group_counts = Counter(r["group_id"] for r in construction)
    if group_counts != {group: 50 for group, _, _, _ in GROUPS}:
        raise ValueError("Construction groups are not the declared four groups of 50")
    for matrix in (input_matrix, gold_matrix):
        if matrix.ndim != 2 or matrix.shape[0] != 200 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ValueError("Wrong construction embedding array")
    if input_matrix.shape != gold_matrix.shape:
        raise ValueError("The gold ablation must use the same embedding dimension")
    specs = [("stratified_random", seed) for seed in ACQUISITION_SEEDS] + [
        ("facility_input_only", None), ("centroid_input_only", None), ("facility_with_construction_gold", None)]
    runs = []
    for policy, seed in specs:
        picked, alternate_weights, cells = [], {}, []
        uses_gold = policy == "facility_with_construction_gold"
        matrix = gold_matrix if uses_gold else input_matrix
        for group, _, _, _ in GROUPS:
            positions = [i for i, row in enumerate(construction) if row["group_id"] == group]
            families = [construction[i]["family_id"] for i in positions]
            method = "uniform_random" if policy == "stratified_random" else \
                     "centroid_proximity" if policy == "centroid_input_only" else "facility_location"
            result = select(method, families, 20, seed=seed or 0,
                            embeddings=None if method == "uniform_random" else matrix[positions])
            group_selected = [families[i] for i in result.indices]
            picked.extend(group_selected)
            for family, weight in zip(group_selected, result.sample_weights):
                alternate_weights[family] = float(weight) / 4
            cells.append({"group_id": group, "construction_families": families,
                          "selected_family_ids": group_selected, "population_count": 50, "sample_count": 20,
                          "raw_selector_metadata": result.metadata})
        picked.sort()
        lookup = {row["family_id"]: row for row in construction}
        if len(picked) != 80 or len(set(picked)) != 80 or Counter(lookup[i]["group_id"] for i in picked) != {g: 20 for g in group_counts}:
            raise ValueError("Initial budget, family or group matching failed")
        runs.append({"selection_id": policy if seed is None else f"{policy}_seed{seed}", "policy": policy, "seed": seed,
                     "selected_family_ids": picked, "selected_example_ids": [lookup[i]["example_id"] for i in picked],
                     "construction_family_ids": [r["family_id"] for r in construction],
                     "budget_families": 80, "per_group": 20,
                     "sample_weights": [1/80] * 80,
                     "estimator": "uniform_mean_within_group_then_scenario_group_weight",
                     "population_inclusion_probabilities": [.4] * 200 if seed is not None else None,
                     "selector_native_alternative_weights": [alternate_weights[i] for i in picked],
                     "alternative_weights_are_primary": False, "cells": cells,
                     "uses_gold_in_selection": uses_gold,
                     "gold_family_answers_revealed_for_selection": 200 if uses_gold else 0,
                     "model_outcomes_used": 0, "future_inputs_used_in_selection": 0,
                     "information_scope": "200 construction inputs plus their gold answers" if uses_gold else "200 construction inputs and observable groups"})
    return runs


def validate_selection_protocol(protocol):
    if protocol["inference_enabled"] is not False or protocol["final_experiment_frozen"] is not False:
        raise ValueError("Only draft selection preparation is allowed")
    if tuple(protocol["acquisition_seeds"]) != ACQUISITION_SEEDS or protocol["device"] != "cpu" or protocol["torch_threads"] != 4:
        raise ValueError("Selection seed/device contract changed")
    if protocol["encoder"] != "sentence-transformers/all-mpnet-base-v2" or protocol["encoder_revision"] != "e8c3b32edf5434bc2275fc9bab85f82640a19130":
        raise ValueError("Pinned encoder changed")
    if protocol["representation"] != "chunk_mean" or protocol["initial_per_group"] != 20 or protocol["planned_initial_sets"] != 23:
        raise ValueError("Fixed representation or allocation changed")


def prepare(base=BASE):
    base = Path(base)
    data = base / "draft_dataset"
    destination = base / "draft_selection_features"
    if destination.exists():
        raise FileExistsError("Existing draft features will not be overwritten")
    protocol_path = base / "draft_selection_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validate_selection_protocol(protocol)
    dataset_manifest = json.loads((data / "manifest.json").read_text())
    if dataset_manifest["inference_enabled"] is not False or dataset_manifest["final_design_frozen"] is not False:
        raise ValueError("Unexpected dataset status")
    for name, evidence in dataset_manifest["files"].items():
        if sha(data / name) != evidence["sha256"]:
            raise ValueError("Reviewed dataset file hash changed: " + name)
    rows = load_jsonl(data / "inputs.jsonl")
    construction = sorted_scope(rows, "construction", "A")
    scopes = {"construction_A": construction, "future_A": sorted_scope(rows, "future", "A"),
              "future_B": sorted_scope(rows, "future", "B")}
    if [len(scope) for scope in scopes.values()] != [200, 400, 400]:
        raise ValueError("Input-only feature scope changed")
    # First compute all three input-only scopes. Gold values are loaded later.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers.utils import logging
    logging.disable_progress_bar(); logging.set_verbosity_error()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + protocol["encoder"].replace("/", "--")) / "snapshots" / protocol["encoder_revision"]
    if not snapshot.is_dir():
        raise ValueError("Pinned encoder is not locally cached; no download permitted")
    model_sources = [{"relative_path": str(path.relative_to(snapshot)), "sha256": sha(path)}
                     for path in sorted(snapshot.rglob("*")) if path.is_file()]
    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True, trust_remote_code=False)
    model.eval()
    destination.mkdir()
    started = time.monotonic()
    feature_evidence, matrices = {}, {}

    def encode(scope_name, items, uses_gold):
        folder = destination / scope_name
        folder.mkdir()
        sequences = model.tokenizer([r["input_text"] for r in items], add_special_tokens=False, truncation=False)["input_ids"]
        matrix, evidence, token_records = run_variant(model, items, sequences, [], "chunk_mean", "cpu", folder)
        if evidence["requests_with_tokens_omitted"] != 0 or evidence["distinct_representation_token_sequences"] != len(items):
            raise ValueError("Input truncation or exact representation collisions detected")
        evidence.update(scope=scope_name, rows=len(items), family_ids=[r["family_id"] for r in items],
                        example_ids=[r["example_id"] for r in items], input_hashes=[r["input_hash"] for r in items],
                        uses_gold=uses_gold, model_outcomes_used=False, order="family_id ascending")
        feature_evidence[scope_name] = evidence
        print(json.dumps({"completed_feature_scope": scope_name, "rows": len(items), "elapsed_seconds": evidence["elapsed_seconds"],
                          "full_tokens": evidence["total_source_tokens"], "omitted_tokens": 0}), flush=True)
        return matrix

    for scope_name, items in scopes.items():
        matrices[scope_name] = encode(scope_name, items, False)
    targets = scoped_gold(data / "gold.jsonl", {r["example_id"] for r in construction})
    gold_rows = gold_feature_rows(construction, targets)
    gold_matrix = encode("construction_A_with_gold", gold_rows, True)
    selections = freeze_initial_sets(construction, matrices["construction_A"], gold_matrix)
    selected_path = destination / "selected_families.json"
    selected_path.write_text(json.dumps({"status": "draft_initial_sets_fixed_pending_final_review", "runs": selections,
                                       "new_model_outcomes_observed": 0, "selection_performed_before_model_outcomes": True}, indent=2) + "\n")
    manifest = {"status": "draft_features_and_selections_complete_pending_final_review", "final_experiment_frozen": False,
                "inference_enabled": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_protocol_sha256": sha(protocol_path), "dataset_manifest_sha256": sha(data / "manifest.json"),
                "dataset_files": dataset_manifest["files"], "source_files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                 (Path(__file__), Path(__file__).with_name("test_refresh_features_v4.py"),
                  Path(__file__).with_name("selection_methods.py"), Path(__file__).with_name("representation_ablation.py"))],
                "encoder": protocol["encoder"], "encoder_revision": protocol["encoder_revision"], "encoder_assets": model_sources,
                "device": "cpu", "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "features": feature_evidence, "selected_families_sha256": sha(selected_path), "initial_sets": len(selections),
                "input_only_feature_rows": 1000, "gold_augmented_construction_feature_rows": 200,
                "future_gold_used": False, "new_model_outcomes_observed": 0, "elapsed_seconds": time.monotonic() - started,
                "selection_information_note": "The gold arm reveals 200 construction answers. The other 22 sets receive only construction inputs. Future embeddings are computed but never enter selectors.",
                "estimator_note": "Principal old-set scores use uniform means within group. Native facility representative weights are saved as an alternative, not substituted."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "path": str(destination), "initial_sets": len(selections),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    prepare(args.base)


if __name__ == "__main__":
    main()
