"""Freeze v4 current-input predictors after construction, before future outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.production_eval.refresh_run_contract_v4 import BASE, ROOT, CONFIG_IDS, read_inputs, resolve, sha, validate_stage
from experiments.production_eval.refresh_estimators_v4 import nearest_group_predictions
from experiments.production_eval.synthetic_tasks import canonical, digest, json_exact_grade


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def scoped_answers(gold_rows, expected):
    answers = {}
    for row in gold_rows:
        identifier = row["example_id"]
        if identifier not in expected:
            continue  # Future/fixture answers never enter the construction map.
        if identifier in answers or row["family_id"] != expected[identifier]["family_id"]:
            raise ValueError("Duplicate or mismatched construction gold")
        if digest(row["answer"]) != row["answer_hash"]:
            raise ValueError("Construction answer hash changed")
        answers[identifier] = row["answer"]
    if set(answers) != set(expected):
        raise ValueError("Incomplete construction gold")
    return answers


def grade_construction(rows, expected, answers, config_id, run_hash):
    if len(rows) != len(expected) or {r["example_id"] for r in rows} != set(expected):
        raise ValueError("Complete unique construction outcomes required")
    labels = {}
    for row in rows:
        original = expected[row["example_id"]]
        if row.get("status") != "success" or row.get("partition") != "construction" or row.get("config_id") != config_id:
            raise ValueError("Only observed construction outcomes of the designated model are permitted")
        if row.get("input_hash") != original["input_hash"] or row.get("family_id") != original["family_id"]:
            raise ValueError("Construction output input identity changed")
        if row.get("run_manifest_sha256") != run_hash:
            raise ValueError("Construction model output belongs to another run")
        text = row.get("response")
        if text is not None and row.get("response_sha256") != digest(text):
            raise ValueError("Recorded raw response hash mismatch")
        labels[original["family_id"]] = float(json_exact_grade(text, answers[row["example_id"]]) if isinstance(text, str) else False)
    if len(labels) != len(expected):
        raise ValueError("Construction contains duplicate families")
    return labels


def load_features(entry, rows):
    if entry.get("uses_gold") is not False or entry.get("model_outcomes_used") is not False:
        raise ValueError("The nearest-neighbor predictor requires input-only representations")
    path = resolve(entry["embedding_file"])
    if sha(path) != entry["embedding_sha256"]:
        raise ValueError("Embedding file changed")
    with np.load(path, allow_pickle=False) as data:
        if data["example_ids"].tolist() != [r["example_id"] for r in rows] or data["input_hashes"].tolist() != [r["input_hash"] for r in rows]:
            raise ValueError("Embedding order or inputs differ from the declared scope")
        matrix = data["embeddings"].copy()
    if entry["family_ids"] != [r["family_id"] for r in rows] or matrix.ndim != 2 or len(matrix) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError("Embedding family identities or numeric matrix invalid")
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, default=BASE / "run_manifest.json")
    args = parser.parse_args()
    checked = validate_stage(args.run_manifest, "gemini_direct", "construction")
    run_hash = checked["provenance"]["run_manifest_sha256"]
    if not any(Path(e["path"]).resolve() == Path(__file__).resolve() for e in checked["provenance"]["code_files"]):
        raise ValueError("Predictor builder must be frozen before construction inference")
    destination = BASE / "predictors"
    gate_path = BASE / "future_predictor_manifest.json"
    if destination.exists() or gate_path.exists():
        raise ValueError("Predictor artifacts already exist; refusing refit or overwrite")
    if list((BASE / "inference").glob("*/future/responses.jsonl")):
        raise ValueError("Cannot freeze predictors after future inference started")
    journal = BASE / "inference/gemini_direct/all_responses.jsonl"
    if journal.exists() and any(r.get("partition") == "future" for r in load_jsonl(journal)):
        raise ValueError("Future Gemini outcomes already exist in the shared journal")
    selection_path = BASE / "selection/selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    for prefix in ("selected_families", "feature_manifest"):
        if sha(resolve(selection[prefix + "_file"])) != selection[prefix + "_sha256"]:
            raise ValueError("Frozen selection dependency changed: " + prefix)
    selected = json.loads(resolve(selection["selected_families_file"]).read_text())["runs"]
    if len(selected) != 23 or len({r["selection_id"] for r in selected}) != 23:
        raise ValueError("Expected all23 initial selections")
    features = json.loads(resolve(selection["feature_manifest_file"]).read_text())["features"]
    dataset = json.loads((BASE / "draft_dataset/manifest.json").read_text())
    gold_path = BASE / "draft_dataset/gold.jsonl"
    if sha(gold_path) != dataset["files"]["gold.jsonl"]["sha256"]:
        raise ValueError("Oracle source changed")
    construction = sorted(checked["rows"], key=lambda r: r["family_id"])
    expected = {r["example_id"]: r for r in construction}
    answers = scoped_answers(load_jsonl(gold_path), expected)
    construction_groups = {r["family_id"]: r["group_id"] for r in construction}
    construction_index = {r["family_id"]: i for i, r in enumerate(construction)}
    construction_matrix = load_features(features["construction_A"], construction)
    future_path = BASE / "draft_dataset/inputs_future.jsonl"
    if sha(future_path) != dataset["files"][future_path.name]["sha256"]:
        raise ValueError("Current input source changed")
    future = read_inputs(future_path, "future")
    scopes = {presentation: sorted([r for r in future if r["presentation"] == presentation], key=lambda r: r["family_id"]) for presentation in ("A", "B")}
    if [r["family_id"] for r in scopes["A"]] != [r["family_id"] for r in scopes["B"]] or len(scopes["A"]) != 400:
        raise ValueError("Current presentation families are not exactly paired")
    matrices = {presentation: load_features(features["future_"+presentation], rows) for presentation, rows in scopes.items()}
    all_labels, construction_sources = {}, []
    for config_id in sorted(CONFIG_IDS):
        path = BASE / "inference" / config_id / "construction/responses.jsonl"
        completed_path = path.parent / "completed.json"
        completed = json.loads(completed_path.read_text())
        if completed.get("status") != "completed" or completed.get("responses_sha256") != sha(path):
            raise ValueError("Construction stage is incomplete or changed")
        completed_run = completed.get("run_manifest_sha256", completed.get("provenance", {}).get("run_manifest_sha256"))
        if completed_run != run_hash:
            raise ValueError("Construction completion belongs to a different frozen run")
        all_labels[config_id] = grade_construction(load_jsonl(path), expected, answers, config_id, run_hash)
        construction_sources.append({"config_id": config_id, "path": str(path), "sha256": sha(path), "count": 200})
    predicted, history = [], []
    for run in selected:
        ids = sorted(run["selected_family_ids"])
        if len(ids) != 80 or len(set(ids)) != 80 or not set(ids) <= set(construction_groups):
            raise ValueError("Initial family selection changed")
        if Counter(construction_groups[i] for i in ids) != {g: 20 for g in set(construction_groups.values())}:
            raise ValueError("Initial group allocation changed")
        subset_matrix = construction_matrix[[construction_index[i] for i in ids]]
        for config_id, labels in all_labels.items():
            golden = {i: labels[i] for i in ids}
            means = {g: sum(golden[i] for i in ids if construction_groups[i] == g)/20 for g in set(construction_groups.values())}
            history.append({"selection_id": run["selection_id"], "policy": run["policy"], "config_id": config_id,
                            "group_means": means, "selected_family_ids": ids})
            for presentation, rows in scopes.items():
                groups = {r["family_id"]: r["group_id"] for r in rows}
                values, neighbors = nearest_group_predictions(ids, golden, construction_groups, groups, subset_matrix, matrices[presentation])
                for family in sorted(values):
                    predicted.append({"selection_id": run["selection_id"], "policy": run["policy"], "config_id": config_id,
                                      "presentation": presentation, "family_id": family, "group_id": groups[family],
                                      "prediction": values[family], "construction_neighbor": neighbors[family]})
    if len(predicted) != 23*3*800 or len(history) != 23*3:
        raise ValueError("Incomplete prediction matrix")
    destination.mkdir()
    files = []
    for name, rows in (("predictions.jsonl", predicted), ("historical_means.jsonl", history),
                       ("construction_grades.jsonl", [{"config_id": c, "family_correctness": labels} for c, labels in all_labels.items()])):
        path = destination / name
        with path.open("x") as stream:
            stream.write("".join(canonical(row)+"\n" for row in rows))
        files.append({"path": str(path), "sha256": sha(path), "rows": len(rows)})
    gate = {"status": "frozen", "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "run_manifest_sha256": run_hash,
            "selection_manifest_sha256": sha(selection_path), "labels_scope": "construction_only",
            "construction_responses": construction_sources, "predictor_files": files,
            "prediction_rows": len(predicted), "initial_selections": 23, "systems": sorted(CONFIG_IDS),
            "current_input_examples": 800, "future_outputs_existed_at_freeze": False,
            "predictor_representation": "Input-only MPNet chunk embeddings for all arms; even the gold-informed selector uses input-only nearest neighbors at prediction time.",
            "source_gold_sha256": sha(gold_path), "construction_gold_answers_used": 200, "future_gold_answers_used_by_predictor": 0,
            "scope": "Labels are correctness on construction. Future procedural answer strings exist in the separate oracle source but are filtered out before construction grading and never supplied to the predictor."}
    with gate_path.open("x") as stream:
        json.dump(gate, stream, indent=2); stream.write("\n")
    print(json.dumps({"status": "future_predictors_frozen", "predictions": len(predicted), "manifest": str(gate_path)}))


if __name__ == "__main__":
    main()
