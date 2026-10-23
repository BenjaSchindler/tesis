"""Shared fail-closed, gold-free inference-stage contract for v4 runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/research/production_eval/experiments_v4"
CONFIG_IDS = frozenset({"random_base_seed17", "random_base_seed29", "gemini_direct"})
STAGE_COUNTS = {"fixture": 32, "construction": 200, "future": 800}
INPUT_FIELDS = frozenset({"example_id", "family_id", "partition", "domain", "group_id", "complexity",
                          "presentation", "input_text", "input_hash", "observable_counts"})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


def verify_entries(entries, label):
    if not entries or len({str(resolve(e["path"]).resolve()) for e in entries}) != len(entries):
        raise ValueError(label + " must list unique nonempty files")
    checked = []
    for entry in entries:
        path = resolve(entry["path"])
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise ValueError(label + " file hash mismatch: " + str(path))
        checked.append({"path": str(path), "sha256": entry["sha256"]})
    return checked


def future_gate(path, run_hash, manifest):
    gate_path = Path(path)
    if not gate_path.is_file():
        raise ValueError("Future inference requires frozen predictions after complete construction")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "frozen" or gate.get("run_manifest_sha256") != run_hash or gate.get("labels_scope") != "construction_only":
        raise ValueError("Future predictor freeze has the wrong run or label scope")
    if gate.get("selection_manifest_sha256") != manifest["selection_manifest_sha256"]:
        raise ValueError("Future predictor used different initial selections")
    construction = gate.get("construction_responses", [])
    if len(construction) != len(CONFIG_IDS) or {e.get("config_id") for e in construction} != CONFIG_IDS:
        raise ValueError("Every prespecified model needs complete construction outputs")
    verify_entries(construction, "Construction response")
    dataset = json.loads((BASE / "draft_dataset/manifest.json").read_text())
    input_path = BASE / "draft_dataset/inputs_construction.jsonl"
    if sha(input_path) != dataset["files"][input_path.name]["sha256"]:
        raise ValueError("Construction inputs changed after dataset freeze")
    expected_inputs = read_inputs(input_path, "construction")
    expected = {r["example_id"] for r in expected_inputs}
    input_hashes = {r["example_id"]: r["input_hash"] for r in expected_inputs}
    for entry in construction:
        rows = [json.loads(line) for line in resolve(entry["path"]).read_text().splitlines()]
        if entry.get("count") != 200 or len(rows) != 200 or {r.get("example_id") for r in rows} != expected:
            raise ValueError("Construction outputs must cover the complete frozen input set")
        if any(r.get("status") != "success" or r.get("config_id") != entry["config_id"] or r.get("partition") != "construction" for r in rows):
            raise ValueError("Unknown or wrong-model outcomes cannot freeze the predictor")
        if any(r.get("input_hash") != input_hashes[r["example_id"]] for r in rows):
            raise ValueError("Construction outputs do not correspond to frozen prompts")
        if any(r.get("run_manifest_sha256") != run_hash for r in rows):
            raise ValueError("Construction output belongs to another frozen run")
    predictors = verify_entries(gate.get("predictor_files", []), "Frozen predictor")
    return {"path": str(gate_path), "sha256": sha(gate_path), "predictor_files_verified": predictors,
            "construction_models_verified": sorted(CONFIG_IDS)}


def read_inputs(path, partition):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != STAGE_COUNTS[partition] or len({r.get("example_id") for r in rows}) != len(rows):
        raise ValueError("Wrong stage input count or duplicate IDs")
    for row in rows:
        if set(row) != INPUT_FIELDS or row["partition"] != partition:
            raise ValueError("Input allowlist or stage mismatch; gold fields forbidden")
        if not isinstance(row["input_text"], str) or hashlib.sha256(row["input_text"].encode()).hexdigest() != row["input_hash"]:
            raise ValueError("Input text hash mismatch")
    return rows


def validate_stage(run_manifest_path, config_id, partition, future_predictor_manifest_path=None):
    """Return rows, config, and verified provenance; never reads gold/programs."""
    if partition not in STAGE_COUNTS or config_id not in CONFIG_IDS:
        raise ValueError("Unknown prespecified stage/model")
    path = Path(run_manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "frozen" or manifest.get("inference_enabled") is not True:
        raise ValueError("Final v4 run is not frozen and enabled")
    configs = manifest.get("configs", [])
    if len(configs) != len(CONFIG_IDS) or {c.get("config_id") for c in configs} != CONFIG_IDS:
        raise ValueError("Run must contain exactly the three prespecified configurations")
    config = next(c for c in configs if c["config_id"] == config_id)
    code = verify_entries(manifest.get("code_files", []), "Frozen inference code")
    if not any(Path(e["path"]).resolve() == Path(__file__).resolve() for e in code):
        raise ValueError("Shared stage contract must itself be frozen")
    required = {"dataset_manifest_sha256": BASE / "draft_dataset/manifest.json",
                "analysis_plan_sha256": BASE / "analysis_plan.json",
                "selection_manifest_sha256": BASE / "selection/selection_manifest.json"}
    for key, artifact in required.items():
        if not artifact.is_file() or sha(artifact) != manifest.get(key):
            raise ValueError("Frozen run dependency changed: " + key)
    dataset = json.loads(required["dataset_manifest_sha256"].read_text())
    input_path = BASE / "draft_dataset" / f"inputs_{partition}.jsonl"
    if sha(input_path) != dataset["files"][input_path.name]["sha256"]:
        raise ValueError("Frozen stage inputs changed")
    schema_path = BASE / "draft_dataset/schemas.json"
    if sha(schema_path) != dataset["files"]["schemas.json"]["sha256"]:
        raise ValueError("Frozen static schemas changed")
    rows = read_inputs(input_path, partition)
    run_hash = sha(path)
    gate = future_gate(future_predictor_manifest_path or BASE / "future_predictor_manifest.json", run_hash, manifest) if partition == "future" else None
    return {"rows": rows, "config": config, "schemas": json.loads(schema_path.read_text()),
            "provenance": {"run_manifest_path": str(path.resolve()), "run_manifest_sha256": run_hash,
                           "input_file": str(input_path), "input_sha256": sha(input_path),
                           "code_files": code, "future_gate": gate}}
