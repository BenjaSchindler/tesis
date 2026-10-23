"""Run the six selectors on real feasibility inputs with pinned local encoders.

This is an engineering rehearsal, not a longitudinal quality experiment. It
does not call a generative model or create validated outcome labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.selection_methods import POLICIES, select
from experiments.production_eval.audit_records import load_audits

OUT = ROOT / "output/research/production_eval/execution"
ENCODERS = {
    "mpnet": (
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    ),
}
RANDOM_POLICIES = {"uniform_random", "stratified_random", "random_plus_targeted"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def encode(rows, encoder, device, output):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()
    transformer_logging.set_verbosity_error()
    if device == "cpu":
        torch.set_num_threads(8)

    name, revision = ENCODERS[encoder]
    snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError("Pinned encoder snapshot is not cached locally")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if device.startswith("cuda"):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True)
    texts = [row["input_text"] for row in rows]
    lengths = [len(model.tokenizer.encode(text, truncation=False)) for text in texts]
    features = model.tokenize(texts)
    token_groups = defaultdict(list)
    for row, token_ids, mask in zip(rows, features["input_ids"], features["attention_mask"]):
        scored_tokens = token_ids[mask.bool()].tolist()
        token_hash = hashlib.sha256(json.dumps(scored_tokens).encode()).hexdigest()
        token_groups[token_hash].append(row["example_id"])
    embedding = model.encode(texts, batch_size=32, show_progress_bar=False,
                             normalize_embeddings=False, convert_to_numpy=True)
    if not np.isfinite(embedding).all():
        raise RuntimeError("Nonfinite embeddings")
    path = output / (encoder + "_embeddings.npz")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
        np.savez_compressed(stream, example_ids=np.array([row["example_id"] for row in rows]),
                            embeddings=embedding)
    result = {
        "model": name, "revision": revision, "device": device,
        "shape": list(embedding.shape), "elapsed_seconds": round(time.monotonic() - started, 3),
        "max_seq_length": model.max_seq_length,
        "requests_exceeding_encoder_limit": sum(length > model.max_seq_length for length in lengths),
        "distinct_tokenized_inputs_after_truncation": len(token_groups),
        "identical_tokenized_input_groups": [group for group in token_groups.values() if len(group) > 1],
        "token_collision_scope": "Equality of actual input tokens after this encoder's preprocessing/truncation. Distinct request payloads may be hidden behind identical leading templates.",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "truncation_note": "Encoder uses its first max_seq_length tokens; raw input is preserved for audit.",
        "file": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve()),
        "file_sha256": file_hash(path),
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return embedding, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT / "audit_requests.jsonl")
    parser.add_argument("--audit", type=Path, default=OUT / "audit.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUT / "selector_rehearsal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cohort", choices=["technical", "machine_eligible"], default="machine_eligible")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    annotations = load_audits(args.audit, rows)
    rows = [row for row in rows if row.get("technical_eligible")]
    if args.cohort == "machine_eligible":
        rows = [row for row in rows if annotations.get(row["input_hash"], {}).get("eligible_machine_prediction")]
    rows.sort(key=lambda row: row["example_id"])
    if len(rows) < 16:
        raise RuntimeError("Fewer than 16 real requests in requested rehearsal cohort")
    ids = [row["example_id"] for row in rows]
    texts = [row["input_text"] for row in rows]
    strata = [annotations.get(row["input_hash"], {}).get("annotation", {}).get("task_family", "unknown") for row in rows]
    lengths = np.array([len(text) for text in texts])
    risk_threshold_characters = float(np.quantile(lengths, 0.8))
    risk = (lengths >= risk_threshold_characters).astype(float)
    target_cells = len(set(zip(strata, risk.tolist())))
    budget = min(100, max(target_cells, len(rows) // 2))
    if budget >= len(rows):
        raise RuntimeError("Cohort too small for a noncensus selector rehearsal")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    encoder_results = {}
    selected_sets = {}
    for encoder in ENCODERS:
        embeddings, evidence = encode(rows, encoder, args.device, args.output_dir)
        encoder_results[encoder] = evidence
        unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-12)
        for policy in POLICIES:
            # Lexical/probability selections do not depend on an encoder. Run
            # these once; repeat only geometric policies for representation QA.
            if encoder == "minilm" and policy not in {"centroid_proximity", "facility_location"}:
                continue
            for seed in ([17, 29, 43] if policy in RANDOM_POLICIES else [17]):
                started = time.monotonic()
                selection = select(policy, ids, budget, seed=seed, embeddings=embeddings,
                                   texts=texts, strata=strata, risk=risk, targeted_fraction=0.5,
                                   num_hashes=128, shingle_size=3, minhash_seed=0)
                elapsed = time.monotonic() - started
                selected_ids = [ids[i] for i in selection.indices]
                assert len(set(selected_ids)) == budget
                assert np.isclose(selection.sample_weights.sum(), 1.0)
                if selection.inclusion_probabilities is not None:
                    assert np.all(selection.inclusion_probabilities > 0)
                weights = selection.sample_weights
                estimate_length = float(weights @ lengths[selection.indices])
                geometry_coverage = float((unit @ unit[selection.indices].T).max(axis=1).mean())
                key = f"{encoder}/{policy}/{seed}"
                selected_sets[key] = set(selected_ids)
                record = {
                    "run_id": key, "encoder": encoder, "policy": policy, "seed": seed,
                    "budget": budget, "elapsed_seconds": round(elapsed, 6),
                    "selected_ids": selected_ids, "sample_weights": weights.tolist(),
                    "population_inclusion_probabilities": None if selection.inclusion_probabilities is None else selection.inclusion_probabilities.tolist(),
                    "selected_family_counts": dict(Counter(strata[i] for i in selection.indices)),
                    "selected_high_length_count": int(risk[selection.indices].sum()),
                    "estimated_pool_mean_input_characters": estimate_length,
                    "absolute_error_pool_mean_input_characters": abs(estimate_length - float(lengths.mean())),
                    "mean_max_cosine_to_selected_in_same_encoder": geometry_coverage,
                    "diagnostic_warning": "Length and embedding coverage are engineering checks, not model-quality outcomes. Geometry coverage uses the selection encoder.",
                    "selection_metadata": selection.metadata,
                }
                results.append(record)
                print(json.dumps({"run": key, "selected": budget, "seconds": round(elapsed, 3)}), flush=True)
    overlaps = {}
    for policy in ("centroid_proximity", "facility_location"):
        first, second = (selected_sets[f"{encoder}/{policy}/17"] for encoder in ENCODERS)
        overlaps[policy] = {"intersection_count": len(first & second),
                            "jaccard": len(first & second) / len(first | second)}
    summary = {
        "status": "technical_rehearsal_completed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "cohort": args.cohort, "pool_size": len(rows), "selected_budget": budget,
        "input_sha256": file_hash(args.input), "audit_sha256": file_hash(args.audit),
        "input_ids": ids, "task_family_counts": dict(Counter(strata)),
        "risk_signal": "Input length at or above within-feasibility-pool 80th percentile; a length-targeting heuristic, not observed failure or model difficulty.",
        "risk_threshold_characters": risk_threshold_characters,
        "pool_mean_input_characters": float(lengths.mean()),
        "acquisition_seeds": [17, 29, 43], "encoders": encoder_results,
        "encoder_selection_overlap": overlaps, "runs": results,
        "outcome_labels_collected": 0, "human_labels_collected": 0,
        "longitudinal_predictive_performance_measured": False,
        "interpretation": "Verifies real-data parsing, local embedding execution, matched counts, estimator plumbing and representation sensitivity. No selector winner can be inferred without independent paired system outcomes and later reference requests.",
    }
    private_json(args.output_dir / "results.json", summary)
    print(json.dumps({"status": summary["status"], "pool_size": len(rows), "budget": budget,
                      "selector_runs": len(results), "cross_encoder_overlap": overlaps,
                      "output": str(args.output_dir / "results.json")}), flush=True)


if __name__ == "__main__":
    main()
