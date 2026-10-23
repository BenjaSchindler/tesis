"""Freeze input-only family selectors before the full verifiable-outcome run.

Uses two public construction partitions and base inputs of two controlled
synthetic construction partitions. Answer fields are discarded at loading and
are never accessible to representation or selection code. No API calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.representation_ablation import ENCODERS, run_variant, unit
from experiments.production_eval.selection_methods import POLICIES, select

BASE = ROOT / "output/research/production_eval/experiments_v2"
BUDGETS = {"gsm8k": [20, 50, 100], "bbh_boolean_expressions": [10, 25, 50],
           "structured_records": [5, 10, 15], "inventory_tracking": [5, 10, 15],
           "balanced_mixture": [20, 40, 60]}
RANDOM = {"uniform_random", "stratified_random", "random_plus_targeted"}
GEOMETRIC = {"centroid_proximity", "facility_location"}
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    with os.fdopen(os.open(path, flags, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def load_construction(public_path, synthetic_path):
    """Return only input fields, checking family and input disjointness first."""
    domains, family_roles, hash_roles = defaultdict(list), defaultdict(set), defaultdict(set)
    for path, synthetic in ((public_path, False), (synthetic_path, True)):
        for line in Path(path).read_text().splitlines():
            original = json.loads(line)
            text = original["input_text"]
            if hashlib.sha256(text.encode()).hexdigest() != original["input_hash"]:
                raise ValueError("Source input hash mismatch")
            domain = original["task_family"] if synthetic else original["dataset"]
            role = original["split"] if synthetic else original["data_role"]
            family_roles[original["family_id"]].add(role)
            hash_roles[original["input_hash"]].add(role)
            if role != "construction":
                continue
            if synthetic and original["metadata"]["variant"] != "base":
                continue
            if domain not in BUDGETS:
                raise ValueError("Unexpected domain")
            # Deliberate allowlist: answer and difficulty/oracle fields do not
            # enter this return value, any feature file, or any selector.
            domains[domain].append({
                "example_id": original["family_id"], "family_id": original["family_id"],
                "representative_example_id": original["example_id"],
                "input_text": text, "input_hash": original["input_hash"],
                "dataset": domain, "data_role": "construction",
                "representation_source": "base_variant" if synthetic else "published_input",
            })
    if any(len(roles) != 1 for roles in family_roles.values()):
        raise ValueError("Family overlaps data roles")
    if any(len(roles) != 1 for roles in hash_roles.values()):
        raise ValueError("Exact input overlaps data roles")
    if set(domains) != set(BUDGETS) - {"balanced_mixture"}:
        raise ValueError("Missing required domain")
    for domain, rows in domains.items():
        rows.sort(key=lambda row: row["family_id"])
        if len({row["family_id"] for row in rows}) != len(rows):
            raise ValueError("More than one representative per construction family")
        if max(BUDGETS[domain]) >= len(rows):
            raise ValueError("Budget must be smaller than pool")
    return dict(domains), {"family_roles_disjoint": True, "exact_inputs_disjoint_across_roles": True,
                           "total_source_families": len(family_roles),
                           "construction_counts": {key: len(value) for key, value in domains.items()}}


def input_features(rows):
    """Equal-frequency rank quartiles and top fifth, using input characters."""
    n = len(rows)
    ordered = sorted(range(n), key=lambda i: (len(rows[i]["input_text"]), rows[i]["family_id"]))
    strata, risk = [None] * n, np.zeros(n, dtype=float)
    for rank, index in enumerate(ordered):
        strata[index] = f"length_quartile_{min(3, 4 * rank // n)}"
        risk[index] = float(rank >= n - math.ceil(n * 0.2))
    return strata, risk


def qwen_encode(rows, model, device, folder):
    import torch
    if model.prompts.get("document") != "" or model.default_prompt_name is not None:
        raise ValueError("Unexpected Qwen document prompt configuration")
    model.max_seq_length = 8192
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    texts = [row["input_text"] for row in rows]
    full = model.tokenizer([text.strip() for text in texts], truncation=False)["input_ids"]
    native = model.tokenize(texts)
    token_rows, groups = [], defaultdict(list)
    for row, complete, actual, mask in zip(rows, full, native["input_ids"], native["attention_mask"]):
        tokens = actual[mask.bool()].tolist()
        token_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        groups[token_hash].append(row["family_id"])
        token_rows.append({"family_id": row["family_id"], "representative_example_id": row["representative_example_id"],
                           "input_hash": row["input_hash"], "untruncated_token_count": len(complete),
                           "actual_token_count": len(tokens), "omitted_token_count": len(complete) - len(tokens),
                           "actual_token_ids": tokens, "actual_tokens_sha256": token_hash})
    embedding = model.encode(texts, prompt_name="document", batch_size=4, show_progress_bar=False,
                             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    embedding = unit(embedding).astype(np.float32)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    path = folder / "qwen3_document.npz"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
        np.savez_compressed(stream, family_ids=np.asarray([row["family_id"] for row in rows]),
                            input_hashes=np.asarray([row["input_hash"] for row in rows]), embeddings=embedding)
    write_json(folder / "qwen3_document_tokens.json", token_rows)
    return embedding, {"variant": "qwen3_document", "model": "Qwen/Qwen3-Embedding-0.6B",
                       "revision": QWEN_REVISION, "dtype": str(next(model.parameters()).dtype),
                       "attention_implementation": model[0].auto_model.config._attn_implementation,
                       "prompt_name": "document", "prompt_text": "", "batch_size": 4,
                       "max_seq_length": 8192, "pooling": "last_token_then_unit_normalization",
                       "requests_with_tokens_omitted": sum(row["omitted_token_count"] > 0 for row in token_rows),
                       "distinct_representation_token_sequences": len(groups),
                       "token_collision_groups": [value for value in groups.values() if len(value) > 1],
                       "total_actual_nonpadding_tokens": sum(row["actual_token_count"] for row in token_rows),
                       "elapsed_seconds": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "embedding_file": str(path.relative_to(ROOT)), "embedding_sha256": sha(path)}


def make_selections(domain, rows, embeddings, strata_override=None):
    ids = [row["family_id"] for row in rows]
    strata, risk = input_features(rows)
    if strata_override is not None:
        if len(strata_override) != len(rows):
            raise ValueError("Wrong override strata length")
        strata = list(strata_override)
    runs = []
    for variant, matrix in embeddings.items():
        policies = POLICIES if variant == "mpnet_head" else sorted(GEOMETRIC)
        for policy in policies:
            for budget in BUDGETS[domain]:
                for seed in range(100, 110) if policy in RANDOM else [None]:
                    start = time.monotonic()
                    choice = select(policy, ids, budget, seed=seed if seed is not None else 0,
                                    embeddings=matrix, texts=[row["input_text"] for row in rows],
                                    strata=strata, risk=risk, targeted_fraction=0.5,
                                    risk_threshold=0.5, num_hashes=128, shingle_size=3, minhash_seed=0)
                    selected = [ids[i] for i in choice.indices]
                    if len(set(selected)) != budget or not set(selected) <= set(ids):
                        raise ValueError("Invalid selected family budget")
                    if not np.isclose(choice.sample_weights.sum(), 1) or np.any(choice.sample_weights <= 0):
                        raise ValueError("Invalid sample weights")
                    if choice.inclusion_probabilities is not None:
                        pi = choice.inclusion_probabilities
                        if np.any(pi <= 0) or np.any(pi > 1) or not np.isclose(pi.sum(), budget):
                            raise ValueError("Invalid inclusion probabilities")
                        expected = 1 / (len(ids) * pi[choice.indices])
                        if not np.allclose(expected, choice.sample_weights):
                            raise ValueError("HT weights disagree with inclusion probabilities")
                    feature_variant = variant if policy in GEOMETRIC else None
                    estimator = "horvitz_thompson_finite_pool_mean" if choice.inclusion_probabilities is not None else "nearest_representative_mass_approximation"
                    runs.append({"run_id": f"{domain}/{feature_variant or 'input_only'}/{policy}/b{budget}/seed{seed}",
                                 "dataset": domain, "policy": policy, "budget": budget, "seed": seed,
                                 "selected_family_ids": selected, "sample_weights": choice.sample_weights.tolist(),
                                 "embedding_variant": feature_variant, "estimator": estimator,
                                 "construction_family_ids": ids,
                                 "population_inclusion_probabilities": None if choice.inclusion_probabilities is None else choice.inclusion_probabilities.tolist(),
                                 "elapsed_seconds": time.monotonic() - start, "selection_metadata": choice.metadata})
    return runs


def make_mixture(domains, features, folder):
    """Balanced artificial pool fixed by RNG before any outcome is read."""
    rng = np.random.default_rng(20260917)
    selected = []
    for domain in sorted(domains):
        rows = domains[domain]
        indices = list(range(len(rows))) if len(rows) == 25 else sorted(rng.choice(len(rows), size=25, replace=False).tolist())
        selected.extend((domain, index, rows[index]) for index in indices)
    selected.sort(key=lambda value: value[2]["family_id"])
    rows = [row for _, _, row in selected]
    family_domain = {row["family_id"]: domain for domain, _, row in selected}
    matrices = {variant: np.stack([features[domain][variant][index] for domain, index, _ in selected])
                for variant in next(iter(features.values()))}
    location = folder / "balanced_mixture"
    location.mkdir(parents=True, exist_ok=True)
    evidence = []
    for variant, matrix in matrices.items():
        path = location / (variant + ".npz")
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
            np.savez_compressed(stream, family_ids=np.asarray([row["family_id"] for row in rows]),
                                input_hashes=np.asarray([row["input_hash"] for row in rows]), embeddings=matrix)
        evidence.append({"dataset": "balanced_mixture", "embedding_variant": variant,
                         "embedding_file": str(path.relative_to(ROOT)), "embedding_sha256": sha(path),
                         "feature_reuse": "Exact rows from the same frozen per-domain embedding arrays; original token diagnostics are retained in each domain folder.",
                         "source_rows": [{"source_domain": domain, "source_row_index": index,
                                          "family_id": row["family_id"]} for domain, index, row in selected],
                         "additional_encoder_forward_passes": 0,
                         "standalone_feature_cost": "Not separately timed; shared research embedding cost is already recorded per domain."})
    return rows, matrices, family_domain, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    output = BASE / "frozen_selections.json"
    if output.exists():
        raise RuntimeError("Selections already frozen; never overwrite a frozen result")
    folder = BASE / "selection_features"
    folder.mkdir(parents=True, exist_ok=True)
    public, synthetic = BASE / "benchmarks/normalized_records.jsonl", BASE / "synthetic/normalized_tasks.jsonl"
    domains, role_checks = load_construction(public, synthetic)
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in [public, synthetic]}
    protocol = {"declared_at": datetime.now(timezone.utc).isoformat(), "budgets": BUDGETS,
                "random_seeds": list(range(100, 110)), "deterministic_seed_field": None,
                "representations": ["mpnet_head", "mpnet_head_tail", "mpnet_mean_chunks", "qwen3_document"],
                "nongeometric_embedding_variant": None,
                "strata": "Input Unicode character-count ranks; ties by family_id. floor(4*rank/N) makes four quartiles.",
                "risk": "Top ceil(0.2*N) input-character ranks in same construction pool; not failure or difficulty.",
                "minhash": {"num_hashes": 128, "shingle_size": 3, "minhash_seed": 0},
                "targeted_fraction": 0.5, "risk_threshold": 0.5,
                "selection_unit": "family_id; synthetic representation uses base variant only",
                "source_hashes": source_hashes, "role_checks": role_checks,
                "outcome_access": "No outcomes loaded; answer fields removed by construction-loader allowlist.",
                "synthetic_scope": "Controlled development family sampling; all family variants inherit selected membership.",
                "causal_scope": "Input-only selection frozen before the full generation run; published benchmarks remain retrospective."}
    protocol["balanced_mixture"] = {"sampling_seed": 20260917, "families_per_domain": 25,
                                    "sampling": "One numpy default_rng, sorted domain order, sorted family IDs; choose 25 public families without replacement, retain all 25 synthetic families.",
                                    "strata": "Source domain", "risk": "Global input-character rank top fifth; ties by family_id",
                                    "reference_domain_weights": {key: 0.25 for key in sorted(domains)},
                                    "scope": "Artificial balanced mixture, not production traffic; target is the equal-domain mean of all independent reference families."}
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if any(old.get(key) != value for key, value in protocol.items() if key != "declared_at"):
            raise RuntimeError("A different protocol is already declared")
        protocol = old
    else:
        write_json(protocol_path, protocol, exclusive=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers.utils import logging
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    torch.set_num_threads(8)
    torch.cuda.init()
    features, evidence, input_manifests = {}, [], {}
    for domain, rows in domains.items():
        path = folder / domain
        path.mkdir(parents=True, exist_ok=True)
        strata, risk = input_features(rows)
        input_manifest = [{"family_id": row["family_id"], "representative_example_id": row["representative_example_id"],
                           "input_hash": row["input_hash"], "input_characters": len(row["input_text"]),
                           "stratum": strata[i], "risk_length_flag": bool(risk[i])} for i, row in enumerate(rows)]
        input_manifests[domain] = input_manifest
        write_json(path / "input_features.json", input_manifest)
        features[domain] = {}
    name, revision = ENCODERS["mpnet"]
    snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
    start = time.monotonic()
    model = SentenceTransformer(str(snapshot), device=args.device, local_files_only=True, trust_remote_code=False)
    model.eval()
    mpnet_load_seconds = time.monotonic() - start
    for domain, rows in domains.items():
        sequences = model.tokenizer([row["input_text"].strip() for row in rows], add_special_tokens=False, truncation=False)["input_ids"]
        for variant, output_name in [("head", "mpnet_head"), ("head_tail", "mpnet_head_tail"), ("chunk_mean", "mpnet_mean_chunks")]:
            embedding, record, _ = run_variant(model, rows, sequences, [], variant, args.device, folder / domain)
            features[domain][output_name] = embedding
            record.update(dataset=domain, embedding_variant=output_name, model=name, revision=revision,
                          model_load_seconds=mpnet_load_seconds)
            evidence.append(record)
            print(json.dumps({"feature": domain + "/" + output_name, "rows": len(rows),
                              "omitted": record["requests_with_tokens_omitted"], "seconds": round(record["elapsed_seconds"], 3)}), flush=True)
    del model
    torch.cuda.empty_cache()
    # Check every pinned Qwen asset against the completed download manifest.
    assets = json.loads((BASE / "models/models_manifest.json").read_text())["assets"]
    qwen_assets = [item for item in assets if item["repo"] == "Qwen/Qwen3-Embedding-0.6B"]
    if not qwen_assets or any(item["revision"] != QWEN_REVISION or sha(item["local_path"]) != item["sha256"] for item in qwen_assets):
        raise RuntimeError("Pinned Qwen assets failed hash verification")
    start = time.monotonic()
    model = SentenceTransformer(str(BASE / "models/Qwen3-Embedding-0.6B"), device=args.device,
                                local_files_only=True, trust_remote_code=False,
                                model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"})
    model.eval()
    qwen_load_seconds = time.monotonic() - start
    for domain, rows in domains.items():
        embedding, record = qwen_encode(rows, model, args.device, folder / domain)
        features[domain]["qwen3_document"] = embedding
        record.update(dataset=domain, embedding_variant="qwen3_document", model_load_seconds=qwen_load_seconds)
        evidence.append(record)
        print(json.dumps({"feature": domain + "/qwen3_document", "rows": len(rows),
                          "omitted": record["requests_with_tokens_omitted"], "seconds": round(record["elapsed_seconds"], 3)}), flush=True)
    del model
    torch.cuda.empty_cache()
    runs = []
    for domain, rows in domains.items():
        runs.extend(make_selections(domain, rows, features[domain]))
    mixture_rows, mixture_features, family_domain, mixture_evidence = make_mixture(domains, features, folder)
    evidence.extend(mixture_evidence)
    mixture_runs = make_selections("balanced_mixture", mixture_rows, mixture_features,
                                  strata_override=[family_domain[row["family_id"]] for row in mixture_rows])
    weights = {key: 0.25 for key in sorted(domains)}
    for run in mixture_runs:
        run.update(family_domain=family_domain, reference_domain_weights=weights,
                   reference_target="Equal-domain mean of all independent reference families; artificial mixture.")
    runs.extend(mixture_runs)
    _, mixture_risk = input_features(mixture_rows)
    input_manifests["balanced_mixture"] = [{"family_id": row["family_id"],
                                            "representative_example_id": row["representative_example_id"],
                                            "input_hash": row["input_hash"], "input_characters": len(row["input_text"]),
                                            "stratum": family_domain[row["family_id"]],
                                            "risk_length_flag": bool(mixture_risk[i])}
                                           for i, row in enumerate(mixture_rows)]
    write_json(folder / "balanced_mixture/input_features.json", input_manifests["balanced_mixture"])
    if len(runs) != 585 or len({run["run_id"] for run in runs}) != len(runs):
        raise RuntimeError("Wrong experiment count or duplicate run IDs")
    if any(sha(ROOT / path) != value for path, value in source_hashes.items()):
        raise RuntimeError("Input sources changed during freeze")
    verification = {**role_checks, "exact_run_count": len(runs), "unique_run_ids": True,
                    "budgets_exact": True, "selected_families_in_construction": True,
                    "positive_normalized_weights": True, "inclusion_probabilities_checked": True,
                    "source_hashes_still_match": True, "qwen_assets_hash_verified": True}
    frozen_at = datetime.now(timezone.utc).isoformat()
    result = {"status": "selections_frozen", "frozen_timestamp": frozen_at, "created_at": frozen_at,
              "protocol": protocol, "sources_hash": source_hashes, "input_manifest": input_manifests,
              "script_hashes": {str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__),
                               ROOT / "experiments/production_eval/representation_ablation.py",
                               ROOT / "experiments/production_eval/selection_methods.py"]},
              "feature_evidence": evidence, "verification": verification, "runs": runs,
              "family_domain": family_domain, "reference_domain_weights": weights,
              "api_calls": 0, "outcome_labels_read": 0,
              "torch_version": torch.__version__, "numpy_version": np.__version__}
    write_json(output, result, exclusive=True)
    write_json(folder / "freeze_verification.json", {**verification, "frozen_manifest_sha256": sha(output)})
    print(json.dumps({"status": result["status"], "runs": len(runs), "sha256": sha(output), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
