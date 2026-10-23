"""Small, isolated, procedural-data SFT experiment with a local Qwen model.

prepare: make new family partitions and freeze matched-example training arms.
smoke: train only a separately seeded mechanical fixture; never main examples.
train: explicitly run one frozen arm or all arms (not called by preparation).
No APIs, installed dependencies, remote code, benchmark answers or v2 references.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_tasks import (
    DOMAINS, ORACLES, build_inventory, build_records, canonical, derived_seed,
    digest, json_exact_grade, render,
)
from experiments.production_eval.representation_ablation import ENCODERS, run_variant
from experiments.production_eval.selection_methods import select

BASE = ROOT / "output/research/production_eval/experiments_v3"
MODEL = BASE / "models/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
VERSION = "procedural_sft_v3"
MASTER_SEED = 2026092103
ARMS = ("random_base", "coverage_base", "centroid_base", "coverage_random_mix",
        "coverage_mixed_wording", "coverage_paired_wording")
PARAMETERS = {"model": "Qwen/Qwen3-0.6B", "revision": REVISION,
              "training_seeds": [17, 29], "acquisition_seed": 2026092301,
              "examples_per_arm": 64, "examples_per_domain": 32,
              "epochs": 4, "micro_batch_size": 1, "gradient_accumulation": 8,
              "optimizer_steps_per_arm": 32, "learning_rate": 2e-5,
              "optimizer": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01,
              "eps": 1e-8, "gradient_clip_norm": 1.0, "warmup_steps": 2,
              "schedule": "linear_warmup_then_constant", "max_train_sequence_tokens": 2048,
              "max_eval_input_tokens": 8192, "max_new_tokens": 512,
              "parameter_dtype": "float32", "optimizer_state_dtype": "float32",
              "compute_autocast_dtype": "bfloat16", "gradient_checkpointing": True,
              "attention_implementation": "sdpa", "enable_thinking": False,
              "decode": "greedy, use_cache=True, no output repair", "evaluation_batch_size": 1,
              "checkpoint_selection": "Fixed last step; no validation-selected checkpoint or hyperparameters",
              "loss": "Per-example mean causal CE on answer plus EOS only; prompt and padding excluded",
              "training_quantity_control": "64 examples and256 example presentations per arm,32 optimizer steps. Token counts are recorded, not matched.",
              "arm_seeds_interpretation": "Two optimization/dataloader seeds; acquisition selection fixed independently. Not two acquisition replicates."}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC), 0o600), "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def setup():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    from transformers.utils import logging
    logging.disable_progress_bar()
    logging.set_verbosity_error()


def new_tasks():
    rows = []
    specs = [("train_pool", "stationary", 128), ("validation", "stationary", 10),
             ("reference", "stationary", 10), ("reference", "changed_template", 10),
             ("reference", "more_complex", 10)]
    for domain in DOMAINS:
        for role, regime, count in specs:
            for index in range(count):
                family = digest(f"{VERSION}|{MASTER_SEED}|{role}|{domain}|{regime}|{index}")
                builder = build_records if domain == "structured_records" else build_inventory
                problem, _, _ = builder(derived_seed(family, "data"), derived_seed(family, "entity"), regime)
                primary, secondary = ORACLES[domain]
                answer = primary(problem)
                if answer != secondary(problem):
                    raise ValueError("Independent procedural oracles disagree")
                for variant in ("base", "paraphrase"):
                    text = render(domain, problem, variant == "paraphrase", regime, derived_seed(family, "template"))
                    rows.append({"example_id": digest(family + "|" + variant), "family_id": family,
                                 "input_text": text, "input_hash": digest(text), "answer": canonical(answer),
                                 "answer_hash": digest(canonical(answer)), "domain": domain, "role": role,
                                 "variant": variant, "regime": regime,
                                 "label_provenance": "independent_python_and_sql_oracles_agree",
                                 "generator_namespace": VERSION, "master_seed": MASTER_SEED})
    roles = defaultdict(set)
    for row in rows:
        roles[row["family_id"]].add(row["role"])
    if any(len(value) != 1 for value in roles.values()) or len({row["input_hash"] for row in rows}) != len(rows):
        raise ValueError("New task partitions contain family/input leakage")
    # Directly compare IDs/hashes with the previous frozen synthetic source.
    old = ROOT / "output/research/production_eval/experiments_v2/synthetic/normalized_tasks.jsonl"
    if old.exists():
        prior = [json.loads(line) for line in old.read_text().splitlines()]
        if set(roles) & {row["family_id"] for row in prior} or {row["input_hash"] for row in rows} & {row["input_hash"] for row in prior}:
            raise ValueError("New task data overlaps v2 frozen synthetic families/inputs")
    return rows


def token_example(tokenizer, row):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["input_text"]}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
    if tokenizer.decode(answer_ids, skip_special_tokens=False) != row["answer"]:
        raise ValueError("Target cannot round-trip exactly through tokenizer")
    answer_ids.append(tokenizer.eos_token_id)
    if not answer_ids or len(answer_ids) > PARAMETERS["max_new_tokens"]:
        raise ValueError("Target exceeds generation cap")
    if row["role"] == "train_pool" and len(prompt_ids) + len(answer_ids) > PARAMETERS["max_train_sequence_tokens"]:
        raise ValueError("Training sequence exceeds cap; never truncate a procedural task")
    if len(prompt_ids) > PARAMETERS["max_eval_input_tokens"]:
        raise ValueError("Evaluation input exceeds cap; never truncate a procedural task")
    return {"example_id": row["example_id"], "prompt_ids": prompt_ids, "answer_ids": answer_ids,
            "input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids}


def freeze_arms(rows, matrices):
    train = [row for row in rows if row["role"] == "train_pool"]
    lookup = {(row["family_id"], row["variant"]): row for row in train}
    arms = {arm: [] for arm in ARMS}
    selections = {}
    for domain in DOMAINS:
        pool = sorted([row for row in train if row["domain"] == domain and row["variant"] == "base"], key=lambda row: row["family_id"])
        ids = [row["family_id"] for row in pool]
        matrix = matrices[domain]
        seed = PARAMETERS["acquisition_seed"]
        random_ids = [ids[i] for i in select("uniform_random", ids, 32, seed=seed).indices]
        coverage_ids = [ids[i] for i in select("facility_location", ids, 32, embeddings=matrix).indices]
        centroid_ids = [ids[i] for i in select("centroid_proximity", ids, 32, embeddings=matrix).indices]
        coverage_16 = [ids[i] for i in select("facility_location", ids, 16, embeddings=matrix).indices]
        remaining = sorted(set(ids) - set(coverage_16))
        rng = np.random.default_rng(seed)
        mixed_ids = coverage_16 + [remaining[i] for i in sorted(rng.choice(len(remaining), size=16, replace=False).tolist())]
        for arm, families in [("random_base", random_ids), ("coverage_base", coverage_ids),
                              ("centroid_base", centroid_ids), ("coverage_random_mix", mixed_ids)]:
            arms[arm].extend(lookup[(family, "base")]["example_id"] for family in families)
        # Same32 families as coverage_base: exactly16 alternate wordings/domain.
        for index, family in enumerate(sorted(coverage_ids)):
            arms["coverage_mixed_wording"].append(lookup[(family, "paraphrase" if index % 2 else "base")]["example_id"])
        # Same64 total training examples, but only32 independent families overall.
        for family in coverage_16:
            arms["coverage_paired_wording"].extend(lookup[(family, variant)]["example_id"] for variant in ("base", "paraphrase"))
        selections[domain] = {"construction_family_ids": ids, "random_family_ids": random_ids,
                              "coverage_family_ids": coverage_ids, "centroid_family_ids": centroid_ids,
                              "coverage_16_family_ids": coverage_16, "hybrid_family_ids": mixed_ids}
    by_id = {row["example_id"]: row for row in rows}
    result = []
    for arm, examples in arms.items():
        chosen = [by_id[identifier] for identifier in examples]
        if len(examples) != 64 or len(set(examples)) != 64 or Counter(row["domain"] for row in chosen) != {domain: 32 for domain in DOMAINS}:
            raise ValueError("Training quantity/domain matching failed")
        for seed in PARAMETERS["training_seeds"]:
            result.append({"run_id": f"{arm}_seed{seed}", "arm": arm, "training_seed": seed,
                           "training_example_ids": sorted(examples), "training_examples": len(examples),
                           "unique_training_families": len({row["family_id"] for row in chosen}),
                           "domain_counts": dict(Counter(row["domain"] for row in chosen)),
                           "variant_counts": dict(Counter(row["variant"] for row in chosen))})
    return result, selections


def prepare(device):
    manifest_path = BASE / "experiment_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("A v3 experiment is already frozen; refusing overwrite")
    BASE.mkdir(parents=True, exist_ok=True)
    setup()
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    rows = new_tasks()
    tokens = {row["example_id"]: token_example(tokenizer, row) for row in rows}
    data_path = BASE / "tasks.jsonl"
    with data_path.open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    evidence, matrices = [], {}
    name, revision = ENCODERS["mpnet"]
    snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True, trust_remote_code=False)
    model.eval()
    for domain in DOMAINS:
        pool = sorted([row for row in rows if row["domain"] == domain and row["role"] == "train_pool" and row["variant"] == "base"], key=lambda row: row["family_id"])
        path = BASE / "selection_features" / domain
        path.mkdir(parents=True, exist_ok=True)
        sequences = model.tokenizer([row["input_text"].strip() for row in pool], add_special_tokens=False, truncation=False)["input_ids"]
        matrices[domain], record, _ = run_variant(model, pool, sequences, [], "chunk_mean", device, path)
        record.update(domain=domain, model=name, revision=revision)
        evidence.append(record)
    del model
    torch.cuda.empty_cache()
    runs, selections = freeze_arms(rows, matrices)
    for run in runs:
        run["training_input_tokens_per_epoch"] = sum(len(tokens[i]["prompt_ids"]) for i in run["training_example_ids"])
        run["training_target_tokens_per_epoch"] = sum(len(tokens[i]["answer_ids"]) for i in run["training_example_ids"])
    model_manifest = BASE / "models/models_manifest.json"
    if json.loads(model_manifest.read_text())["status"] != "complete":
        raise ValueError("Model download is incomplete")
    result = {"status": "experiment_frozen_not_trained", "created_at": datetime.now(timezone.utc).isoformat(),
              "generator_namespace": VERSION, "master_seed": MASTER_SEED, "parameters": PARAMETERS,
              "runs": runs, "selections": selections, "feature_evidence": evidence,
              "tasks_file": str(data_path.relative_to(ROOT)), "tasks_sha256": sha(data_path),
              "model_manifest_sha256": sha(model_manifest), "script_sha256": sha(Path(__file__)),
              "task_counts": dict(Counter(row["role"] for row in rows)),
              "family_counts": {role: len({row["family_id"] for row in rows if row["role"] == role}) for role in {row["role"] for row in rows}},
              "max_tokens_by_role": {role: {"prompt": max(len(tokens[row["example_id"]]["prompt_ids"]) for row in rows if row["role"] == role),
                                               "target": max(len(tokens[row["example_id"]]["answer_ids"]) for row in rows if row["role"] == role)}
                                     for role in {row["role"] for row in rows}},
              "verification": {"family_roles_disjoint": True, "v2_families_and_inputs_disjoint": True,
                               "all_targets_roundtrip": True, "no_input_truncation": True,
                               "independent_python_sql_oracles_agree": True, "examples_matched": 64},
              "wording_caveat": "Paraphrase rendering also changes whitespace/serialization; it is a joint wording-and-format augmentation.",
              "evaluation": "Strict JSON-value exact match, syntax-validity and family-paired results by domain/regime/wording. Bootstrap families, not sibling rows. No real-production or temporal-drift claim.",
              "sources": ["https://huggingface.co/Qwen/Qwen3-0.6B", "https://arxiv.org/abs/2505.09388"]}
    write_json(manifest_path, result, exclusive=True)
    print(json.dumps({"status": result["status"], "runs": len(runs), "counts": result["task_counts"],
                      "max_tokens_by_role": result["max_tokens_by_role"], "manifest": str(manifest_path)}), flush=True)


def fixture_rows():
    # Distinct minimal arithmetic fixtures make the backward check cheap and
    # have no families or input hashes shared with any main data partition.
    values = []
    rng = random.Random(derived_seed(VERSION, "smoke_only", 987123))
    for index in range(8):
        a, b = rng.randint(2, 90), rng.randint(2, 90)
        text = f'Compute {a}+{b}. Return only JSON with one integer key total.'
        family = digest(f"{VERSION}|smoke_only|{index}")
        values.append({"example_id": family, "family_id": family, "role": "smoke_only",
                       "input_text": text, "input_hash": digest(text), "answer": canonical({"total": a + b})})
    return values


def masked_tensors(example, device):
    import torch
    ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    return ids, labels


def response_only_loss(model, ids, labels):
    """Standard causal loss, computing vocabulary logits only where supervised.

    Batch size is one in this pilot. Avoiding prompt-token logits saves memory;
    the numerical equivalence is checked against the library loss in smoke.
    """
    import torch
    hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    target = labels[:, 1:]
    mask = target != -100
    logits = model.lm_head(hidden[:, :-1, :][mask]).float()
    return torch.nn.functional.cross_entropy(logits, target[mask])


def load_model(device):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def train_loop(model, token_rows, seed, device, *, epochs=4, smoke_steps=None, output=None):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    accumulation = 2 if smoke_steps else PARAMETERS["gradient_accumulation"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMETERS["learning_rate"],
                                 betas=tuple(PARAMETERS["betas"]), eps=PARAMETERS["eps"],
                                 weight_decay=PARAMETERS["weight_decay"], foreach=False)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    records, step, micro = [], 0, 0
    norm_total, target_total, input_total = 0., 0, 0
    rng = random.Random(seed)
    for epoch in range(epochs if not smoke_steps else smoke_steps):
        order = list(range(len(token_rows)))
        rng.shuffle(order)
        for index in order:
            example = token_rows[index]
            ids, labels = masked_tensors(example, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = response_only_loss(model, ids, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / accumulation).backward()
            norm_total += float(loss.detach())
            target_total += len(example["answer_ids"])
            input_total += len(example["prompt_ids"])
            micro += 1
            if micro % accumulation:
                continue
            step += 1
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), PARAMETERS["gradient_clip_norm"]))
            if not math.isfinite(grad_norm):
                raise ValueError("Nonfinite gradient norm")
            lr = PARAMETERS["learning_rate"] * min(1., step / PARAMETERS["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, "epoch_index": epoch, "mean_loss": norm_total / accumulation,
                      "unclipped_gradient_norm": grad_norm, "learning_rate": lr,
                      "cumulative_input_tokens": input_total, "cumulative_target_tokens": target_total,
                      "cumulative_example_presentations": micro, "elapsed_seconds": time.monotonic() - start}
            records.append(record)
            if output:
                with output.open("a") as stream:
                    stream.write(canonical(record) + "\n")
            norm_total = 0.
            if step % 4 == 0 or smoke_steps:
                print(json.dumps({"optimizer_step": step, "loss": round(record["mean_loss"], 5),
                                  "seconds": round(record["elapsed_seconds"], 2)}), flush=True)
            if smoke_steps and step >= smoke_steps:
                break
        if smoke_steps and step >= smoke_steps:
            break
    if micro % accumulation:
        raise ValueError("Incomplete gradient accumulation would break matched steps")
    torch.cuda.synchronize(device)
    state = next(iter(optimizer.state.values()))
    report = {"optimizer_steps": step, "example_presentations": micro, "elapsed_seconds": time.monotonic() - start,
              "input_tokens": input_total, "target_tokens": target_total,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "parameter_dtype": str(next(model.parameters()).dtype), "optimizer_exp_avg_dtype": str(state["exp_avg"].dtype),
              "optimizer_exp_avg_sq_dtype": str(state["exp_avg_sq"].dtype), "records": records}
    del optimizer
    return report


def smoke(device):
    setup()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    path = BASE / "smoke"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "results.json").exists():
        raise RuntimeError("Smoke result exists; do not silently repeat or overwrite")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    fixtures = fixture_rows()
    examples = [token_example(tokenizer, row) for row in fixtures]
    model = load_model(device)
    model.eval()
    ids, labels = masked_tensors(examples[0], device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        standard = float(model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss)
        optimized = float(response_only_loss(model, ids, labels))
    if abs(standard - optimized) > 2e-4:
        raise ValueError("Response-only loss differs from standard masked causal loss")
    parameter = next(model.parameters())
    before = parameter.detach().float().norm().item()
    report = train_loop(model, examples, 991, device, smoke_steps=3)
    after = parameter.detach().float().norm().item()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after_loss = float(response_only_loss(model, ids, labels))
    result = {"status": "smoke_completed", "completed_at": datetime.now(timezone.utc).isoformat(),
              "model_revision": REVISION, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "fixtures": [{"example_id": row["example_id"], "input_hash": row["input_hash"]} for row in fixtures],
              "standard_masked_loss": standard, "optimized_response_loss": optimized,
              "loss_abs_difference": abs(standard - optimized), "same_fixture_loss_after": after_loss,
              "first_parameter_norm_before": before, "first_parameter_norm_after": after,
              "main_training_examples_used": 0, "validation_or_reference_examples_used": 0,
              "training": report, "model_saved": False,
              "interpretation": "Mechanical backward/optimizer/memory check on independent tiny fixtures, not adaptation effectiveness or a chosen hyperparameter result."}
    write_json(path / "results.json", result, exclusive=True)
    print(json.dumps({key: result[key] for key in ["status", "trainable_parameters", "loss_abs_difference", "same_fixture_loss_after"]}), flush=True)
    print(json.dumps({"smoke_training": report}), flush=True)


def evaluate(model, tokenizer, rows, device, path):
    import torch
    model.eval()
    model.gradient_checkpointing_disable()
    records = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            token = token_example(tokenizer, row)
            ids = torch.tensor([token["prompt_ids"]], device=device)
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           max_new_tokens=PARAMETERS["max_new_tokens"], do_sample=False,
                                           use_cache=True, pad_token_id=tokenizer.pad_token_id)
            output_ids = generated[0, len(token["prompt_ids"]):].tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            record = {"example_id": row["example_id"], "family_id": row["family_id"],
                      "input_hash": row["input_hash"], "domain": row["domain"], "role": row["role"],
                      "regime": row["regime"], "variant": row["variant"], "response": output,
                      "generated_tokens": len(output_ids), "generation_limit_hit": len(output_ids) >= PARAMETERS["max_new_tokens"] and output_ids[-1] != tokenizer.eos_token_id,
                      "exact_match": json_exact_grade(output, row["answer"]),
                      "elapsed_seconds": time.monotonic() - start}
            records.append(record)
            with path.open("a") as stream:
                stream.write(canonical(record) + "\n")
            if (index + 1) % 20 == 0:
                print(json.dumps({"evaluated": index + 1, "of": len(rows)}), flush=True)
    return records


def train(device, requested_run):
    setup()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    manifest_path = BASE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["parameters"] != PARAMETERS or sha(ROOT / manifest["tasks_file"]) != manifest["tasks_sha256"]:
        raise ValueError("Frozen experiment settings or task data changed")
    if not (BASE / "smoke/results.json").exists():
        raise ValueError("Mechanical smoke must complete before full training")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in (ROOT / manifest["tasks_file"]).read_text().splitlines()]
    by_id = {row["example_id"]: row for row in rows}
    evaluation = [row for row in rows if row["role"] in {"validation", "reference"}]
    selected_runs = manifest["runs"] if requested_run == "all" else [run for run in manifest["runs"] if run["run_id"] == requested_run]
    if not selected_runs:
        raise ValueError("Unknown frozen run ID")
    # A standalone baseline is generated once before any selected training.
    baseline = BASE / "baseline"
    if not (baseline / "completed.json").exists():
        baseline.mkdir(parents=True, exist_ok=True)
        if (baseline / "responses.jsonl").exists():
            raise RuntimeError("Partial baseline exists; inspect it rather than overwrite")
        model = load_model(device)
        baseline_records = evaluate(model, tokenizer, evaluation, device, baseline / "responses.jsonl")
        write_json(baseline / "completed.json", {"evaluated": len(baseline_records), "manifest_sha256": sha(manifest_path),
                                               "completed_at": datetime.now(timezone.utc).isoformat()}, exclusive=True)
        del model
        torch.cuda.empty_cache()
    for run in selected_runs:
        destination = BASE / "runs" / run["run_id"]
        if destination.exists():
            raise RuntimeError("Run directory exists; inspect state instead of restarting or overwriting")
        destination.mkdir(parents=True)
        write_json(destination / "started.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                                                  "manifest_sha256": sha(manifest_path), "run": run}, exclusive=True)
        torch.manual_seed(run["training_seed"])
        model = load_model(device)
        examples = [token_example(tokenizer, by_id[i]) for i in run["training_example_ids"]]
        report = train_loop(model, examples, run["training_seed"], device, output=destination / "training_steps.jsonl")
        if report["optimizer_steps"] != PARAMETERS["optimizer_steps_per_arm"] or report["example_presentations"] != 256:
            raise ValueError("Matched training schedule violated")
        # Save actual FP32 master weights used for evaluation, not a lossy
        # conversion whose predictions might differ from the measured model.
        model.save_pretrained(str(destination / "checkpoint"), safe_serialization=True)
        tokenizer.save_pretrained(str(destination / "checkpoint"))
        records = evaluate(model, tokenizer, evaluation, device, destination / "responses.jsonl")
        write_json(destination / "completed.json", {"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                                                    "run": run, "training": report, "evaluated": len(records),
                                                    "manifest_sha256": sha(manifest_path)}, exclusive=True)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"completed_run": run["run_id"], "evaluated": len(records)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "smoke", "train"])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--run", default="all")
    args = parser.parse_args()
    {"prepare": lambda: prepare(args.device), "smoke": lambda: smoke(args.device),
     "train": lambda: train(args.device, args.run)}[args.mode]()


if __name__ == "__main__":
    main()
