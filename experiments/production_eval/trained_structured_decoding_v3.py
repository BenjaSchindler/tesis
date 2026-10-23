"""Frozen exploratory JSON-decoding follow-up for both existing random SFT seeds.

No training or checkpoint selection. Each original trained checkpoint is read
without modification; the static grammar and evaluation inputs come from the
already frozen base-model control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.structured_decoding_v3 import (
    BASE, CONTROL, PARAMETERS, DomainPrefixFilter, frozen_tasks, input_only,
    prompt_tokens, sha, tokenizer_cpu, verify_control,
)
from experiments.production_eval.synthetic_sft_v3 import write_json
from experiments.production_eval.synthetic_tasks import canonical, digest, json_exact_grade
from experiments.production_eval.analyze_sft_v3 import format_diagnostics

FOLLOWUP = CONTROL / "trained_random_followup"
RUNS = ("random_base_seed17", "random_base_seed29")
SOURCES = ("trained_structured_decoding_v3.py", "structured_decoding_v3.py", "synthetic_sft_v3.py",
           "synthetic_tasks.py", "analyze_sft_v3.py", "field_diagnostics.py")


def source_hashes():
    return {name: sha(ROOT / "experiments/production_eval" / name) for name in SOURCES}


def checkpoint_inventory(run_id):
    if run_id not in RUNS:
        raise ValueError("Only the two prespecified random seeds are allowed")
    directory = BASE / "runs" / run_id
    completion = json.loads((directory / "completed.json").read_text())
    main_manifest = json.loads((BASE / "experiment_manifest.json").read_text())
    planned = next(run for run in main_manifest["runs"] if run["run_id"] == run_id)
    if completion["status"] != "completed" or completion["manifest_sha256"] != sha(BASE / "experiment_manifest.json") or completion["run"] != planned:
        raise ValueError("Trained checkpoint does not match its frozen experiment")
    if completion["training"]["optimizer_steps"] != 32 or completion["training"]["example_presentations"] != 256:
        raise ValueError("Checkpoint did not receive the frozen amount of training")
    files = {str(path.relative_to(ROOT)): sha(path) for path in sorted((directory / "checkpoint").rglob("*")) if path.is_file()}
    if not any(path.endswith("model.safetensors") for path in files):
        raise ValueError("Missing checkpoint weights")
    return {"run": planned, "checkpoint_files": files,
            "completion_sha256": sha(directory / "completed.json"),
            "original_free_responses_sha256": sha(directory / "responses.jsonl")}


def prepare():
    destination = FOLLOWUP / "followup_manifest.json"
    if destination.exists():
        raise ValueError("Follow-up already frozen; refusing to overwrite")
    base_manifest = verify_control()
    frozen_tasks()
    result = {"status": "frozen_before_trained_constrained_outputs", "created_at": datetime.now(timezone.utc).isoformat(),
              "exploratory": True,
              "motivation": "Evaluate both previously fixed random SFT seeds with the same static grammar as the original model, without choosing a winner. Proposed after observing free SFT outcomes and while the base constrained control was running. No causal subtraction of format from reasoning gains.",
              "run_ids": list(RUNS), "checkpoints": {run: checkpoint_inventory(run) for run in RUNS},
              "control_manifest_sha256": sha(CONTROL / "control_manifest.json"),
              "v3_manifest_sha256": sha(BASE / "experiment_manifest.json"),
              "evaluation_example_ids": base_manifest["evaluation_example_ids"],
              "prompt_token_evidence": base_manifest["prompt_token_evidence"],
              "schemas": base_manifest["schemas"], "parser_settings": base_manifest["parser_settings"],
              "source_hashes": source_hashes(),
              "recipe": {"loading": "AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False, torch_dtype=float32, attn_implementation=sdpa); assert 751632384 unique parameters and untied embedding/head storage",
                         "tokenizer": "Original pinned model tokenizer, exactly as in original SFT evaluation and base control",
                         "compute_dtype": "bfloat16_autocast", "device": "cuda:0", "batch_size": 1,
                         "do_sample": False, "max_new_tokens": PARAMETERS["max_new_tokens"], "enable_thinking": False,
                         "checkpoint_modification": False, "output_repair": False}}
    write_json(destination, result, exclusive=True)
    print(json.dumps({"status": result["status"], "runs": list(RUNS), "manifest_sha256": sha(destination)}), flush=True)


def verify_followup():
    manifest = json.loads((FOLLOWUP / "followup_manifest.json").read_text())
    if manifest["source_hashes"] != source_hashes() or manifest["control_manifest_sha256"] != sha(CONTROL / "control_manifest.json"):
        raise ValueError("Frozen follow-up source changed")
    verify_control()
    if manifest["checkpoints"] != {run: checkpoint_inventory(run) for run in RUNS}:
        raise ValueError("Trained checkpoints or original free outputs changed")
    return manifest


def load_trained(run_id, device):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(BASE / "runs" / run_id / "checkpoint"),
                                               local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model.eval()
    model.gradient_checkpointing_disable()
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    evidence = {"parameter_dtype": str(next(model.parameters()).dtype),
                "unique_parameter_count": sum(p.numel() for p in model.parameters()),
                "config_tie_word_embeddings": model.config.tie_word_embeddings,
                "embedding_lm_head_same_storage": embedding.data_ptr() == head.data_ptr(),
                "embedding_lm_head_equal_values": bool(torch.equal(embedding, head)),
                "generation_config": model.generation_config.to_dict(), "device": str(next(model.parameters()).device)}
    if evidence["parameter_dtype"] != "torch.float32" or evidence["unique_parameter_count"] != 751632384 or evidence["embedding_lm_head_same_storage"]:
        raise ValueError("Loaded trained model differs from the fixed untied FP32 architecture")
    return model, evidence


def run(run_id, device):
    import torch
    manifest = verify_followup()
    if run_id not in manifest["run_ids"]:
        raise ValueError("Run is not frozen")
    base_complete = json.loads((CONTROL / "evaluate/completed.json").read_text())
    if base_complete["evaluated"] != 160 or base_complete["control_manifest_sha256"] != manifest["control_manifest_sha256"]:
        raise ValueError("Base constrained control must finish before this follow-up")
    tokenizer, tokenizer_data = tokenizer_cpu()
    tasks = {row["example_id"]: row for row in frozen_tasks()}
    destination = FOLLOWUP / run_id
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(destination / "started.json", {"started_at": started, "run_id": run_id, "device": device,
                                              "followup_manifest_sha256": sha(FOLLOWUP / "followup_manifest.json")}, exclusive=True)
    model, evidence = load_trained(run_id, device)
    records = []
    start_all = time.monotonic()
    with torch.inference_mode():
        for index, identifier in enumerate(manifest["evaluation_example_ids"]):
            row = input_only(tasks[identifier])
            prompt = prompt_tokens(tokenizer, row)
            if manifest["prompt_token_evidence"][identifier] != {"tokens": len(prompt), "sha256": digest(canonical(prompt))}:
                raise ValueError("Frozen prompt changed")
            ids = torch.tensor([prompt], device=device)
            prefix = DomainPrefixFilter(tokenizer_data, row["domain"], prompt)
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           max_new_tokens=PARAMETERS["max_new_tokens"], do_sample=False,
                                           use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                           prefix_allowed_tokens_fn=prefix)
            output_ids = generated[0, len(prompt):].tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            record = {key: value for key, value in row.items() if key != "input_text"}
            record.update(response=output, generated_tokens=len(output_ids), output_token_ids=output_ids,
                          generation_limit_hit=len(output_ids) >= PARAMETERS["max_new_tokens"] and output_ids[-1] != tokenizer.eos_token_id,
                          elapsed_seconds=time.monotonic() - start, prompt_token_sha256=digest(canonical(prompt)),
                          prefix_filter_calls=prefix.calls, **format_diagnostics(output, row["domain"]))
            record["exact_match"] = json_exact_grade(output, tasks[identifier]["answer"])
            records.append(record)
            with (destination / "responses.jsonl").open("a") as stream:
                stream.write(canonical(record) + "\n")
            if (index + 1) % 10 == 0:
                print(json.dumps({"run_id": run_id, "evaluated": index + 1, "of": 160}), flush=True)
    report = {"status": "completed", "run_id": run_id, "evaluated": len(records), "started_at": started,
              "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start_all,
              "model_evidence": evidence, "strict_json_valid": sum(r["strict_json_valid"] for r in records),
              "required_schema_valid": sum(r["required_schema_valid"] for r in records),
              "exact_match": sum(r["exact_match"] for r in records), "generation_limit_hit": sum(r["generation_limit_hit"] for r in records),
              "responses_sha256": sha(destination / "responses.jsonl"),
              "followup_manifest_sha256": sha(FOLLOWUP / "followup_manifest.json")}
    write_json(destination / "completed.json", report, exclusive=True)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if args.run is None:
            parser.error("--run is required")
        run(args.run, args.device)


if __name__ == "__main__":
    main()
