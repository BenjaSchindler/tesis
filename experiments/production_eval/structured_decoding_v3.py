"""Exploratory, answer-blind JSON grammar control for the frozen HF v3 model.

Preparation and CPU checks do not run a language model. GPU execution is explicit.
The reference outcomes were already inspected before proposing this control.
No training, target-dependent grammar, output repair, or configuration search.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_sft_v3 import (
    BASE, MODEL, PARAMETERS, REVISION, load_model, setup, write_json,
)
from experiments.production_eval.synthetic_tasks import canonical, digest, json_exact_grade
from experiments.production_eval.analyze_sft_v3 import format_diagnostics

CONTROL = BASE / "structured_decoding"
VENDOR = CONTROL / "vendor"
RUN_ID = "hf_fp32_untied_greedy_json_schema"
PARSER_SETTINGS = {"max_consecutive_whitespaces": 12, "force_json_field_order": False,
                   "max_json_array_length": 512}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def schema_for_domain(domain):
    """Only public field names/types; no input IDs, sizes, values or gold access."""
    integer = {"type": "integer"}
    if domain == "structured_records":
        properties = {"selected_ids": {"type": "array", "items": {"type": "string"}},
                      "total_units": integer, "total_value": integer}
    elif domain == "inventory_tracking":
        item = {"type": "object", "properties": {"item": {"type": "string"},
                                                "east": integer, "west": integer},
                "required": ["item", "east", "west"], "additionalProperties": False}
        properties = {"stock": {"type": "array", "items": item}, "total_units": integer}
    else:
        raise ValueError("Unknown domain")
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def activate_vendor():
    if not VENDOR.is_dir():
        raise ValueError("Pinned experiment-only vendor dependencies are missing")
    sys.path.insert(0, str(VENDOR))
    import lmformatenforcer
    if not Path(lmformatenforcer.__file__).resolve().is_relative_to(VENDOR.resolve()):
        raise ValueError("lm-format-enforcer was imported outside the isolated vendor")


def build_tokenizer_data(tokenizer):
    """MIT adaptation of lm-format-enforcer 0.11.3 Transformers integration.

    Original: noamgat/lm-format-enforcer, integrations/transformers.py. The
    published integration imports a removed Transformers 5 module. This local
    adapter uses its unchanged tokenizer conversion and public core API, avoiding
    any vendor patch or global import shim. The original MIT license is vendored.
    """
    activate_vendor()
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    zero = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        after_zero = tokenizer.decode([zero, token_id])[1:]
        alone = tokenizer.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(alone)))
    return TokenEnforcerTokenizerData(regular, lambda ids: tokenizer.decode(ids).rstrip("�"),
                                     tokenizer.eos_token_id, False, len(tokenizer))


class DomainPrefixFilter:
    def __init__(self, tokenizer_data, domain, prompt_ids):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
        from lmformatenforcer.tokenenforcer import TokenEnforcer
        self.prompt_ids = tuple(prompt_ids)
        config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet, **PARSER_SETTINGS)
        parser = JsonSchemaParser(schema_for_domain(domain), config=config)
        self.enforcer = TokenEnforcer(tokenizer_data, parser)
        # 0.11.3 TokenEnforcer resets parser.config; explicitly restore our fixed
        # heuristics after its initialization (alphabet remains identical).
        parser.config = config
        self.calls = 0

    def __call__(self, batch_id, sent):
        sequence = sent.tolist()
        if batch_id != 0 or tuple(sequence[:len(self.prompt_ids)]) != self.prompt_ids:
            raise ValueError("Only this frozen, single-example prompt may use the filter")
        if self.calls == 0 and len(sequence) != len(self.prompt_ids):
            raise ValueError("Filter must initialize at the prompt boundary")
        self.calls += 1
        return self.enforcer.get_allowed_tokens(sequence).allowed_tokens


def input_only(row):
    return {key: row[key] for key in ("example_id", "family_id", "input_hash", "input_text",
                                     "domain", "role", "regime", "variant")}


def prompt_tokens(tokenizer, row):
    if set(row) - {"example_id", "family_id", "input_hash", "input_text", "domain", "role", "regime", "variant"}:
        raise ValueError("Prompt construction accepts only input provenance, never answers")
    if digest(row["input_text"]) != row["input_hash"]:
        raise ValueError("Input hash mismatch")
    text = tokenizer.apply_chat_template([{"role": "user", "content": row["input_text"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > PARAMETERS["max_eval_input_tokens"]:
        raise ValueError("Input exceeds frozen cap; truncation is prohibited")
    return ids


def frozen_tasks():
    manifest_path = BASE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["parameters"] != PARAMETERS or sha(ROOT / manifest["tasks_file"]) != manifest["tasks_sha256"]:
        raise ValueError("Frozen v3 inputs/parameters changed")
    if sha(BASE / "models/models_manifest.json") != manifest["model_manifest_sha256"]:
        raise ValueError("Frozen model manifest changed")
    rows = [json.loads(line) for line in (ROOT / manifest["tasks_file"]).read_text().splitlines()]
    evaluation = [row for row in rows if row["role"] in {"validation", "reference"}]
    if len(evaluation) != 160 or Counter(row["role"] for row in evaluation) != {"validation": 40, "reference": 120}:
        raise ValueError("Expected all 160 frozen validation/reference examples")
    return evaluation


def mechanical_fixtures():
    values = [
        ("structured_records", 'Return exactly a JSON object with selected_ids ["fixture_B","fixture_Q"], total_units -7, and total_value 913.'),
        ("inventory_tracking", 'Return exactly a JSON object with stock [{"item":"fixture_H","east":-5,"west":999}] and total_units 994.'),
    ]
    return [{"example_id": digest("structured_smoke_v3|" + domain),
             "family_id": digest("structured_smoke_v3|" + domain), "domain": domain,
             "role": "independent_smoke", "regime": "mechanical_only", "variant": "base",
             "input_text": text, "input_hash": digest(text)} for domain, text in values]


def cpu_checks(tokenizer, tokenizer_data):
    import torch
    fixture_values = {
        "structured_records": {"selected_ids": [f"unseen_{i}" for i in range(28)], "total_units": -999, "total_value": 987654321},
        "inventory_tracking": {"stock": [{"item": "unseen", "east": -8, "west": 1234567}], "total_units": -77},
    }
    reports = []
    for row in mechanical_fixtures():
        prompt = prompt_tokens(tokenizer, row)
        prefix = DomainPrefixFilter(tokenizer_data, row["domain"], prompt)
        sequence = list(prompt)
        allowed = prefix(0, torch.tensor(sequence))
        if tokenizer.eos_token_id in allowed:
            raise ValueError("EOS allowed before required JSON object")
        output = canonical(fixture_values[row["domain"]])
        target = tokenizer(output, add_special_tokens=False)["input_ids"]
        for token in target:
            if token not in allowed:
                raise ValueError(f"Grammar blocks a semantically arbitrary schema-valid fixture token: {row['domain']}")
            sequence.append(token)
            allowed = prefix(0, torch.tensor(sequence))
        if tokenizer.eos_token_id not in allowed:
            raise ValueError("EOS blocked after valid complete JSON fixture")
        reports.append({"domain": row["domain"], "fixture_output_sha256": digest(output),
                        "fixture_tokens": len(target), "eos_rejected_initially": True,
                        "all_arbitrary_fixture_tokens_allowed": True, "eos_allowed_after_complete_object": True})
    return reports


def dependency_files():
    return {str(path.relative_to(CONTROL)): sha(path) for path in sorted(VENDOR.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def tokenizer_cpu():
    setup()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    data = build_tokenizer_data(tokenizer)
    return tokenizer, data


def prepare():
    path = CONTROL / "control_manifest.json"
    if path.exists():
        raise ValueError("Control already fixed; refusing to overwrite")
    rows = frozen_tasks()
    tokenizer, data = tokenizer_cpu()
    tests = cpu_checks(tokenizer, data)
    prompts = {row["example_id"]: prompt_tokens(tokenizer, input_only(row)) for row in rows}
    replay = []
    for domain in ("structured_records", "inventory_tracking"):
        replay += [row["example_id"] for row in sorted([row for row in rows if row["domain"] == domain],
                                                       key=lambda row: digest("hf_free_replay_v3|" + row["example_id"]))[:10]]
    deps = json.loads((CONTROL / "dependency_manifest.json").read_text())
    for package in deps["packages"]:
        if sha(CONTROL / "packages" / package["filename"]) != package["sha256"]:
            raise ValueError("Downloaded wheel hash mismatch")
    result = {"status": "prepared_before_control_outputs", "created_at": datetime.now(timezone.utc).isoformat(),
              "run_id": RUN_ID, "exploratory": True,
              "interpretation": "Post-hoc mechanism control after viewing free/SFT reference outcomes; no tuning or selection from constrained outcomes. Procedural tasks, not real production traffic.",
              "model_revision": REVISION, "model_loading": "Reuse synthetic_sft_v3.load_model without tying weights; exact FP32 parameters, BF16 autocast, SDPA, batch1, greedy, 512 generated tokens",
              "evaluation_example_ids": [row["example_id"] for row in rows], "free_replay_example_ids": replay,
              "free_replay_selection": "10/domain, lowest SHA256(hf_free_replay_v3|example_id) across all160; fixed before replay outputs, outcomes ignored",
              "schemas": {domain: schema_for_domain(domain) for domain in ("structured_records", "inventory_tracking")},
              "parser_settings": PARSER_SETTINGS,
              "array_limit_note": "Fixed library heuristic 512 for every domain, independent of all answers/cardinalities; no min/maxItems/enum/ranges supplied in schemas. 512 generation-token cap still applies.",
              "adapter_note": "Published0.11.3 integration imports removed Transformers5 module. Local MIT tokenizer/prefix adapter uses unmodified core TokenEnforcer; restores explicit parser settings after core constructor reset.",
              "prompt_token_evidence": {identifier: {"tokens": len(ids), "sha256": digest(canonical(ids))} for identifier, ids in prompts.items()},
              "cpu_checks": tests, "dependency_manifest_sha256": sha(CONTROL / "dependency_manifest.json"),
              "vendor_files": dependency_files(), "script_sha256": sha(__file__),
              "v3_script_sha256": sha(ROOT / "experiments/production_eval/synthetic_sft_v3.py"),
              "v3_manifest_sha256": sha(BASE / "experiment_manifest.json"),
              "model_files": json.loads((BASE / "models/models_manifest.json").read_text())["assets"],
              "runtime_versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "pydantic", "numpy")}}
    write_json(path, result, exclusive=True)
    print(json.dumps({"status": result["status"], "examples": len(rows), "cpu_checks": tests, "manifest_sha256": sha(path)}), flush=True)


def verify_control():
    manifest = json.loads((CONTROL / "control_manifest.json").read_text())
    checks = {"script_sha256": __file__, "v3_script_sha256": ROOT / "experiments/production_eval/synthetic_sft_v3.py",
              "v3_manifest_sha256": BASE / "experiment_manifest.json", "dependency_manifest_sha256": CONTROL / "dependency_manifest.json"}
    for key, path in checks.items():
        if sha(path) != manifest[key]:
            raise ValueError(f"Frozen control dependency changed: {key}")
    if dependency_files() != manifest["vendor_files"]:
        raise ValueError("Vendor files changed")
    for entry in manifest["model_files"]:
        if sha(entry["local_path"]) != entry["sha256"]:
            raise ValueError("Official model/tokenizer file changed")
    return manifest


def run(mode, device):
    import torch
    manifest = verify_control()
    rows = frozen_tasks()
    tokenizer, tokenizer_data = tokenizer_cpu()
    by_id = {row["example_id"]: row for row in rows}
    if mode == "smoke":
        chosen = mechanical_fixtures()
    else:
        ids = manifest["free_replay_example_ids"] if mode == "free_replay" else manifest["evaluation_example_ids"]
        chosen = [input_only(by_id[identifier]) for identifier in ids]
    destination = CONTROL / mode
    destination.mkdir(parents=True, exist_ok=False)
    constrained = mode != "free_replay"
    started = datetime.now(timezone.utc).isoformat()
    write_json(destination / "started.json", {"started_at": started, "control_manifest_sha256": sha(CONTROL / "control_manifest.json"),
                                              "device": device, "mode": mode, "count": len(chosen)}, exclusive=True)
    model = load_model(device)
    model.eval()
    model.gradient_checkpointing_disable()
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    model_evidence = {"parameter_dtype": str(next(model.parameters()).dtype),
                      "unique_parameter_count": sum(p.numel() for p in model.parameters()),
                      "config_tie_word_embeddings": model.config.tie_word_embeddings,
                      "embedding_lm_head_same_storage": embedding.data_ptr() == head.data_ptr(),
                      "embedding_lm_head_equal_values": bool(torch.equal(embedding, head)),
                      "generation_config": model.generation_config.to_dict(), "device": str(next(model.parameters()).device)}
    records = []
    start_all = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(chosen):
            prompt = prompt_tokens(tokenizer, row)
            if mode != "smoke":
                expected = manifest["prompt_token_evidence"][row["example_id"]]
                if expected != {"tokens": len(prompt), "sha256": digest(canonical(prompt))}:
                    raise ValueError("Frozen tokenized prompt changed")
            ids = torch.tensor([prompt], device=device)
            prefix = DomainPrefixFilter(tokenizer_data, row["domain"], prompt) if constrained else None
            kwargs = {"prefix_allowed_tokens_fn": prefix} if constrained else {}
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           max_new_tokens=PARAMETERS["max_new_tokens"], do_sample=False,
                                           use_cache=True, pad_token_id=tokenizer.pad_token_id, **kwargs)
            output_ids = generated[0, len(prompt):].tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            formatting = format_diagnostics(output, row["domain"])
            record = {key: value for key, value in row.items() if key != "input_text"}
            record.update(response=output, generated_tokens=len(output_ids),
                          generation_limit_hit=len(output_ids) >= PARAMETERS["max_new_tokens"] and output_ids[-1] != tokenizer.eos_token_id,
                          elapsed_seconds=time.monotonic() - start,
                          prompt_token_sha256=digest(canonical(prompt)), output_token_ids=output_ids,
                          prefix_filter_calls=prefix.calls if prefix else 0, **formatting)
            # Ground truth is first consulted after generation; never passed to
            # prompt construction, schema construction, or the decoding filter.
            if mode != "smoke":
                record["exact_match"] = json_exact_grade(output, by_id[row["example_id"]]["answer"])
            records.append(record)
            with (destination / "responses.jsonl").open("a") as stream:
                stream.write(canonical(record) + "\n")
            if (index + 1) % 10 == 0 or mode == "smoke":
                print(json.dumps({"mode": mode, "evaluated": index + 1, "of": len(chosen)}), flush=True)
    report = {"status": "completed", "mode": mode, "evaluated": len(records), "started_at": started,
              "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start_all,
              "model_evidence": model_evidence, "strict_json_valid": sum(r["strict_json_valid"] for r in records),
              "required_schema_valid": sum(r["required_schema_valid"] for r in records),
              "generation_limit_hit": sum(r["generation_limit_hit"] for r in records),
              "exact_match": sum(r.get("exact_match", False) for r in records) if mode != "smoke" else None,
              "responses_sha256": sha(destination / "responses.jsonl"),
              "control_manifest_sha256": sha(CONTROL / "control_manifest.json")}
    write_json(destination / "completed.json", report, exclusive=True)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "smoke", "free_replay", "evaluate"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run(args.mode, args.device)


if __name__ == "__main__":
    main()
