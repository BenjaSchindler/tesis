"""Time/memory check on new procedural fixtures, excluded from every v3 split."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval import synthetic_sft_v3 as sft


def main():
    import torch
    from transformers import AutoTokenizer
    sft.setup()
    torch.set_num_threads(8)
    device = "cuda:1"
    rows = []
    for domain in sft.DOMAINS:
        builder = sft.build_records if domain == "structured_records" else sft.build_inventory
        for index in range(4):
            family = sft.digest(f"resource_probe_v3|2026092199|{domain}|{index}")
            problem, _, _ = builder(sft.derived_seed(family, "data"), sft.derived_seed(family, "entities"), "stationary")
            primary, secondary = sft.ORACLES[domain]
            answer = primary(problem)
            if answer != secondary(problem):
                raise ValueError("Resource fixture oracles disagree")
            text = sft.render(domain, problem, False, "stationary", sft.derived_seed(family, "template"))
            rows.append({"example_id": family, "family_id": family, "input_text": text,
                         "input_hash": sft.digest(text), "answer": sft.canonical(answer),
                         "role": "resource_fixture", "domain": domain, "regime": "stationary", "variant": "base"})
    main_rows = [json.loads(line) for line in (sft.BASE / "tasks.jsonl").read_text().splitlines()]
    if {row["input_hash"] for row in rows} & {row["input_hash"] for row in main_rows}:
        raise ValueError("Resource fixtures overlap experiment inputs")
    path = sft.BASE / "resource_probe"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "results.json").exists():
        raise RuntimeError("Resource probe already completed")
    tokenizer = AutoTokenizer.from_pretrained(str(sft.MODEL), local_files_only=True, trust_remote_code=False)
    examples = [sft.token_example(tokenizer, row) for row in rows]
    model = sft.load_model(device)
    report = sft.train_loop(model, examples, 9919, device, smoke_steps=3)
    generation = sft.evaluate(model, tokenizer, [rows[0], rows[4]], device, path / "responses.jsonl")
    result = {"status": "resource_probe_completed", "completed_at": datetime.now(timezone.utc).isoformat(),
              "main_examples_used": 0, "fixtures": [{"example_id": row["example_id"], "input_hash": row["input_hash"]} for row in rows],
              "fixture_prompt_tokens": [len(row["prompt_ids"]) for row in examples],
              "fixture_target_tokens": [len(row["answer_ids"]) for row in examples], "training": report,
              "generation": [{"generated_tokens": row["generated_tokens"], "elapsed_seconds": row["elapsed_seconds"]} for row in generation],
              "interpretation": "One short compute check with realistic procedural lengths, after three fixture training steps. Timing only; no main validation/reference outcomes or effectiveness estimate. Checkpoint discarded."}
    sft.write_json(path / "results.json", result, exclusive=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
