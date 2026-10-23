"""Hash every saved asset of completed v3 runs without changing the trainer."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/research/production_eval/experiments_v3"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    manifest_path = BASE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    trainer_path = ROOT / 'experiments/production_eval/synthetic_sft_v3.py'
    source = json.loads((BASE / 'source_provenance.json').read_text())
    trainer_modified = sha(trainer_path) != source['source_hashes'][str(trainer_path.relative_to(ROOT))]
    if trainer_modified:
        raise ValueError('Original frozen trainer changed')
    report = {"status": "completed_checkpoints_verified_partial", "verified_at": datetime.now(timezone.utc).isoformat(),
              "experiment_manifest_sha256": manifest_hash, "planned_runs": len(manifest["runs"]),
              "required_files": ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
              "runs": [], "pending_runs": [], "trainer_modified": trainer_modified,
              "architecture_variant": "Untied embedding and lm_head; 751632384 FP32 master parameters",
              "verifier_sha256": sha(Path(__file__))}
    for run in manifest["runs"]:
        directory = BASE / "runs" / run["run_id"]
        completion_path = directory / "completed.json"
        if not completion_path.exists():
            report["pending_runs"].append(run["run_id"])
            continue
        completion = json.loads(completion_path.read_text())
        if completion["manifest_sha256"] != manifest_hash or completion["run"] != run:
            raise ValueError("Completed run no longer matches frozen plan")
        checkpoint = directory / "checkpoint"
        if any(not (checkpoint / filename).is_file() for filename in report["required_files"]):
            raise ValueError("Completed checkpoint lacks config/tokenizer assets")
        assets = []
        tensors, parameter_count, tensor_names = 0, 0, set()
        for path in sorted(checkpoint.rglob("*")):
            if not path.is_file():
                continue
            record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}
            if path.suffix == ".safetensors":
                with path.open("rb") as stream:
                    header_size = int.from_bytes(stream.read(8), "little")
                    if not 0 < header_size < 10_000_000:
                        raise ValueError("Unexpected safetensors header size")
                    header = json.loads(stream.read(header_size))
                tensor_headers = [value for key, value in header.items() if key != "__metadata__"]
                tensor_names.update(key for key in header if key != '__metadata__')
                parameter_count += sum(math.prod(value['shape']) for value in tensor_headers)
                dtypes = sorted({value["dtype"] for value in tensor_headers})
                if dtypes != ["F32"]:
                    raise ValueError("Checkpoint is not the measured FP32 master model")
                record.update(stored_tensor_count=len(tensor_headers), stored_dtypes=dtypes)
                tensors += len(tensor_headers)
            assets.append(record)
        if tensors == 0:
            raise ValueError("Completed checkpoint has no safetensors weights")
        if parameter_count != 751632384 or not {'model.embed_tokens.weight', 'lm_head.weight'} <= tensor_names:
            raise ValueError('Completed checkpoint differs from the frozen untied variant')
        report["runs"].append({"run_id": run["run_id"], "training_example_ids": run["training_example_ids"],
                               "unique_training_families": run["unique_training_families"],
                               "completed_metadata_sha256": sha(completion_path),
                               "training_steps_sha256": sha(directory / "training_steps.jsonl"),
                               "responses_sha256": sha(directory / "responses.jsonl"), "checkpoint_assets": assets,
                               "stored_parameter_count": parameter_count,
                               "execution_audit": completion.get('execution_audit'),
                               "resumed_response_prefix_count": completion.get('resumed_response_prefix_count', 0)})
    if not report["pending_runs"]:
        report["status"] = "all_completed_checkpoints_verified"
    destination = BASE / "checkpoint_verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "verified_runs": len(report["runs"]),
                      "pending_runs": len(report["pending_runs"]), "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
