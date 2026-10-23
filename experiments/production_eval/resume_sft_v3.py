"""Recover an interrupted frozen untied SFT cohort without rerunning outputs.

The original trainer stays byte-for-byte unchanged. Complete checkpoints resume
only a validated response prefix. New runs use its unmodified train/eval code.
No optimizer restart is attempted for partial training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval import synthetic_sft_v3 as sft

BASE = sft.BASE
EXPECTED_UNTIED_PARAMETERS = 751632384


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_prefix(path, evaluation):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete JSONL tail; preserve and inspect before recovery')
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) > len(evaluation):
        raise ValueError('More responses than frozen evaluation rows')
    for response, row in zip(records, evaluation):
        for key in ('example_id', 'family_id', 'input_hash', 'domain', 'role', 'regime', 'variant'):
            if response[key] != row[key]:
                raise ValueError('Responses are not the exact frozen evaluation prefix')
        grade = sft.json_exact_grade(response['response'], row['answer'])
        if type(response['exact_match']) is not bool or response['exact_match'] != grade:
            raise ValueError('Recorded grade disagrees with the original exact grader')
    return records


def validate_steps(steps, examples):
    p = sft.PARAMETERS
    if len(steps) != p['optimizer_steps_per_arm']:
        raise ValueError('Training is incomplete; no optimizer-state recovery is available')
    for i, row in enumerate(steps, 1):
        if row['step'] != i or row['cumulative_example_presentations'] != i * p['gradient_accumulation']:
            raise ValueError('Training step sequence is inconsistent')
        expected_lr = p['learning_rate'] * min(1., i / p['warmup_steps'])
        if row['learning_rate'] != expected_lr:
            raise ValueError('Training learning-rate schedule differs from frozen recipe')
    final = steps[-1]
    for field, token_key in [('cumulative_input_tokens', 'prompt_ids'), ('cumulative_target_tokens', 'answer_ids')]:
        if final[field] != p['epochs'] * sum(len(example[token_key]) for example in examples):
            raise ValueError('Training token totals differ from the frozen examples')
    return final


def checkpoint_assets(directory):
    checkpoint = directory / 'checkpoint'
    required = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
    if any(not (checkpoint / filename).is_file() for filename in required):
        raise ValueError('Incomplete saved checkpoint assets')
    assets, tensor_count, parameter_count, names = [], 0, 0, set()
    for path in sorted(checkpoint.iterdir()):
        if not path.is_file():
            continue
        assets.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        if path.suffix != '.safetensors':
            continue
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            if not 0 < size < 10_000_000:
                raise ValueError('Invalid safetensors header length')
            header = json.loads(stream.read(size))
        data_length = path.stat().st_size - 8 - size
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if value['dtype'] != 'F32' or name in names:
                raise ValueError('Checkpoint dtype or tensor uniqueness is invalid')
            names.add(name)
            count = 1
            for dim in value['shape']:
                count *= dim
            start, end = value['data_offsets']
            if end - start != count * 4 or not 0 <= start <= end <= data_length:
                raise ValueError('Checkpoint tensor bytes are incomplete')
            tensor_count += 1
            parameter_count += count
    if parameter_count != EXPECTED_UNTIED_PARAMETERS:
        raise ValueError('Checkpoint does not contain the frozen untied parameter count')
    if not {'model.embed_tokens.weight', 'lm_head.weight'} <= names:
        raise ValueError('Checkpoint lacks separate embedding/head tensors')
    return {'assets': assets, 'tensor_count': tensor_count, 'parameter_count': parameter_count}


def assert_untied(model):
    embed, head = model.model.embed_tokens.weight, model.lm_head.weight
    if embed.data_ptr() == head.data_ptr() or sum(p.numel() for p in model.parameters()) != EXPECTED_UNTIED_PARAMETERS:
        raise ValueError('Loaded architecture differs from the frozen untied variant')
    if str(next(model.parameters()).dtype) != 'torch.float32':
        raise ValueError('Loaded model is not FP32 master weights')


def load_checkpoint(directory, device):
    # Reuse the identical loader, with only its already-saved weight path changed.
    original = sft.MODEL
    try:
        sft.MODEL = directory / 'checkpoint'
        model = sft.load_model(device)
    finally:
        sft.MODEL = original
    assert_untied(model)
    return model


def recovered_report(directory, steps, examples):
    last = validate_steps(steps, examples)
    persisted = directory / 'training_report.json'
    if persisted.exists():
        report = json.loads(persisted.read_text())
        if report['records'] != steps or report['optimizer_steps'] != 32 or report['example_presentations'] != 256:
            raise ValueError('Persisted training report disagrees with complete step log')
        return report
    return {'optimizer_steps': len(steps), 'example_presentations': last['cumulative_example_presentations'],
            'elapsed_seconds': None, 'last_logged_step_elapsed_seconds': last['elapsed_seconds'],
            'input_tokens': last['cumulative_input_tokens'], 'target_tokens': last['cumulative_target_tokens'],
            'peak_cuda_allocated_bytes': None, 'parameter_dtype': 'torch.float32',
            'optimizer_exp_avg_dtype': None, 'optimizer_exp_avg_sq_dtype': None, 'records': steps,
            'recovery_note': 'Original process ended during evaluation after checkpoint save. Schedule and token totals verified from complete step log. Final synchronized duration, peak memory and measured optimizer dtypes were not persisted; they remain null.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--run', default='all')
    args = parser.parse_args()
    sft.setup()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    lock = (BASE / 'cohort_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = BASE / 'experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha(manifest_path)
    provenance = json.loads((BASE / 'source_provenance.json').read_text())
    original_trainer = ROOT / 'experiments/production_eval/synthetic_sft_v3.py'
    if sha(original_trainer) != provenance['source_hashes'][str(original_trainer.relative_to(ROOT))]:
        raise ValueError('Original trainer source changed')
    if manifest['parameters'] != sft.PARAMETERS or sha(ROOT / manifest['tasks_file']) != manifest['tasks_sha256']:
        raise ValueError('Frozen recipe or task data changed')
    tasks = [json.loads(line) for line in (ROOT / manifest['tasks_file']).read_text().splitlines()]
    evaluation = [row for row in tasks if row['role'] in {'validation', 'reference'}]
    by_id = {row['example_id']: row for row in tasks}
    if not (BASE / 'smoke/results.json').exists():
        raise ValueError('Original mechanical smoke is absent')
    baseline = json.loads((BASE / 'baseline/completed.json').read_text())
    if baseline['manifest_sha256'] != manifest_hash or len(read_prefix(BASE / 'baseline/responses.jsonl', evaluation)) != len(evaluation):
        raise ValueError('Baseline is incomplete or belongs to a different plan')
    selected = manifest['runs'] if args.run == 'all' else [r for r in manifest['runs'] if r['run_id'] == args.run]
    if not selected:
        raise ValueError('Unknown frozen run ID')
    tokenizer = AutoTokenizer.from_pretrained(str(sft.MODEL), local_files_only=True, trust_remote_code=False)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    recovery = BASE / 'recovery' / stamp
    recovery.mkdir(parents=True)
    audit_path = recovery / 'events.jsonl'
    def event(value):
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), **value}
        with audit_path.open('a') as stream:
            stream.write(sft.canonical(row) + '\n')
        print(json.dumps(row), flush=True)
    sft.write_json(recovery / 'started.json', {'manifest_sha256': manifest_hash, 'wrapper_sha256': sha(Path(__file__)),
                                             'original_trainer_sha256': sha(original_trainer), 'device': args.device,
                                             'selected_runs': [r['run_id'] for r in selected]}, exclusive=True)
    for run in selected:
        directory = BASE / 'runs' / run['run_id']
        completion = directory / 'completed.json'
        if completion.exists():
            saved = json.loads(completion.read_text())
            if saved['manifest_sha256'] != manifest_hash or saved['run'] != run or len(read_prefix(directory / 'responses.jsonl', evaluation)) != len(evaluation):
                raise ValueError('Existing completed run failed validation')
            event({'run_id': run['run_id'], 'action': 'skip_completed', 'completed_sha256': sha(completion)})
            continue
        examples = [sft.token_example(tokenizer, by_id[i]) for i in run['training_example_ids']]
        prefix_count, prefix_bytes, prefix_hash = 0, 0, None
        if directory.exists():
            started = json.loads((directory / 'started.json').read_text())
            if started['manifest_sha256'] != manifest_hash or started['run'] != run:
                raise ValueError('Interrupted run differs from the frozen plan')
            steps = [json.loads(line) for line in (directory / 'training_steps.jsonl').read_text().splitlines()]
            report = recovered_report(directory, steps, examples)
            checkpoint = checkpoint_assets(directory)
            prefix = read_prefix(directory / 'responses.jsonl', evaluation)
            prefix_count = len(prefix)
            if (directory / 'responses.jsonl').exists():
                snapshot = recovery / (run['run_id'] + '.responses.prefix.jsonl')
                shutil.copyfile(directory / 'responses.jsonl', snapshot)
                prefix_bytes, prefix_hash = snapshot.stat().st_size, sha(snapshot)
            event({'run_id': run['run_id'], 'action': 'resume_saved_checkpoint', 'prefix_count': prefix_count,
                   'prefix_bytes': prefix_bytes, 'prefix_sha256': prefix_hash, 'checkpoint': checkpoint,
                   'training_steps_sha256': sha(directory / 'training_steps.jsonl')})
            model = load_checkpoint(directory, args.device)
        else:
            directory.mkdir(parents=True)
            sft.write_json(directory / 'started.json', {'started_at': datetime.now(timezone.utc).isoformat(),
                           'manifest_sha256': manifest_hash, 'run': run,
                           'execution_wrapper': str(Path(__file__).relative_to(ROOT)),
                           'execution_wrapper_sha256': sha(Path(__file__))}, exclusive=True)
            torch.manual_seed(run['training_seed'])
            model = sft.load_model(args.device)
            assert_untied(model)
            event({'run_id': run['run_id'], 'action': 'train_frozen_recipe'})
            report = sft.train_loop(model, examples, run['training_seed'], args.device,
                                    output=directory / 'training_steps.jsonl')
            validate_steps(report['records'], examples)
            sft.write_json(directory / 'training_report.json', report, exclusive=True)
            model.save_pretrained(str(directory / 'checkpoint'), safe_serialization=True)
            tokenizer.save_pretrained(str(directory / 'checkpoint'))
            sft.write_json(directory / 'checkpoint_saved.json', checkpoint_assets(directory), exclusive=True)
        if prefix_count < len(evaluation):
            sft.evaluate(model, tokenizer, evaluation[prefix_count:], args.device, directory / 'responses.jsonl')
        if prefix_bytes:
            with (directory / 'responses.jsonl').open('rb') as stream:
                if hashlib.sha256(stream.read(prefix_bytes)).hexdigest() != prefix_hash:
                    raise ValueError('Existing response prefix was changed during recovery')
        records = read_prefix(directory / 'responses.jsonl', evaluation)
        if len(records) != len(evaluation):
            raise ValueError('Evaluation did not complete')
        sft.write_json(completion, {'status': 'completed', 'completed_at': datetime.now(timezone.utc).isoformat(),
                       'run': run, 'training': report, 'evaluated': len(records), 'manifest_sha256': manifest_hash,
                       'execution_audit': str(audit_path.relative_to(ROOT)),
                       'resumed_response_prefix_count': prefix_count}, exclusive=True)
        del model
        torch.cuda.empty_cache()
        event({'run_id': run['run_id'], 'action': 'completed', 'evaluated': len(records),
               'preserved_prefix_sha256': prefix_hash, 'responses_sha256': sha(directory / 'responses.jsonl')})
    event({'action': 'cohort_selected_runs_complete'})


if __name__ == '__main__':
    main()
