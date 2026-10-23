"""Separately frozen tied-weight sensitivity control for the original SFT study.

Uses random_base's same64 examples and seeds17/29, with the original recipe.
Only embedding/head sharing changes. Run only after the12 untied runs finish.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval import synthetic_sft_v3 as sft
from experiments.production_eval.resume_sft_v3 import sha, read_prefix, validate_steps

BASE = sft.BASE / 'tied_control'
TIED_PARAMETERS = 596049920


def force_equal_original_tying(model):
    import torch
    embed, head = model.model.embed_tokens.weight, model.lm_head.weight
    if not torch.equal(embed, head):
        raise ValueError('Refuse to tie unequal tensors: this would change the initial function')
    model.config.tie_word_embeddings = True
    model.tie_weights()
    # from_pretrained removed this runtime mapping when both original copies
    # existed. Restore the actual now-shared mapping before serialization.
    model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)
    assert_tied(model)
    return model


def assert_tied(model):
    if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise ValueError('Embedding/head storage is not shared')
    if sum(p.numel() for p in model.parameters()) != TIED_PARAMETERS:
        raise ValueError('Unexpected tied parameter count')
    if str(next(model.parameters()).dtype) != 'torch.float32':
        raise ValueError('Tied control requires FP32 master parameters')


def load_saved(path, device):
    original = sft.MODEL
    try:
        sft.MODEL = path
        model = sft.load_model(device)
    finally:
        sft.MODEL = original
    # No manual repair on reload: serialization must itself express sharing.
    assert_tied(model)
    return model


def probe(model, examples, device):
    import torch
    values = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            ids, labels = sft.masked_tensors(example, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                mask = labels[:, 1:] != -100
                logits = model.lm_head(hidden[:, :-1, :][mask]).float()
                loss = sft.response_only_loss(model, ids, labels)
            values.append((logits.detach().cpu(), float(loss)))
    return values


def compare_probes(before, after):
    import torch
    if len(before) != len(after):
        raise ValueError('Different fixture counts')
    rows = []
    for (old_logits, old_loss), (new_logits, new_loss) in zip(before, after):
        identical = torch.equal(old_logits, new_logits) and old_loss == new_loss
        rows.append({'identical_logits': torch.equal(old_logits, new_logits),
                     'max_abs_logit_difference': float((old_logits - new_logits).abs().max()),
                     'loss_before': old_loss, 'loss_after': new_loss, 'loss_abs_difference': abs(old_loss - new_loss)})
        if not identical:
            raise ValueError('Saved/retied model failed exact fixture forward/loss equivalence')
    return rows


def prepare():
    original_path = sft.BASE / 'experiment_manifest.json'
    original = json.loads(original_path.read_text())
    if original['parameters'] != sft.PARAMETERS:
        raise ValueError('Original frozen recipe changed')
    runs = [r for r in original['runs'] if r['arm'] == 'random_base']
    if [r['training_seed'] for r in runs] != [17, 29] or any(r['training_examples'] != 64 for r in runs):
        raise ValueError('Missing matched random runs')
    record = {'status': 'frozen_before_control_training_or_evaluation',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'control': 'Explicitly tied embedding/head sensitivity; exploratory follow-up to observed untied architecture',
              'original_manifest_sha256': sha(original_path), 'tasks_file': original['tasks_file'],
              'tasks_sha256': original['tasks_sha256'], 'parameters': original['parameters'], 'runs': runs,
              'architecture': {'tie_word_embeddings': True, 'unique_trainable_parameters': TIED_PARAMETERS,
                               'original_untied_parameters': 751632384},
              'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                               Path(sft.__file__), ROOT / 'experiments/production_eval/resume_sft_v3.py']},
              'fixtures': {'source': 'First two original independent arithmetic mechanical fixtures',
                           'ids': [r['example_id'] for r in sft.fixture_rows()[:2]],
                           'initial_and_checkpoint_reload_tolerance': 'Exact logits and scalar response-only loss equality'},
              'evaluation': 'Same160 fixed validation/reference inputs, greedy free decoding,512 new tokens, no repair',
              'analysis': 'Paired by optimization seed and reference family against the original untied random runs. No architecture or data-selection winner claim from two training seeds.',
              'observations_before_freeze': {'original_completed_runs': [r['run_id'] for r in original['runs'] if (sft.BASE / 'runs' / r['run_id'] / 'completed.json').exists()],
                                            'scope': 'Original baseline and partial untied cohort outcomes already observed. This control is not independently preregistered; original data, selectors and recipe remain unchanged.'},
              'baseline': 'Original forward function matches on independent fixtures before any optimizer; original baseline reused descriptively, no new baseline selected',
              'resource_policy': 'GPU1 after all12 original runs; no new dependencies, APIs or dataset/model downloads'}
    sft.write_json(BASE / 'manifest.json', record, exclusive=True)
    print(json.dumps({'status': record['status'], 'manifest_sha256': sha(BASE / 'manifest.json')}), flush=True)


def run(device):
    sft.setup()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    manifest_path = BASE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    original_path = sft.BASE / 'experiment_manifest.json'
    original = json.loads(original_path.read_text())
    if sha(original_path) != manifest['original_manifest_sha256'] or manifest['parameters'] != sft.PARAMETERS:
        raise ValueError('Original frozen manifest changed')
    for path, expected in manifest['source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Control source changed after freezing')
    if any(not (sft.BASE / 'runs' / r['run_id'] / 'completed.json').exists() for r in original['runs']):
        raise ValueError('Original untied cohort must finish before this control starts')
    lock = (sft.BASE / 'cohort_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(ROOT / manifest['tasks_file']) != manifest['tasks_sha256']:
        raise ValueError('Frozen tasks changed')
    tasks = [json.loads(line) for line in (ROOT / manifest['tasks_file']).read_text().splitlines()]
    by_id = {r['example_id']: r for r in tasks}
    evaluation = [r for r in tasks if r['role'] in {'reference', 'validation'}]
    tokenizer = AutoTokenizer.from_pretrained(str(sft.MODEL), local_files_only=True, trust_remote_code=False)
    fixtures = [r for r in sft.fixture_rows() if r['example_id'] in manifest['fixtures']['ids']]
    if {r['input_hash'] for r in fixtures} & {r['input_hash'] for r in tasks}:
        raise ValueError('Mechanical fixtures overlap experiment inputs')
    token_fixtures = [sft.token_example(tokenizer, r) for r in fixtures]
    smoke_path = BASE / 'initial_tying_smoke.json'
    if not smoke_path.exists():
        model = sft.load_model(device)
        before = probe(model, token_fixtures, device)
        force_equal_original_tying(model)
        after = probe(model, token_fixtures, device)
        results = compare_probes(before, after)
        sft.write_json(smoke_path, {'status': 'exact_fixture_forward_and_loss_equivalence',
                       'manifest_sha256': sha(manifest_path), 'fixtures': results,
                       'unique_parameters': sum(p.numel() for p in model.parameters()),
                       'shared_storage': True}, exclusive=True)
        del model, before, after
        torch.cuda.empty_cache()
    for planned in manifest['runs']:
        directory = BASE / 'runs' / planned['run_id']
        completion_path = directory / 'completed.json'
        if completion_path.exists():
            completion = json.loads(completion_path.read_text())
            if completion['manifest_sha256'] != sha(manifest_path) or len(read_prefix(directory / 'responses.jsonl', evaluation)) != len(evaluation):
                raise ValueError('Completed tied run failed validation')
            continue
        examples = [sft.token_example(tokenizer, by_id[i]) for i in planned['training_example_ids']]
        if directory.exists():
            started = json.loads((directory / 'started.json').read_text())
            if started['manifest_sha256'] != sha(manifest_path) or started['run'] != planned:
                raise ValueError('Interrupted tied control differs from frozen plan')
            report = json.loads((directory / 'training_report.json').read_text())
            validate_steps(report['records'], examples)
            saved = json.loads((directory / 'checkpoint_verified.json').read_text())
            if sha(directory / 'checkpoint/model.safetensors') != saved['weights_sha256']:
                raise ValueError('Tied checkpoint changed after verification')
            model = load_saved(directory / 'checkpoint', device)
            prefix = read_prefix(directory / 'responses.jsonl', evaluation)
        else:
            directory.mkdir(parents=True)
            sft.write_json(directory / 'started.json', {'manifest_sha256': sha(manifest_path), 'run': planned,
                           'started_at': datetime.now(timezone.utc).isoformat()}, exclusive=True)
            torch.manual_seed(planned['training_seed'])
            model = force_equal_original_tying(sft.load_model(device))
            assert_tied(model)
            report = sft.train_loop(model, examples, planned['training_seed'], device,
                                    output=directory / 'training_steps.jsonl')
            validate_steps(report['records'], examples)
            sft.write_json(directory / 'training_report.json', report, exclusive=True)
            before = probe(model, token_fixtures, device)
            model.save_pretrained(str(directory / 'checkpoint'), safe_serialization=True)
            tokenizer.save_pretrained(str(directory / 'checkpoint'))
            del model
            torch.cuda.empty_cache()
            model = load_saved(directory / 'checkpoint', device)
            after = probe(model, token_fixtures, device)
            results = compare_probes(before, after)
            sft.write_json(directory / 'checkpoint_verified.json', {'weights_sha256': sha(directory / 'checkpoint/model.safetensors'),
                           'fixtures': results, 'shared_storage_after_reload': True,
                           'unique_parameters': sum(p.numel() for p in model.parameters())}, exclusive=True)
            del before, after
            prefix = []
        if len(prefix) < len(evaluation):
            sft.evaluate(model, tokenizer, evaluation[len(prefix):], device, directory / 'responses.jsonl')
        responses = read_prefix(directory / 'responses.jsonl', evaluation)
        if len(responses) != len(evaluation):
            raise ValueError('Incomplete tied-control evaluation')
        sft.write_json(completion_path, {'status': 'completed', 'run': planned, 'training': report,
                       'manifest_sha256': sha(manifest_path), 'evaluated': len(responses),
                       'completed_at': datetime.now(timezone.utc).isoformat()}, exclusive=True)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({'completed_tied_run': planned['run_id'], 'evaluated': len(responses)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else run(args.device)


if __name__ == '__main__':
    main()
