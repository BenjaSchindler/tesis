"""Compare the separately frozen tied control with matched untied random runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.analyze_sft_v3 import (
    read_responses, summarize_responses, paired_arm_rows, bootstrap_families, sha,
)
from experiments.production_eval.tied_sft_control_v3 import BASE, TIED_PARAMETERS


def main():
    manifest_path = BASE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    original_base = BASE.parent
    if sha(original_base / 'experiment_manifest.json') != manifest['original_manifest_sha256']:
        raise ValueError('Original experiment manifest changed')
    if sha(ROOT / manifest['tasks_file']) != manifest['tasks_sha256']:
        raise ValueError('Frozen task data changed')
    smoke = json.loads((BASE / 'initial_tying_smoke.json').read_text())
    if smoke['manifest_sha256'] != sha(manifest_path) or smoke['unique_parameters'] != TIED_PARAMETERS or not smoke['shared_storage']:
        raise ValueError('Initial tying smoke failed validation')
    all_tasks = [json.loads(line) for line in (ROOT / manifest['tasks_file']).read_text().splitlines()]
    tasks = {row['example_id']: row for row in all_tasks if row['role'] in {'reference', 'validation'}}
    tied, untied, runs = {}, {}, []
    for planned in manifest['runs']:
        directory = BASE / 'runs' / planned['run_id']
        completed = json.loads((directory / 'completed.json').read_text())
        if completed['manifest_sha256'] != sha(manifest_path) or completed['run'] != planned:
            raise ValueError('Tied run differs from frozen control')
        checkpoint = json.loads((directory / 'checkpoint_verified.json').read_text())
        if sha(directory / 'checkpoint/model.safetensors') != checkpoint['weights_sha256'] or not checkpoint['shared_storage_after_reload'] or checkpoint['unique_parameters'] != TIED_PARAMETERS:
            raise ValueError('Tied saved checkpoint failed validation')
        original_run = original_base / 'runs' / planned['run_id']
        original_completed = json.loads((original_run / 'completed.json').read_text())
        if original_completed['manifest_sha256'] != manifest['original_manifest_sha256'] or original_completed['run'] != planned:
            raise ValueError('Untied comparator differs from original frozen selection')
        for result in (completed, original_completed):
            if result['training']['optimizer_steps'] != 32 or result['training']['example_presentations'] != 256:
                raise ValueError('Training quantities differ')
        seed = planned['training_seed']
        tied[seed] = read_responses(directory / 'responses.jsonl', tasks)
        untied[seed] = read_responses(original_run / 'responses.jsonl', tasks)
        runs.append({'seed': seed, 'tied': summarize_responses(tasks, tied[seed]),
                     'untied': summarize_responses(tasks, untied[seed]),
                     'tied_checkpoint_sha256': checkpoint['weights_sha256'],
                     'tied_responses_sha256': sha(directory / 'responses.jsonl'),
                     'untied_responses_sha256': sha(original_run / 'responses.jsonl'),
                     'tied_training': {k: v for k, v in completed['training'].items() if k != 'records'}})
    comparisons = {}
    for metric in ('exact_match', 'required_schema_valid'):
        rows = paired_arm_rows(tasks, tied, untied, metric)
        comparisons[metric] = {'overall': bootstrap_families(rows),
                               'strata': {stratum: bootstrap_families([r for r in rows if r['stratum'] == stratum]) for stratum in sorted({r['stratum'] for r in rows})}}
    report = {'status': 'complete', 'generated_at': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': sha(manifest_path), 'analysis_sha256': sha(Path(__file__)), 'runs': runs,
              'paired_reference': comparisons,
              'interpretation': 'Exploratory architecture sensitivity. Same fixed64 random training examples, optimizer schedule and seeds;596049920 tied versus751632384 untied parameters. Family intervals condition on these two fitted seeds and do not estimate training-population uncertainty.',
              'smoke': smoke, 'primary_study_changed': False}
    output = BASE / 'analysis.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Control de pesos compartidos', '',
             'Este control cambia únicamente si embedding y cabeza de salida comparten sus parámetros. Ambos modelos reciben los mismos 64 ejemplos aleatorios y las mismas semillas y actualizaciones. Es una sensibilidad exploratoria posterior a detectar que el cargador dejaba los pesos separados.', '',
             '| Semilla | Exactitud, separados | Exactitud, compartidos | Formato, separados | Formato, compartidos |',
             '|---:|---:|---:|---:|---:|']
    for run in runs:
        u, t = run['untied']['reference'], run['tied']['reference']
        lines.append(f"| {run['seed']} | {u['exact_match'] * 100:.2f}% | {t['exact_match'] * 100:.2f}% | {u['required_schema_valid'] * 100:.2f}% | {t['required_schema_valid'] * 100:.2f}% |")
    delta = comparisons['exact_match']['overall']['delta']
    lo, hi = delta['ci95_conditional_family_bootstrap']
    lines += ['', f"Diferencia media de exactitud, compartidos menos separados: **{delta['estimate'] * 100:+.2f} pp**, con intervalo 95% condicionado a estas ejecuciones de [{lo * 100:+.2f}, {hi * 100:+.2f}] pp.", '',
              'La unidad de remuestreo es la familia, conservando ambas redacciones y promediando las semillas emparejadas. Dos semillas no permiten caracterizar toda la variación del entrenamiento. No se seleccionan parámetros ni checkpoints por esta referencia.', '',
              'La prueba inicial verificó igualdad exacta de logits y pérdida sobre fixtures independientes antes de entrenar. Cada checkpoint compartido se recargó y volvió a verificar, preservando 596,049,920 parámetros únicos y el mismo almacenamiento de embedding/cabeza.', '',
              f'[Resultados y hashes]({output}) · [Manifiesto del control]({manifest_path})']
    (BASE / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'analysis': str(output), 'delta_exact_pp': delta['estimate'] * 100}))


if __name__ == '__main__':
    main()
