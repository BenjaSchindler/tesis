"""Create a compact, source-linked report from the verified SFT analysis JSON."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/research/production_eval/experiments_v3'
LABELS = {'random_base': 'Aleatorio', 'coverage_base': 'Cobertura', 'centroid_base': 'Centroide',
          'coverage_random_mix': 'Cobertura + aleatorio', 'coverage_mixed_wording': 'Cobertura, redacción mixta',
          'coverage_paired_wording': 'Cobertura, parejas de redacción'}


def render(analysis, source):
    baseline = analysis['baseline']['reference']
    complete = analysis['status'] == 'complete'
    lines = ['# Fine-tuning con datos procedurales y selección geométrica', '',
             f"Estado: **{len(analysis['completed_run_ids'])}/{analysis['planned_runs']} entrenamientos completos**. Análisis generado {analysis['generated_at']}.", '',
             'Se comparan seis selecciones de 64 ejemplos, con dos semillas de optimización por selección. Cada entrenamiento recibe 256 presentaciones y 32 actualizaciones. La referencia contiene 120 entradas de 60 familias nuevas; cada familia conserva sus dos redacciones.', '',
             '**Arquitectura medida:** Qwen3-0.6B cargado con embedding y cabeza de salida independientes: 751,632,384 parámetros FP32. No equivale al fine-tuning habitual con pesos compartidos. El control separado de esa diferencia está fijado en `tied_control/manifest.json`.', '',
             '## Exactitud y formato en referencia', '',
             '| Selección | Semilla | Respuestas exactas | Esquema válido | Válidas pero incorrectas |',
             '|---|---:|---:|---:|---:|',
             f"| Modelo inicial | — | {baseline['exact_match'] * 100:.2f}% | {baseline['required_schema_valid'] * 100:.2f}% | {baseline['schema_valid_but_wrong_count']}/120 |"]
    for run in analysis['runs']:
        result = run['outcomes']['reference']
        lines.append(f"| {LABELS[run['arm']]} | {run['training_seed']} | {result['exact_match'] * 100:.2f}% | {result['required_schema_valid'] * 100:.2f}% | {result['schema_valid_but_wrong_count']}/120 |")
    lines += ['', 'La métrica principal exige que todo el objeto sea correcto. El esquema válido mide únicamente formato y tipos; puede mejorar aunque fallen los cálculos. Las respuestas inválidas se mantienen en el denominador.', '',
              '## Comparación con selección aleatoria', '',
              'Solo se muestran brazos con ambas semillas terminadas. Los intervalos son un bootstrap emparejado de familias dentro de dominio y régimen, condicionado a estas dos ejecuciones; no describen la variabilidad entre todos los entrenamientos posibles. Esta comparación explícita se agregó después de observar los tres primeros runs y se reporta como análisis secundario.', '',
              '| Selección | Exactitud media | Diferencia vs. aleatorio | Intervalo 95% condicionado |',
              '|---|---:|---:|---:|']
    for comparison in analysis.get('posthoc_paired_arm_comparisons', {}).get('comparisons', []):
        metrics = comparison['metrics']['exact_match']['overall']
        lo, hi = metrics['delta']['ci95_conditional_family_bootstrap']
        lines.append(f"| {LABELS[comparison['arm']]} | {metrics['trained']['estimate'] * 100:.2f}% | {metrics['delta']['estimate'] * 100:+.2f} pp | [{lo * 100:+.2f}, {hi * 100:+.2f}] pp |")
    lines += ['', 'No se elige un ganador a partir de estos intervalos ni se cambian selecciones o hiperparámetros tras observar la referencia. La selección de adquisición es única por brazo; las dos semillas varían optimización y orden de datos.', '',
              '## Cambios de distribución', '',
              '| Selección | Dominio | Estacionario | Plantilla modificada | Mayor complejidad |', '|---|---|---:|---:|---:|']
    grouped = defaultdict(lambda: defaultdict(list))
    for run in analysis['runs']:
        for row in run['outcomes']['reference_slices']:
            grouped[(run['arm'], row['domain'])][row['regime']].append(row['exact_match'])
    complete_arms = {arm['arm'] for arm in analysis['arms']}
    for arm in LABELS:
        if arm not in complete_arms:
            continue
        for domain in ['structured_records', 'inventory_tracking']:
            values = [sum(grouped[(arm, domain)][regime]) / len(grouped[(arm, domain)][regime]) for regime in ['stationary', 'changed_template', 'more_complex']]
            label = 'Filtrado y agregación' if domain == 'structured_records' else 'Inventario'
            lines.append(f"| {LABELS[arm]} | {label} | " + ' | '.join(f'{v * 100:.2f}%' for v in values) + ' |')
    lines += ['', 'Son cambios procedurales controlados, sin evidencia todavía sobre tráfico real ni deriva temporal de producción. Las variantes comparten oráculo y familia; no se cuentan como observaciones independientes.', '',
              '## Trazabilidad', '',
              f'- [Análisis y resultados por familia]({source.resolve()})',
              f'- [Plan original congelado]({BASE / "experiment_manifest.json"})',
              f'- [Checkpoint y hashes]({BASE / "checkpoint_verification.json"})',
              f'- [Auditoría de pesos independientes]({BASE / "weight_tying_audit.json"})',
              f'- [Recuperación y recursos]({BASE / "resource_plan.md"})', '',
              'El cuarto run se recuperó desde su checkpoint terminado y conservó las 76 respuestas anteriores a la interrupción. No se volvió a entrenar ni se sustituyeron salidas. Su tiempo final sincronizado y memoria máxima no se habían persistido y permanecen desconocidos; los pasos, tokens y pesos sí se verificaron.']
    if complete:
        lines += ['', f'![Exactitud y formato por semilla]({BASE / "sft_reference_outcomes.png"})']
    return '\n'.join(lines) + '\n'


def plot(analysis, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    x = np.arange(len(LABELS))
    for axis, metric, title in zip(axes, ['exact_match', 'required_schema_valid'], ['Respuesta completa correcta', 'Esquema JSON válido']):
        for seed, offset, color in [(17, -.14, '#246b8e'), (29, .14, '#c76b35')]:
            scores = [next(r for r in analysis['runs'] if r['arm'] == arm and r['training_seed'] == seed)['outcomes']['reference'][metric] * 100 for arm in LABELS]
            axis.bar(x + offset, scores, width=.27, color=color, label=f'Semilla {seed}')
        axis.axhline(analysis['baseline']['reference'][metric] * 100, color='#555555', linestyle='--', label='Modelo inicial', linewidth=1)
        axis.set_title(title, fontsize=11)
        axis.set_ylim(0, 105 if metric == 'required_schema_valid' else max(25, max(r['outcomes']['reference'][metric] * 100 for r in analysis['runs']) * 1.2))
        axis.set_ylabel('% de 120 entradas de referencia')
        axis.set_xticks(x, ['Aleatorio', 'Cobertura', 'Centroide', 'Cobertura\n+ aleatorio', 'Redacción\nmixta', 'Parejas de\nredacción'], rotation=25, ha='right', fontsize=8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Selección de 64 ejemplos · 32 actualizaciones · pesos de embedding/cabeza independientes', fontsize=12)
    fig.tight_layout()
    fig.savefig(destination / 'sft_reference_outcomes.png', dpi=180)
    fig.savefig(destination / 'sft_reference_outcomes.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    output = BASE / ('report.md' if analysis['status'] == 'complete' else 'report_partial.md')
    output.write_text(render(analysis, args.analysis))
    if analysis['status'] == 'complete':
        plot(analysis, BASE)
    print(json.dumps({'report': str(output), 'completed_runs': len(analysis['completed_run_ids'])}))


if __name__ == '__main__':
    main()
