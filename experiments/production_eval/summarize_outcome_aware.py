"""Readable secondary report; no selection, inference or grading is performed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


MIDDLE = {"gsm8k": 50, "bbh_boolean_expressions": 25, "structured_records": 10,
          "inventory_tracking": 10, "balanced_mixture": 40}
NAMES = {"uniform_random": "Aleatorio uniforme",
         "error_targeted_naive": "Solo fallos; media sin corregir",
         "error_hybrid_ht": "Aleatorio + fallos; pesos HT",
         "disagreement_targeted_naive": "Desacuerdos; media sin corregir",
         "disagreement_hybrid_ht": "Aleatorio + desacuerdos; pesos HT"}
REGIMES = {"random_benchmark_split": "Referencia aleatoria", "changed_template": "Otra plantilla",
           "more_complex": "Más compleja", "stationary": "Estacionaria",
           "equal_domain_reference_mixture": "Referencia con igual peso por dominio"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-analysis", type=Path, required=True)
    parser.add_argument("--outcome-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_analysis.read_text())["selector_estimation"]
    targeted = json.loads((args.outcome_dir / "selector_analysis.json").read_text())
    provenance = json.loads((args.outcome_dir / "provenance.json").read_text())
    combined = base["aggregates"] + targeted["aggregates"]
    discoveries = base["failure_discovery"] + targeted["failure_discovery"]
    lines = ["# Seleccionar fallos y estimar rendimiento son objetivos diferentes", "",
             "Análisis exploratorio adicional: sus reglas se fijaron antes de las respuestas locales principales, pero después del inicio de Gemini. No forma parte de la selección inicial de 585 corridas. Se ejecutaron 600 selecciones adicionales, sobre los mismos pools de construcción y sin usar referencias para seleccionar.", "",
             "Los fallos provienen de corregir Qwen directo con las etiquetas u oráculos del pool de construcción. El desacuerdo compara las respuestas de los dos formatos de Qwen. Por tanto, encontrar más fallos de esos mismos modelos es un resultado condicionado por la selección; la comprobación útil es cuánto se deforma el score y cómo se compara con una referencia disjunta.", "",
             "HT significa Horvitz–Thompson: cada familia se pondera según su probabilidad conocida de inclusión. La mezcla conserva probabilidad positiva para todas las familias. Sus garantías se refieren al pool finito de construcción, no a tráfico futuro.", "",
             "## Estimación del cambio entre los dos formatos de Qwen", "",
             "Objetivo: Qwen con pasos intermedios menos Qwen directo. Errores absolutos en puntos porcentuales, promediados sobre 10 semillas de adquisición; son repeticiones de muestreo sobre resultados fijos, no estudios independientes.", ""]
    exports = []
    for dataset in MIDDLE:
        rows = [r for r in combined if r["dataset"] == dataset
                and r["budget_families"] == MIDDLE[dataset]
                and r["anchor_config"] == "qwen_direct" and r["candidate_config"] == "qwen_work"
                and r["policy"] in NAMES]
        refs = sorted({key for r in rows for key in r["reference_mean_absolute_errors"]})
        lines += [f"### {dataset} · {MIDDLE[dataset]} familias", ""]
        if dataset in {"structured_records", "inventory_tracking"}:
            lines += ["**Comparación no informativa para discriminar selectores:** ambos Qwen fallan todos los casos y su diferencia es constante, cero.", ""]
        lines += ["| Selector | Error construcción | " + " | ".join(REGIMES.get(key, key) for key in refs) + " |",
                  "|---|---:|" + "---:|" * len(refs)]
        for row in sorted(rows, key=lambda r: list(NAMES).index(r["policy"])):
            values = [row["mean_construction_absolute_error"]] + [row["reference_mean_absolute_errors"].get(key) for key in refs]
            lines.append("| " + NAMES[row["policy"]] + " | " + " | ".join("desconocido" if v is None else f"{100*v:.2f}" for v in values) + " |")
            exports.append(row)
        lines.append("")
    lines += ["## Hallazgo de fallos de Qwen directo", "",
              "Un fallo de familia significa al menos una respuesta incorrecta entre todos sus descendientes. Las familias sintéticas tienen cuatro respuestas; las públicas, una. El presupuesto familiar no iguala por sí solo el número de respuestas revisadas en la mezcla.", "",
              "| Dominio | Selector | Respuestas revisadas, media | Familias fallidas por 100 seleccionadas | Prevalencia real en construcción |", "|---|---|---:|---:|---:|"]
    grouped = defaultdict(list)
    for row in discoveries:
        if row["config_id"] == "qwen_direct" and row["policy"] in NAMES and row["budget_families"] == MIDDLE[row["dataset"]]:
            grouped[(row["dataset"], row["policy"])].append(row)
    for (dataset, policy), rows in sorted(grouped.items()):
        yield_values = [r["full_selection_failing_family_yield_per_100"] for r in rows]
        truth_values = [r["full_construction_family_failure_rate"] for r in rows]
        if any(v is None for v in yield_values + truth_values):
            raise ValueError("Incomplete discovery results cannot be summarized as complete")
        lines.append(f"| {dataset} | {NAMES[policy]} | {mean(r['expected_selected_responses'] for r in rows):.1f} | {mean(yield_values):.2f} | {100*mean(truth_values):.2f}% |")
    lines += ["", "El enriquecimiento de fallos de la selección no estima prevalencia. Por ejemplo, revisar solo fallos conocidos puede producir un 100% de hallazgos aun cuando la mayoría del pool no falle.", "",
              "## Coste y alcance", "",
              "Construir las dos señales requirió disponer de respuestas de ambos formatos locales para todo el pool de construcción, además de los oráculos de corrección. Es trabajo previo a la selección; reutilizarlo en este estudio no lo hace gratuito para una aplicación. Solo fallos necesita las respuestas de Qwen directo; desacuerdo necesita ambas configuraciones. No se ha medido coste humano ni se afirma ahorro en minutos de revisión.", "",
              "La referencia booleana tiene un cambio de score distinto al de construcción incluso con división aleatoria. Los regímenes sintéticos cambian familias además de dificultad o plantilla. Ninguna de estas comprobaciones es una validación temporal de producción a 30 o 60 días.", "",
              f"[Resultados y métricas completos]({(args.outcome_dir / 'selector_analysis.json').resolve()}) · [Trazabilidad]({(args.outcome_dir / 'provenance.json').resolve()})", ""]
    (args.outcome_dir / "report.md").write_text("\n".join(lines))
    (args.outcome_dir / "middle_budget_comparison.json").write_text(json.dumps(exports, indent=2) + "\n")
    print(json.dumps({"report": str((args.outcome_dir / "report.md").resolve()), "comparison_rows": len(exports)}))


if __name__ == "__main__":
    main()
