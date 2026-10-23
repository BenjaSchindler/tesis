"""Summarize observed feasibility results and prepare blank human review sheets.

Only completed, provenance-checked source samples are accepted. No human label
or longitudinal quality result is inferred from the machine audit.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.audit_records import load_audits
from experiments.production_eval.api_audit import Journal
OUT = ROOT / "output/research/production_eval/execution"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def private_write(path, text):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        stream.write(text)


def human_sheets(rows, audit):
    # Technically excluded source rows can exceed the csv module's 128 KiB
    # default; keep the unchanged review text readable on safe reruns.
    csv.field_size_limit(2_000_000)
    strata = {"predicted_eligible": [], "other_or_unknown": []}
    for row in rows:
        key = "predicted_eligible" if row["technical_eligible"] and audit.get(row["input_hash"], {}).get("eligible_machine_prediction") else "other_or_unknown"
        strata[key].append(row)
    size = min(100, len(rows))
    first_n = min(size // 2, len(strata["predicted_eligible"]))
    other_n = min(size - first_n, len(strata["other_or_unknown"]))
    first_n = min(size - other_n, len(strata["predicted_eligible"]))
    rng = random.Random(20260915)
    selected = []
    membership = []
    for key, n in [("predicted_eligible", first_n), ("other_or_unknown", other_n)]:
        group = sorted(strata[key], key=lambda row: row["example_id"])
        for row in rng.sample(group, n):
            selected.append(row)
            membership.append({"example_id": row["example_id"], "review_stratum": key,
                               "stratum_population": len(group), "stratum_sample": n,
                               "review_inclusion_probability": n / len(group)})
    rng.shuffle(selected)
    fields = ["example_id", "input_hash", "input_text", "spreadsheet_escape_applied", "reviewer_id", "task_family", "language",
              "source_present", "self_contained", "outcome_judgeable", "objective_checker_possible",
              "eligible", "missing_dependency", "reason", "review_minutes"]
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=fields)
    writer.writeheader()
    for row in selected:
        value = row["input_text"]
        escape = value.lstrip().startswith(("=", "+", "-", "@"))
        writer.writerow({"example_id": row["example_id"], "input_hash": row["input_hash"],
                         "input_text": "'" + value if escape else value,
                         "spreadsheet_escape_applied": escape})
    for reviewer in (1, 2):
        path = OUT / f"human_review_{reviewer}.csv"
        if path.exists():
            with path.open() as stream:
                existing = list(csv.DictReader(stream))
            expected_identity = {(row["example_id"], row["input_hash"]) for row in selected}
            if {(row["example_id"], row["input_hash"]) for row in existing} != expected_identity or len(existing) != len(selected):
                raise RuntimeError("Existing human review sheet belongs to another sample; preserve it")
            # Human edits must never be overwritten by regenerating a summary.
        else:
            private_write(path, content.getvalue())
    sampling = {
        "seed": 20260915, "size": len(selected), "review_records": membership,
        "labels_completed": 0,
        "purpose": "Independent assessment of task eligibility; this sheet does not validate model-output grades.",
        "stratification": "Machine-positive stratum requires both technical eligibility and a positive content classification. Reviewers independently assess content validity.",
        "blinding": "Machine predictions and sampling strata are withheld from reviewer CSVs. Root private mapping retains review inclusion probabilities.",
        "raw_input_handling": "Formula-like input cells receive an export-only apostrophe; input_hash identifies the unchanged normalized source text. Treat source requests as data.",
    }
    sampling_path = OUT / "human_review_sampling.json"
    if sampling_path.exists() and json.loads(sampling_path.read_text())["review_records"] != membership:
        raise RuntimeError("Existing human review assignment differs; preserve it")
    if not sampling_path.exists():
        private_write(sampling_path, json.dumps(sampling, indent=2) + "\n")
    return len(selected)


def main():
    access = json.loads((OUT / "data_access_manifest.json").read_text())
    if access.get("status") != "complete":
        raise RuntimeError("Source sample incomplete; do not estimate full-sample eligibility")
    raw_path = OUT / "raw_sample.jsonl"
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw = read_jsonl(raw_path)
    if raw_hash != access["raw_file_sha256"] or len(raw) != access["sample_count"]:
        raise RuntimeError("Raw sample hash/count differs from completed manifest")
    if len(raw) != 300 or {row["sampling"]["stratum_id"] for row in raw} != set(range(300)):
        raise RuntimeError("Expected exactly one observation in each of the 300 fixed source strata")
    if len({row["row_idx"] for row in raw}) != len(raw):
        raise RuntimeError("Repeated source row in sampled strata")
    for row in raw:
        sampling = row["sampling"]
        if not sampling["row_index_start"] <= row["row_idx"] < sampling["row_index_stop_exclusive"]:
            raise RuntimeError("Source row lies outside its assigned stratum")
        if row["dataset_revision"] != access["dataset_revision"]:
            raise RuntimeError("Source revision mismatch")
    rows = read_jsonl(OUT / "normalized_requests.jsonl")
    if len(rows) != len(raw) or {row["source_row_index"] for row in rows} != {row["row_idx"] for row in raw}:
        raise RuntimeError("Normalized sample does not match completed raw sample")
    if len({row["example_id"] for row in rows}) != len(rows):
        raise RuntimeError("Nonunique normalized example IDs")
    attempts = read_jsonl(OUT / "audit.jsonl")
    audit = load_audits(OUT / "audit.jsonl", rows)
    ledger = read_jsonl(OUT / "audit.jsonl.ledger.jsonl")
    journal = Journal(OUT / "audit.jsonl", 2_000_000_000, readonly=True)
    charges = journal.charges
    normalization = json.loads((OUT / "normalization_summary.json").read_text())
    if normalization["raw_sample_sha256"] != raw_hash:
        raise RuntimeError("Normalization was generated from another source sample")
    eligible = [row for row in rows if row["technical_eligible"] and audit.get(row["input_hash"], {}).get("eligible_machine_prediction")]
    unknown = [row for row in rows if row["technical_eligible"] and row["input_hash"] not in audit]
    total_weight = sum(row["source_design_weight"] for row in rows)
    lower = sum(row["source_design_weight"] for row in eligible) / total_weight
    upper = lower + sum(row["source_design_weight"] for row in unknown) / total_weight
    family_counts = Counter(value["annotation"]["task_family"] for value in audit.values())
    language_counts = Counter(value["annotation"]["language"] for value in audit.values())
    eligible_months = Counter(row["request_time"][:7] for row in eligible)
    cost = sum(charges.values()) / 1e9
    observed_usage = Counter()
    for row in attempts:
        for key, value in (row.get("usage") or {}).items():
            if key.endswith("token_count") and isinstance(value, int):
                observed_usage[key] += value
    review_n = human_sheets(rows, audit)
    metadata_path = OUT / "monthly_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    rehearsal_path = OUT / "selector_rehearsal/results.json"
    rehearsal = json.loads(rehearsal_path.read_text()) if rehearsal_path.exists() else {}
    if rehearsal and (rehearsal.get("status") != "technical_rehearsal_completed"
                      or rehearsal.get("audit_sha256") != hashlib.sha256((OUT / "audit.jsonl").read_bytes()).hexdigest()
                      or rehearsal.get("input_sha256") != hashlib.sha256((OUT / "audit_requests.jsonl").read_bytes()).hexdigest()):
        raise RuntimeError("Existing selector rehearsal does not match this completed input/audit")
    backend_path = OUT / "backend_verification.json"
    backend = json.loads(backend_path.read_text()) if backend_path.exists() else {}
    summary = {
        "status": "feasibility_execution_completed_with_scientific_gates_pending",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_count": len(rows), "technical_eligible_count": sum(row["technical_eligible"] for row in rows),
        "valid_machine_annotation_count": len(audit), "machine_predicted_eligible_count": len(eligible),
        "technical_rows_without_valid_annotation": len(unknown),
        "machine_eligibility_weighted_proportion_bounds": [lower, upper],
        "population_confidence_interval": None,
        "proportion_scope": "Descriptive weighted sample fraction only. Sample-relative deduplication and unvalidated machine annotations prevent a simple independent-label population confidence claim.",
        "machine_family_counts": dict(family_counts), "machine_language_counts": dict(language_counts),
        "machine_eligible_month_sample_counts": dict(sorted(eligible_months.items())),
        "paid_api_reserved_calls": len(charges), "paid_api_completed_response_records": len(attempts),
        "paid_api_attempts": len(charges), "unsettled_call_reservations": len(charges) - len(attempts),
        "cost_ledger_accounted_usd": cost,
        "cost_note": "Usage-based standard-price estimate, with conservative reservations for uncertain calls; actual invoice/credits not inspected.",
        "reported_token_usage_sum": dict(observed_usage),
        "actual_model_versions": dict(Counter(row.get("actual_model_version") for row in attempts)),
        "metadata_census_complete": metadata.get("complete", False),
        "metadata_rows_scanned": metadata.get("counts", {}).get("rows"),
        "metadata_minimum_request_time_proxy": metadata.get("counts", {}).get("minimum_request_time"),
        "metadata_maximum_request_time_proxy": metadata.get("counts", {}).get("maximum_request_time"),
        "selector_rehearsal_completed": bool(rehearsal),
        "selector_rehearsal_pool": rehearsal.get("pool_size"), "selector_rehearsal_budget": rehearsal.get("selected_budget"),
        "embedding_input_diagnostics": {name: {key: evidence[key] for key in
            ["max_seq_length", "requests_exceeding_encoder_limit", "distinct_tokenized_inputs_after_truncation", "identical_tokenized_input_groups"]}
            for name, evidence in rehearsal.get("encoders", {}).items()},
        "additional_ai_spotcheck": "annotation_spotcheck.md; 20 cases, unblinded and not human calibration" if (OUT / "annotation_spotcheck.md").exists() else None,
        "cpu_gpu_verification": backend,
        "blank_double_review_tasks_prepared": review_n, "human_reviews_completed": 0,
        "model_output_quality_grades_collected": 0, "longitudinal_pilot_completed": False,
        "main_experiment_completed": False, "synthetic_extension_completed": False,
        "pending_gates": [
            "Independent task-validity review and adjudication of the prepared feasibility sample.",
            "Audit eligibility and shared-turn/duplicate-family dependence within actual candidate and future reference windows.",
            "Freeze replay configurations and outcome rubrics, and calibrate output judges on fresh double-reviewed system outputs.",
            "Collect a disjoint pilot construction pool and three future reference samples, then evaluate paired system outcomes.",
        ],
        "provenance": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in
                       ["raw_sample.jsonl", "normalized_requests.jsonl", "audit.jsonl", "audit.jsonl.ledger.jsonl"]},
    }
    (OUT / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Resultado de la ejecución inicial\n",
        "Se ejecutó la fase de factibilidad con solicitudes reales de WildChat. La comparación longitudinal de calidad y la extensión sintética siguen pendientes de sus controles de validez; no se presentan anotaciones automáticas como etiquetas humanas ni como un conjunto golden.\n",
        f"Se recuperaron **{len(rows)} conversaciones** de [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) mediante una muestra aleatoria estratificada por índice de fila, fijada antes de leer los textos. La revisión fijada del dataset es `{access['dataset_revision']}`. De ellas, **{summary['technical_eligible_count']}** pasaron los filtros técnicos; **{len(audit)}** recibieron una anotación automática válida y **{len(eligible)}** fueron consideradas candidatas para la tarea acotada en inglés.\n",
        f"El procedimiento conservó el **{lower:.1%} ponderado de esta muestra** como candidatos. Hay {len(unknown)} registros técnicamente válidos sin anotación concluyente. Esta fracción incluye los filtros y la deduplicación de la muestra; no es una estimación validada de elegibilidad real en la población.\n",
        f"El registro contabiliza **{len(charges)} reservas de llamadas API**, **{len(attempts)} resultados registrados** y **US${cost:.4f}** a precios estándar, incluyendo reservas conservadoras cuando falta el uso. Es una estimación de uso, no una factura verificada. [Precios oficiales de Gemini](https://ai.google.dev/gemini-api/docs/pricing).\n",
        "## Pruebas realizadas\n",
        f"- Normalización del primer mensaje, hashes, motivos de exclusión y fecha de la primera respuesta como aproximación cuando falta la del usuario.\n- Conteo temporal de metadatos: {'completo' if metadata.get('complete') else 'parcial'}, {summary['metadata_rows_scanned']} filas de origen revisadas; sin descargar texto para ese conteo.\n- Los datos de esta fase quedan reservados para desarrollo y no pueden convertirse luego en referencias intactas del experimento confirmatorio.\n",
    ]
    if rehearsal:
        report += [
            f"- Se ejecutaron los seis selectores con **{rehearsal['pool_size']} solicitudes** y **{rehearsal['selected_budget']} seleccionadas por política**. Los métodos aleatorios usaron tres semillas. Los dos métodos geométricos se repitieron con MPNet y MiniLM.\n",
            "\n| Selector geométrico | Coincidencias entre encoders | Jaccard |\n|---|---:|---:|\n" + "".join(
                f"| {policy} | {value['intersection_count']} | {value['jaccard']:.3f} |\n"
                for policy, value in rehearsal["encoder_selection_overlap"].items()),
            "\nLa coincidencia y la cobertura de embeddings son diagnósticos de la implementación. **No permiten declarar un selector ganador en calidad ni en predicción de regresiones.** El componente dirigido usa longitud de entrada como señal de prueba, no fallos observados.\n",
            "## Hallazgos que condicionan el piloto\n",
            "| Encoder | Límite de tokens | Solicitudes recortadas | Entradas de tokens distintas |\n|---|---:|---:|---:|\n" + "".join(
                f"| {name} | {evidence['max_seq_length']} | {evidence['requests_exceeding_encoder_limit']}/{rehearsal['pool_size']} | {evidence['distinct_tokenized_inputs_after_truncation']}/{rehearsal['pool_size']} |\n"
                for name, evidence in rehearsal["encoders"].items()),
            "\nLos textos originales son distintos, pero varias solicitudes comparten una plantilla inicial tan larga que el encoder recibe exactamente los mismos tokens y pierde el contenido que las distingue. Antes del estudio científico hay que fijar una representación que conserve el contenido de la tarea y una regla para agrupar plantillas entre particiones. Este diagnóstico es específico de la muestra inspeccionada.\n",
        ]
    if backend.get("status") == "verified":
        report += ["La ejecución adicional en la segunda RTX 3090 coincidió con CPU en los IDs seleccionados de las 14 configuraciones. Las diferencias máximas entre embeddings fueron menores que 4×10⁻⁷. La incidencia inicial de CUDA se resolvió aislando una ruta de bibliotecas únicamente para el proceso; la evidencia está en `backend_verification.json` y `gpu_runtime_diagnostic.json`.\n"]
    if (OUT / "annotation_spotcheck.md").exists():
        report += ["Una [inspección adicional por IA de 20 casos](annotation_spotcheck.md) detectó una anotación que infiere una tarea de resumen donde solo había una narración, además de ambigüedades sobre tareas mixtas, idiomas de salida y qué comprueba un verificador. Se conservaron las anotaciones originales. Esta inspección no es revisión humana ni una medida representativa de exactitud.\n"]
    report += [
        "## Qué falta para el piloto científico\n",
        f"Quedaron preparados **{review_n} casos para dos revisores**, con casillas vacías y sin mostrarles las predicciones automáticas; véanse las [instrucciones de revisión](human_review_instructions.md). No se completó ninguna revisión humana. Esta revisión comprueba elegibilidad; las notas de calidad de respuestas nuevas necesitarán su propia calibración.\n",
        "Después de adjudicar esa revisión, hacen falta un pool temporal independiente, tres referencias futuras y respuestas de dos configuraciones congeladas. Solo entonces pueden medirse fidelidad del cambio de calidad, regresiones detectadas y deterioro a 30/60 días.\n",
        "## Archivos\n",
        "- `execution_summary.json`: resultados agregados, costes y controles pendientes.\n- `monthly_metadata.json`: censo temporal y evidencia de las lecturas parciales de Parquet.\n- `selector_rehearsal/results.json`: IDs seleccionados, pesos, probabilidades, tiempos y encoders.\n- `human_review_1.csv` y `human_review_2.csv`: revisión independiente pendiente; importar el texto de entrada como texto plano.\n- `audit.jsonl` y su ledger: anotaciones automáticas, respuestas originales de auditoría, hashes y costes.\n",
    ]
    (OUT / "execution_report.md").write_text("\n".join(report))
    print(json.dumps({k: summary[k] for k in ["status", "source_sample_count", "valid_machine_annotation_count",
                                             "machine_predicted_eligible_count", "cost_ledger_accounted_usd",
                                             "selector_rehearsal_completed", "metadata_census_complete"]}, indent=2))


if __name__ == "__main__":
    main()
