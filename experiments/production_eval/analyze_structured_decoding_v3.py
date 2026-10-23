"""Describe the exploratory schema-decoding control without changing any outputs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.structured_decoding_v3 import CONTROL, BASE, frozen_tasks, verify_control, sha
from experiments.production_eval.analyze_sft_v3 import (
    read_responses, summarize_responses, paired_family_rows, bootstrap_families,
)
from experiments.production_eval.field_diagnostics import aggregate


def replay_comparison(replay, original):
    if not replay or not set(replay) <= set(original):
        raise ValueError("Replay must be a nonempty subset of the original evaluation")
    rows = []
    for identifier, current in replay.items():
        previous = original[identifier]
        if current["input_hash"] != previous["input_hash"]:
            raise ValueError("Replay input differs from original")
        rows.append({"example_id": identifier, "domain": current["domain"], "role": current["role"],
                     "response_identical": current["response"] == previous["response"],
                     "generated_token_count_identical": current["generated_tokens"] == previous["generated_tokens"],
                     "exact_match_identical": current["exact_match"] == previous["exact_match"],
                     "required_schema_valid_identical": current["required_schema_valid"] == previous["required_schema_valid"],
                     "original_exact_match": previous["exact_match"], "replay_exact_match": current["exact_match"],
                     "original_schema_valid": previous["required_schema_valid"],
                     "replay_schema_valid": current["required_schema_valid"],
                     "original_response_sha256": previous["response_sha256"], "replay_response_sha256": current["response_sha256"]})
    keys = ("response_identical", "generated_token_count_identical", "exact_match_identical", "required_schema_valid_identical")
    return {"examples": len(rows), "counts": {key: sum(row[key] for row in rows) for key in keys},
            "original_exact_match": sum(row["original_exact_match"] for row in rows),
            "replay_exact_match": sum(row["replay_exact_match"] for row in rows),
            "original_schema_valid": sum(row["original_schema_valid"] for row in rows),
            "replay_schema_valid": sum(row["replay_schema_valid"] for row in rows), "rows": rows}


def field_rows(config, responses):
    return [{"config_id": config, "example_id": row["example_id"], "family_id": row["family_id"],
             "domain": row["domain"], "role": row["role"], "regime": row["regime"],
             "availability": "available" if row["required_schema_valid"] else "completed_invalid_format",
             **row["field_diagnostics"]} for row in responses.values()]


def main():
    manifest = verify_control()
    tasks = {row["example_id"]: row for row in frozen_tasks()}
    baseline_completion = json.loads((BASE / "baseline/completed.json").read_text())
    if baseline_completion["manifest_sha256"] != sha(BASE / "experiment_manifest.json") or baseline_completion["evaluated"] != len(tasks):
        raise ValueError("Original free baseline belongs to another experiment or is incomplete")
    baseline = read_responses(BASE / "baseline/responses.jsonl", tasks)
    evaluated = {}
    sources = {"original_free": {"responses_sha256": sha(BASE / "baseline/responses.jsonl"),
                                  "completion_sha256": sha(BASE / "baseline/completed.json")}}
    for mode in ("smoke", "free_replay", "evaluate"):
        directory = CONTROL / mode
        completed = json.loads((directory / "completed.json").read_text())
        if completed["status"] != "completed" or completed["control_manifest_sha256"] != sha(CONTROL / "control_manifest.json"):
            raise ValueError("Incomplete or different control execution")
        if completed["responses_sha256"] != sha(directory / "responses.jsonl"):
            raise ValueError("Control responses changed after completion")
        evidence = completed["model_evidence"]
        if evidence["unique_parameter_count"] != 751632384 or evidence["parameter_dtype"] != "torch.float32" or evidence["embedding_lm_head_same_storage"] or not evidence["embedding_lm_head_equal_values"]:
            raise ValueError("Control did not use the fixed original untied FP32 model")
        sources[mode] = {"responses_sha256": completed["responses_sha256"], "completion_sha256": sha(directory / "completed.json"),
                         "elapsed_seconds": completed["elapsed_seconds"], "model_evidence": evidence}
        if mode == "smoke":
            if completed["evaluated"] != 2 or completed["required_schema_valid"] != 2:
                raise ValueError("Mechanical smoke did not pass both fixtures")
            continue
        subset = {i: tasks[i] for i in manifest["free_replay_example_ids"]} if mode == "free_replay" else tasks
        evaluated[mode] = read_responses(directory / "responses.jsonl", subset)
        if completed["evaluated"] != len(evaluated[mode]):
            raise ValueError("Completion count mismatch")
        for identifier, row in evaluated[mode].items():
            if row["prompt_token_sha256"] != manifest["prompt_token_evidence"][identifier]["sha256"]:
                raise ValueError("Stored prompt-token provenance differs from freeze")
            if len(row["output_token_ids"]) != row["generated_tokens"]:
                raise ValueError("Generated token count differs from recorded IDs")
            expected_calls = 0 if mode == "free_replay" else row["generated_tokens"]
            if row["prefix_filter_calls"] != expected_calls:
                raise ValueError("Schema filter was not applied at every generated token")
    replay = replay_comparison(evaluated["free_replay"], baseline)
    constrained = evaluated["evaluate"]
    field_groups = aggregate(field_rows("original_free", baseline) + field_rows("original_constrained", constrained))
    pairs = paired_family_rows(tasks, [constrained], baseline)
    schema_as_outcome = lambda responses: {i: {**r, "exact_match": r["required_schema_valid"]} for i, r in responses.items()}
    format_pairs = paired_family_rows(tasks, [schema_as_outcome(constrained)], schema_as_outcome(baseline))
    report = {"status": "complete_exploratory_control", "created_at": datetime.now(timezone.utc).isoformat(),
              "control_manifest_sha256": sha(CONTROL / "control_manifest.json"), "source_files": sources,
              "analysis_code_sha256": sha(__file__), "original_free": summarize_responses(tasks, baseline),
              "original_constrained": summarize_responses(tasks, constrained), "free_replay": replay,
              "paired_reference_exact_match": {dict(baseline="original_free", trained="original_constrained", delta="delta")[k]: v for k, v in bootstrap_families(pairs).items()},
              "paired_reference_required_schema": {dict(baseline="original_free", trained="original_constrained", delta="delta")[k]: v for k, v in bootstrap_families(format_pairs).items()},
              "secondary_field_diagnostics": field_groups,
              "counts_by_domain_role": dict(Counter(r["domain"] + "/" + r["role"] for r in constrained.values())),
              "scope": "Post-hoc control on fixed procedural tasks, proposed after free/SFT outcomes. Static schema only, same model/checkpoint, untied FP32 weights, BF16 compute, greedy decoding, batch 1 and 512-token cap; no answer-dependent IDs, values, ranges or array sizes; no output repair. Baseline originally ran on GPU 1 and control on GPU 0; the fixed 20-case free replay checks but cannot prove stability on un-replayed examples. Format improvement is not semantic correctness. Constrained decoding also changes token trajectories; differences from SFT do not identify an additive causal decomposition into format and reasoning gains. Bootstrap resamples paired families within domain/regime and is conditional on these fixed runs."}
    (CONTROL / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Control exploratorio: decodificación con esquema JSON", "",
             "Mismo checkpoint Qwen3-0.6B que v3: 751632384 parámetros FP32 sin compartir almacenamiento entre embeddings y salida; cómputo BF16, generación greedy y límite de 512 tokens. El esquema fija campos y tipos por dominio. No recibe identificadores, valores, cantidades ni respuestas de cada ejemplo. Propuesto después de observar resultados libres/SFT; no constituye una confirmación independiente preregistrada.", "",
             "| Ejecución | Rol | Ejemplos | Exacto | Esquema válido | Límite de tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for config in ("original_free", "original_constrained"):
        for role in ("validation", "reference"):
            row = report[config][role]
            lines.append(f"| {config} | {role} | {row['examples']} | {row['exact_match']:.2%} | {row['required_schema_valid']:.2%} | {row['generation_limit_count']} |")
    n = replay["examples"]
    lines.extend(["", f"Replay libre en GPU 0: {replay['counts']['response_identical']}/{n} textos idénticos a la ejecución original en GPU 1; {replay['counts']['exact_match_identical']}/{n} decisiones de exact match iguales y {replay['counts']['required_schema_valid_identical']}/{n} decisiones de formato iguales. Los 20 ejemplos (10/dominio) se fijaron por hash antes de generar el control. Esta comprobación no demuestra equivalencia numérica en los 140 ejemplos restantes.", "",
                  "| Ejecución | Dominio, referencia | F1 de IDs | Total unidades exacto | Total valor exacto | Conteos por ubicación |", "|---|---|---:|---:|---:|---:|"])
    for row in field_groups:
        if row["role"] != "reference" or row["regime"] != "all_regimes":
            continue
        m = row["family_mean_metrics"]
        metric = "selected_id_set_f1" if row["domain"] == "structured_records" else "item_set_f1"
        percent = lambda key: f"{m[key]:.2%}" if key in m else "—"
        lines.append(f"| {row['config_id']} | {row['domain']} | {percent(metric)} | {percent('total_units_exact')} | {percent('total_value_exact')} | {percent('expected_item_location_count_accuracy')} |")
    lines.extend(["", "Los diagnósticos por campo son secundarios y mantienen todas las familias en el denominador; el formato inválido recibe cero crédito. El criterio principal sigue siendo coincidencia exacta de todos los campos con el oráculo. Los desgloses por régimen, evidencia del modelo, hashes de fuentes e intervalos por familias están en analysis.json.", "",
                  "La gramática pasó fixtures CPU con 28 elementos, campos reordenados, identificadores arbitrarios y cifras negativas. Su límite fijo de arrays es 512; la longitud efectiva también está limitada por el presupuesto de generación. No se ha aumentado ese presupuesto ni se han reparado respuestas para mejorar las puntuaciones.", "",
                  "Forzar el esquema también cambia la secuencia de tokens elegidos. La comparación con SFT permite comprobar si generar JSON válido basta para resolver estas tareas, pero no identifica una descomposición causal aditiva entre ganancias de formato y de razonamiento.", ""])
    (CONTROL / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "free_reference": report["original_free"]["reference"],
                      "constrained_reference": report["original_constrained"]["reference"], "replay_counts": replay["counts"]}), flush=True)


if __name__ == "__main__":
    main()
