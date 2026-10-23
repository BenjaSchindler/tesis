"""Paired, descriptive analysis of the two frozen random SFT grammar controls."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.trained_structured_decoding_v3 import (
    BASE, CONTROL, FOLLOWUP, RUNS, frozen_tasks, sha, verify_followup,
)
from experiments.production_eval.analyze_sft_v3 import (
    read_responses, summarize_responses, paired_family_rows, bootstrap_families,
)
from experiments.production_eval.analyze_structured_decoding_v3 import field_rows, replay_comparison
from experiments.production_eval.field_diagnostics import aggregate


def paired(tasks, treatment, comparator):
    result = bootstrap_families(paired_family_rows(tasks, treatment, comparator))
    return {"comparator": result["baseline"], "treatment": result["trained"], "delta": result["delta"]}


def main():
    manifest = verify_followup()
    tasks = {row["example_id"]: row for row in frozen_tasks()}
    base_completion = json.loads((CONTROL / "evaluate/completed.json").read_text())
    if base_completion["status"] != "completed" or base_completion["evaluated"] != 160 or base_completion["control_manifest_sha256"] != manifest["control_manifest_sha256"] or base_completion["responses_sha256"] != sha(CONTROL / "evaluate/responses.jsonl"):
        raise ValueError("Original constrained control is incomplete or changed")
    sources = {"original_free": BASE / "baseline/responses.jsonl",
               "original_constrained": CONTROL / "evaluate/responses.jsonl"}
    completions = {}
    for run in RUNS:
        directory = FOLLOWUP / run
        completed = json.loads((directory / "completed.json").read_text())
        if completed["status"] != "completed" or completed["evaluated"] != 160 or completed["run_id"] != run:
            raise ValueError("A prespecified trained control is incomplete")
        if completed["followup_manifest_sha256"] != sha(FOLLOWUP / "followup_manifest.json") or completed["responses_sha256"] != sha(directory / "responses.jsonl"):
            raise ValueError("Trained control source hashes changed")
        evidence = completed["model_evidence"]
        if evidence["unique_parameter_count"] != 751632384 or evidence["parameter_dtype"] != "torch.float32" or evidence["embedding_lm_head_same_storage"]:
            raise ValueError("Trained control architecture differs")
        sources[run + "_free"] = BASE / "runs" / run / "responses.jsonl"
        sources[run + "_constrained"] = directory / "responses.jsonl"
        completions[run] = {"path": str((directory / "completed.json").relative_to(ROOT)), "sha256": sha(directory / "completed.json")}
    response_sets = {name: read_responses(path, tasks) for name, path in sources.items()}
    for name, responses in response_sets.items():
        if not name.endswith("_constrained"):
            continue
        for identifier, row in responses.items():
            if row["prompt_token_sha256"] != manifest["prompt_token_evidence"][identifier]["sha256"] or row["prefix_filter_calls"] != row["generated_tokens"] or len(row["output_token_ids"]) != row["generated_tokens"]:
                raise ValueError("Constrained generation provenance does not match the frozen recipe")
    fields = aggregate([r for name, responses in response_sets.items() for r in field_rows(name, responses)])
    contrasts = []
    for run in RUNS:
        for comparator in ("original_constrained", run + "_free"):
            treatment = run + "_constrained"
            contrasts.append({"treatment": treatment, "comparator": comparator,
                              "reference_exact_match": paired(tasks, [response_sets[treatment]], response_sets[comparator])})
    both = [response_sets[run + "_constrained"] for run in RUNS]
    report = {"status": "complete_exploratory_trained_controls", "created_at": datetime.now(timezone.utc).isoformat(),
              "followup_manifest_sha256": sha(FOLLOWUP / "followup_manifest.json"),
              "source_responses": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in sources.items()},
              "completion_files": completions, "original_constrained_completion_sha256": sha(CONTROL / "evaluate/completed.json"),
              "analysis_code_sha256": sha(__file__),
              "configs": {name: summarize_responses(tasks, responses) for name, responses in response_sets.items()},
              "paired_decoding_response_agreement": {run: replay_comparison(response_sets[run + "_constrained"], response_sets[run + "_free"]) for run in RUNS},
              "response_agreement_scope": "Secondary diagnostic added after the first trained constrained control completed. Compares free versus constrained outputs, not a pure free-decoding hardware replay; no new generation or metric/selection change.",
              "analysis_dependency_sha256": {name: sha(ROOT / "experiments/production_eval" / name) for name in ("analyze_structured_decoding_v3.py", "analyze_sft_v3.py", "field_diagnostics.py")},
              "paired_reference_contrasts": contrasts,
              "two_seed_mean_constrained_vs_original_constrained": paired(tasks, both, response_sets["original_constrained"]),
              "secondary_field_diagnostics": fields,
              "scope": "Both previously fixed random 64-example SFT seeds, without selecting a winner. Same 160 inputs, fixed grammar, FP32 untied architecture, BF16 inference, greedy decoding and 512-token cap. Exploratory follow-up after free outcomes. All constrained runs use GPU 0; original free runs used GPU 1. Only the initial model had a fixed 20-case free replay verifying identical outputs across GPUs; trained checkpoint free replays were not run. Contrasts involving constrained versus free decoding are descriptive; token-trajectory changes preclude additive causal attribution to format alone. Family-bootstrap intervals condition on the two observed trained checkpoints, not a population of optimization seeds. Procedural tasks do not establish production or temporal validity."}
    (FOLLOWUP / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Dos semillas de SFT con el mismo esquema JSON", "",
             "Se evaluaron ambos checkpoints random de 64 ejemplos, sin seleccionar el mejor. El modelo original y los dos adaptados recibieron la misma gramática por dominio y los mismos 160 prompts, con generación greedy y límite de 512 tokens. Se conservaron los pesos FP32 sin compartir embeddings y salida (751632384 parámetros) y el cómputo BF16.", "",
             "| Configuración | Rol | Exacto | Esquema válido | Límite de tokens |", "|---|---|---:|---:|---:|"]
    for name, summary in report["configs"].items():
        for role in ("validation", "reference"):
            row = summary[role]
            lines.append(f"| {name} | {role} | {row['exact_match']:.2%} | {row['required_schema_valid']:.2%} | {row['generation_limit_count']} |")
    result = report["two_seed_mean_constrained_vs_original_constrained"]
    delta = result["delta"]
    interval = delta["ci95_conditional_family_bootstrap"]
    lines.extend(["", f"Media de ambas semillas con gramática: {result['treatment']['estimate']:.2%} exact match de referencia; modelo original con gramática: {result['comparator']['estimate']:.2%}. Diferencia: {100*delta['estimate']:.2f} puntos porcentuales; intervalo por familias condicionado a estos checkpoints: [{100*interval[0]:.2f}, {100*interval[1]:.2f}].", "",
                  "Este seguimiento es exploratorio y mantiene las dos semillas originales. El intervalo reagrupa las dos variantes de cada familia y remuestrea dentro de dominio/régimen; no estima la variación de nuevas semillas de entrenamiento. El JSON conserva todos los contrastes, campos secundarios, regímenes y hashes.", "",
                  "Todos los controles con gramática usaron GPU 0; las evaluaciones libres originales usaron GPU 1. El replay libre de 20 ejemplos confirmó textos idénticos solo para el checkpoint inicial. No se ejecutaron replays libres de los checkpoints entrenados.", "",
                  "Forzar JSON modifica también la trayectoria de tokens. Las diferencias entre generación libre y restringida no pueden restarse para aislar un efecto causal exclusivo del formato. Los resultados describen tareas procedurales controladas; no prueban validez en producción ni estabilidad temporal.", ""])
    for run, agreement in report["paired_decoding_response_agreement"].items():
        lines.append(f"Coincidencia textual libre/restringida para {run}: {agreement['counts']['response_identical']}/{agreement['examples']}. Es un diagnóstico secundario de ambos decodificadores, no un replay libre adicional.")
    lines.append("")
    (FOLLOWUP / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "reference": {name: summary["reference"] for name, summary in report["configs"].items()},
                      "two_seed_constrained_delta": delta}), flush=True)


if __name__ == "__main__":
    main()
