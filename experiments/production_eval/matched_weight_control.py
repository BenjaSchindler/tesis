"""Post-hoc estimator ablation on identical, already selected hybrid families.

No new selection or inference. Exact design moments condition on the complete
observed finite pool and fixed stratified quotas; reference scores are fixed
comparison targets, not populations to which design unbiasedness extends.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.production_eval.analyze_outcomes_v2 import (
    DEFAULT_CONFIGS, analyze_frozen_selections, family_scores, grade_response,
    index_responses, load_jsonl, role, validate_records,
)
from experiments.production_eval.outcome_aware_selection import risk_flags


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/research/production_eval/experiments_v2"
MIDDLE = {"gsm8k": 50, "bbh_boolean_expressions": 25, "structured_records": 10,
          "inventory_tracking": 10, "balanced_mixture": 40}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_published_sources(base):
    """Bind the reused selections/analysis to their already published hashes."""
    completion = base / "completed_primary_manifest.json"
    saved = json.loads(completion.read_text())
    checked = []
    for name in ("selected_families.json", "selector_analysis.json"):
        path = base / "outcome_aware" / name
        entries = [entry for entry in saved["files"] if Path(entry["path"]).name == name]
        if len(entries) != 1 or entries[0]["sha256"] != sha(path):
            raise ValueError("Previously completed source hash mismatch: " + name)
        checked.append({"path": str(path), "sha256": sha(path)})
    return {"completion_manifest_sha256": sha(completion), "verified_files": checked}


def stratified_moments(values, groups, quotas, target, weighted):
    """Exact expectation and variance of independent SRS within fixed cells."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or not math.isfinite(target):
        raise ValueError("Complete finite values and target required")
    if len(groups) != len(quotas) or not groups:
        raise ValueError("Invalid stratified design")
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(values))):
        raise ValueError("Cells must partition the complete population exactly")
    if any(type(q) is not int or not 0 < q <= len(g) for q, g in zip(quotas, groups)):
        raise ValueError("Every nonempty cell requires positive bounded quota")
    budget = sum(quotas)
    expectation, variance = 0.0, 0.0
    for group, quota in zip(groups, quotas):
        cell = values[group]
        size = len(cell)
        coefficient = size / len(values) if weighted else quota / budget
        expectation += coefficient * float(cell.mean())
        # A census or singleton cell has no design variance.
        cell_variance = float(cell.var(ddof=1)) if size > 1 else 0.0
        variance += coefficient ** 2 * (1 - quota / size) * cell_variance / quota
    bias = expectation - target
    return {"expectation": expectation, "bias_to_target": bias,
            "variance": variance, "mse_to_target": variance + bias ** 2,
            "rmse_to_target": math.sqrt(variance + bias ** 2)}


def validate_design(run, flags):
    ids = run["construction_family_ids"]
    selected = run["selected_family_ids"]
    if ids != sorted(set(ids)) or len(selected) != len(set(selected)) or not set(selected) <= set(ids):
        raise ValueError("Invalid family identities")
    if len(selected) != run["budget"]:
        raise ValueError("Selected family budget changed")
    if set(flags) != set(ids):
        raise ValueError("Risk flags must match the complete pool")
    cells = run["selection_metadata"]["cells"]
    groups, quotas, risks = [], [], []
    for cell in cells:
        key = cell["key"]
        if len(key) != 2 or key[0] != "all" or type(key[1]) is not bool:
            raise ValueError("Expected frozen two-cell risk design")
        risk = key[1]
        if risk in risks:
            raise ValueError("Duplicate risk cell")
        risks.append(risk)
        group = [i for i, family in enumerate(ids) if flags[family] == risk]
        quota = cell["sample_count"]
        if len(group) != cell["population_count"] or not group or type(quota) is not int or not 0 < quota <= len(group):
            raise ValueError("Frozen cell counts disagree with observed construction signal")
        if sum(flags[family] == risk for family in selected) != quota:
            raise ValueError("Selection does not obey frozen quota")
        groups.append(group)
        quotas.append(quota)
    if sorted(i for group in groups for i in group) != list(range(len(ids))):
        raise ValueError("Risk cells do not cover the population")
    probabilities = np.zeros(len(ids))
    for group, quota in zip(groups, quotas):
        probabilities[group] = quota / len(group)
    if not np.allclose(probabilities, run["population_inclusion_probabilities"], rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen probabilities disagree with fixed quotas")
    expected = [1 / (len(ids) * probabilities[ids.index(family)]) for family in selected]
    if not np.allclose(expected, run["sample_weights"], rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen weights disagree with known inclusion probabilities")
    return groups, quotas


def naive_copy(run):
    result = copy.deepcopy(run)
    result["policy"] = run["policy"].replace("_hybrid_ht", "_hybrid_same_sample_naive")
    result["sample_weights"] = [1 / run["budget"]] * run["budget"]
    result["population_inclusion_probabilities"] = None
    result["estimator"] = "unweighted_mean_of_identical_hybrid_sample"
    # Original probabilities remain in the paired source run, not misrepresented
    # as the weights used by this deliberately uncorrected estimator.
    return result


def analyze(manifest, records, grades):
    hybrids = [r for r in manifest["runs"] if r["policy"].endswith("_hybrid_ht")]
    if not hybrids:
        raise ValueError("No frozen hybrid samples")
    paired_manifest = {**manifest, "runs": [r for original in hybrids for r in (original, naive_copy(original))]}
    result = analyze_frozen_selections(paired_manifest, records, grades, DEFAULT_CONFIGS)
    originals = {}
    for run in hybrids:
        key = (run["dataset"], run["budget"], run["risk_signal"])
        originals.setdefault(key, []).append(run)
    estimates = result["paired_run_results"]
    moments = []
    for key, runs in sorted(originals.items()):
        prototype = runs[0]
        ids = prototype["construction_family_ids"]
        rows = [r for r in records if role(r) == "construction" and r["family_id"] in set(ids)]
        scoped = {(c, r["example_id"]): grades[(c, r["example_id"])]
                  for r in rows for c in ("qwen_direct", "qwen_work")}
        flags = risk_flags(rows, scoped)[prototype["risk_signal"]]
        groups, quotas = validate_design(prototype, flags)
        for run in runs[1:]:
            if run["construction_family_ids"] != ids or validate_design(run, flags) != (groups, quotas):
                raise ValueError("Design changed across acquisition seeds")
        for anchor, candidate in itertools.combinations(DEFAULT_CONFIGS, 2):
            a, b = (family_scores(rows, grades, config) for config in (anchor, candidate))
            if set(a) != set(ids) or set(b) != set(ids):
                raise ValueError("Unknown family outcomes cannot enter exact moments")
            values = [b[family] - a[family] for family in ids]
            empirical = next(r for r in estimates if r["dataset"] == key[0] and r["budget_families"] == key[1]
                             and r["policy"] == prototype["policy"] and r["seed"] == prototype["seed"]
                             and r["anchor_config"] == anchor and r["candidate_config"] == candidate)
            targets = [("construction", mean(values))] + [(r["regime"], r["full_reference_delta"]) for r in empirical["references"]]
            for target_name, target in targets:
                if target is None:
                    raise ValueError("Reference comparison requires complete fixed outcomes")
                moments.append({"dataset": key[0], "budget_families": key[1], "risk_signal": key[2],
                                "anchor_config": anchor, "candidate_config": candidate,
                                "target": target_name, "target_delta": target,
                                "constant_construction_delta": len(set(values)) == 1,
                                "cell_population_counts": [len(g) for g in groups], "cell_sample_counts": quotas,
                                "ht": stratified_moments(values, groups, quotas, target, True),
                                "naive": stratified_moments(values, groups, quotas, target, False),
                                "srs": stratified_moments(values, [list(range(len(ids)))], [key[1]], target, True)})
    # Both estimators must have identical discoveries and work counts.
    discovery = defaultdict(dict)
    for row in result["failure_discovery"]:
        signal = row["policy"].split("_hybrid")[0]
        key = (row["dataset"], row["budget_families"], row["seed"], signal, row["config_id"])
        discovery[key][row["policy"]] = row
    for pair in discovery.values():
        if len(pair) != 2:
            raise ValueError("Missing paired estimator")
        a, b = pair.values()
        for field in ("expected_selected_responses", "full_selection_failing_family_yield_per_100"):
            if a[field] != b[field]:
                raise ValueError("Estimator ablation changed selected work/failures")
    return {"status": "posthoc_same_sample_estimator_control", "hybrid_samples": len(hybrids),
            "paired_estimators": result, "exact_conditional_design_moments": moments,
            "identical_failure_discovery_and_work_verified": True,
            "scope": "Exact moments average over all possible fixed-quota samples of the observed finite pool. They do not include model, label, population or reference-sampling uncertainty. Reference RMSE conditions on the observed reference score; no future-production guarantee."}


def report(result):
    lines = ["# Control: las mismas familias, con y sin ponderación", "",
             "Análisis añadido después de observar los resultados. No se generaron respuestas ni se seleccionaron familias nuevas: se reutilizaron los 300 conjuntos híbridos congelados. Se mantiene idéntico el trabajo y el número de fallos encontrados; cambia únicamente cómo se calcula el score.", "",
             "Se calcula el error de las diez semillas existentes y, además, la esperanza y la varianza exactas del diseño de muestreo sobre el pool observado. Esto evita interpretar diez semillas como diez estudios independientes.", "",
             "## Error observado sobre las mismas muestras", "",
             "Diferencia objetivo: Qwen con pasos menos Qwen directo. MAE en puntos porcentuales; presupuesto intermedio fijado en el informe anterior.", "",
             "| Dominio | Señal | Familias | Sin pesos | Pesos HT |", "|---|---|---:|---:|---:|"]
    aggregates = result["paired_estimators"]["aggregates"]
    for dataset, budget in MIDDLE.items():
        for signal in ("error", "disagreement"):
            rows = {r["policy"]: r for r in aggregates if r["dataset"] == dataset and r["budget_families"] == budget
                    and r["anchor_config"] == "qwen_direct" and r["candidate_config"] == "qwen_work"}
            unweighted = rows[f"{signal}_hybrid_same_sample_naive"]["mean_construction_absolute_error"]
            weighted = rows[f"{signal}_hybrid_ht"]["mean_construction_absolute_error"]
            lines.append(f"| {dataset} | {signal} | {budget} | {100*unweighted:.2f} | {100*weighted:.2f} |")
    lines += ["", "Los dos dominios sintéticos tienen diferencia constante cero entre estos Qwen: sus errores cero no permiten distinguir estrategias.", "",
              "## RMSE exacto del diseño condicionado al pool", "",
              "No son intervalos de confianza ni estimaciones de producción. El RMSE integra todos los posibles subconjuntos del diseño, sin simulación Monte Carlo. SRS es muestreo uniforme del mismo tamaño; su número de fallos encontrados puede diferir.", "",
              "| Dominio | Señal | Objetivo | Híbrido sin pesos | Híbrido HT | Aleatorio uniforme |", "|---|---|---|---:|---:|---:|"]
    for row in result["exact_conditional_design_moments"]:
        if row["budget_families"] != MIDDLE[row["dataset"]] or row["anchor_config"] != "qwen_direct" or row["candidate_config"] != "qwen_work" or row["constant_construction_delta"]:
            continue
        lines.append(f"| {row['dataset']} | {row['risk_signal']} | {row['target']} | " +
                     " | ".join(f"{100*row[estimator]['rmse_to_target']:.2f}" for estimator in ("naive", "ht", "srs")) + " |")
    lines += ["", "## Transferencia a una comparación que incluye Gemini", "",
              "Gemini no participa en la señal de selección. Qwen con pasos sí participa: esta comprobación cambia uno de los modelos comparados y no es un par formado enteramente por modelos nuevos. RMSE exacto de construcción, en puntos porcentuales.", "",
              "| Dominio | Señal | Qwen pasos−directo: HT / uniforme | Gemini−Qwen pasos: HT / uniforme |", "|---|---|---:|---:|"]
    moments = result["exact_conditional_design_moments"]
    for dataset in ("bbh_boolean_expressions", "gsm8k"):
        for signal in ("error", "disagreement"):
            matched = [r for r in moments if r["dataset"] == dataset and r["budget_families"] == MIDDLE[dataset]
                       and r["risk_signal"] == signal and r["target"] == "construction"]
            known = next(r for r in matched if r["anchor_config"] == "qwen_direct" and r["candidate_config"] == "qwen_work")
            transfer = next(r for r in matched if r["anchor_config"] == "qwen_work" and r["candidate_config"] == "gemini_direct")
            lines.append(f"| {dataset} | {signal} | " + " | ".join(
                f"{100*r['ht']['rmse_to_target']:.2f} / {100*r['srs']['rmse_to_target']:.2f}" for r in (known, transfer)) + " |")
    lines += ["", "La ventaja del muestreo por desacuerdo en el par usado para construir la señal desaparece en estos dos pools al comparar Gemini con Qwen con pasos. Los pesos siguen corrigiendo el sesgo del pool; lo que cambia es la varianza. Esta observación exploratoria justifica probar modelos no observados al seleccionar y no demuestra un patrón general.", "",
              "Los pesos corrigen el sesgo de seleccionar regiones con distinto riesgo para estimar este pool. No corrigen un cambio de distribución. Una media sesgada puede coincidir accidentalmente mejor con otra referencia; se reportan ambos objetivos y todos los pares de sistemas en el JSON.", "",
              "La señal de fallos usa los oráculos y las respuestas previas de construcción; la de desacuerdos usa dos configuraciones. Su coste de adquisición se conserva y no se convierte en ahorro de revisión humana.", "",
              "![Precisión al cambiar uno de los modelos comparados](" + str((BASE / "outcome_aware/matched_weight_control/transfer_precision.png").resolve()) + ")", "",
              "[Resultados completos](" + str((BASE / "outcome_aware/matched_weight_control/results.json").resolve()) + ")"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    source = args.base / "outcome_aware"
    binding = verify_published_sources(args.base)
    provenance = json.loads((source / "provenance.json").read_text())
    for entry in provenance["record_files"] + provenance["response_files"]:
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Source file changed since outcome-aware analysis: " + entry["path"])
    records = [r for entry in provenance["record_files"] for r in load_jsonl(ROOT / entry["path"])]
    responses = [r for entry in provenance["response_files"] for r in load_jsonl(ROOT / entry["path"])]
    indexed = index_responses(responses, validate_records(records), DEFAULT_CONFIGS)
    grades = {(c, r["example_id"]): grade_response(r, indexed.get((c, r["example_id"]))) for c in DEFAULT_CONFIGS for r in records}
    result = analyze(json.loads((source / "selected_families.json").read_text()), records, grades)
    previous = json.loads((source / "selector_analysis.json").read_text())
    reproduced = {}
    for field in ("paired_run_results", "aggregates", "failure_discovery"):
        before = [r for r in previous[field] if r["policy"].endswith("_hybrid_ht")]
        after = [r for r in result["paired_estimators"][field] if r["policy"].endswith("_hybrid_ht")]
        if before != after:
            raise ValueError("Reused grading/estimation diverges from completed analysis: " + field)
        reproduced[field] = len(after)
    result["original_ht_results_exactly_reproduced"] = reproduced
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["provenance"] = {"input_provenance_sha256": sha(source / "provenance.json"),
                            "frozen_selections_sha256": sha(source / "selected_families.json"),
                            "code_sha256": sha(__file__), "original_sources_verified": True,
                            "source_files": provenance["record_files"] + provenance["response_files"],
                            "previous_completion_binding": binding,
                            "code_dependencies": {name: sha(ROOT / "experiments/production_eval" / name)
                                                  for name in ("analyze_outcomes_v2.py", "outcome_aware_selection.py",
                                                               "selection_methods.py", "benchmark_data.py", "synthetic_tasks.py")}}
    destination = source / "matched_weight_control"
    destination.mkdir(exist_ok=True)
    (destination / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (destination / "report.md").write_text(report(result))
    print(json.dumps({"samples": result["hybrid_samples"], "exact_design_rows": len(result["exact_conditional_design_moments"]), "report": str(destination / "report.md")}))


if __name__ == "__main__":
    main()
