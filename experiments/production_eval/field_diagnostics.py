"""Post-hoc, secondary field diagnostics for procedural JSON tasks.

Primary all-fields exact match is untouched. Missing or format-invalid outputs
receive zero credit on every field-success metric. Raw absolute error is
undefined for unavailable numeric fields; a full-denominator mean is therefore
reported only when every value exists. A capped relative error assigns maximal
error to unavailable fields, preserving an unconditional denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.synthetic_tasks import json_exact_grade


def strict_json(text):
    def pairs(values):
        output = {}
        for key, value in values:
            if key in output:
                raise ValueError("Duplicate JSON key")
            output[key] = value
        return output

    def constant(_):
        raise ValueError("Nonfinite JSON constant")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError):
        return None


def set_scores(actual, expected, available):
    truth = set(expected)
    if not available:
        return {"precision": 0., "recall": 0., "f1": 0., "exact": False,
                "extra_count": 0, "missing_count": len(truth), "duplicate_count": 0,
                "order_compliant": False}
    predicted = set(actual)
    overlap = len(predicted & truth)
    precision = overlap / len(predicted) if predicted else float(not truth)
    recall = overlap / len(truth) if truth else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"precision": precision, "recall": recall, "f1": f1,
            "exact": predicted == truth, "extra_count": len(predicted - truth),
            "missing_count": len(truth - predicted), "duplicate_count": len(actual) - len(predicted),
            "order_compliant": actual == sorted(actual)}


def numeric(actual, expected, available):
    available = available and type(actual) is int
    error = abs(actual - expected) if available else None
    if error is None:
        capped = 1.
    elif error >= max(1, abs(expected)):
        capped = 1.
    else:
        capped = error / max(1, abs(expected))
    return {"exact": bool(available and actual == expected), "absolute_error": error,
            "available": bool(available), "capped_relative_absolute_error": capped}


def diagnose(actual, expected, domain, *, available=True):
    available = bool(available and isinstance(actual, dict))
    value = actual if available else {}
    scores = {"response_available_and_format_valid": float(available)}
    errors = {}
    if domain == "structured_records":
        items = value.get("selected_ids")
        ids_valid = available and type(items) is list and all(type(item) is str for item in items)
        ids = set_scores(items if ids_valid else [], expected["selected_ids"], ids_valid)
        scores.update({"selected_id_set_" + key: float(ids[key]) for key in ["precision", "recall", "f1", "exact"]})
        scores.update(selected_id_order_compliance=float(ids["order_compliant"]),
                      selected_id_extra_flag=float(ids["extra_count"] > 0),
                      selected_id_duplicate_flag=float(ids["duplicate_count"] > 0),
                      selected_id_missing_flag=float(ids["missing_count"] > 0))
        totals = ("total_units", "total_value")
        details = {"selected_id_diagnostics": ids}
    elif domain == "inventory_tracking":
        stock = value.get("stock")
        stock_valid = available and type(stock) is list and all(isinstance(row, dict) and type(row.get("item")) is str for row in stock)
        stock = stock if stock_valid else []
        names = [row["item"] for row in stock]
        expected_names = [row["item"] for row in expected["stock"]]
        ids = set_scores(names, expected_names, stock_valid)
        counts = Counter(names)
        unique_rows = {row["item"]: row for row in stock if counts[row["item"]] == 1}
        cells = []
        for row in expected["stock"]:
            for location in ("east", "west"):
                observed = unique_rows.get(row["item"], {}).get(location)
                cells.append(numeric(observed, row[location], stock_valid and row["item"] in unique_rows))
        scores.update({"item_set_" + key: float(ids[key]) for key in ["precision", "recall", "f1", "exact"]})
        scores.update(item_order_compliance=float(ids["order_compliant"]),
                      item_extra_flag=float(ids["extra_count"] > 0), item_duplicate_flag=float(ids["duplicate_count"] > 0),
                      item_missing_flag=float(ids["missing_count"] > 0),
                      expected_item_location_count_accuracy=mean(float(row["exact"]) for row in cells),
                      expected_item_location_capped_relative_error=mean(row["capped_relative_absolute_error"] for row in cells))
        all_numeric = stock_valid and all(type(row.get(location)) is int for row in stock for location in ("east", "west"))
        internal_sum = sum(row[location] for row in stock for location in ("east", "west")) if all_numeric else None
        scores["reported_total_equals_sum_of_reported_stock"] = float(all_numeric and type(value.get("total_units")) is int and value["total_units"] == internal_sum)
        totals = ("total_units",)
        errors["expected_item_location_mean_absolute_error"] = (sum(row["absolute_error"] for row in cells) / len(cells)
                                                                   if all(row["absolute_error"] is not None for row in cells) else None)
        details = {"item_diagnostics": ids, "expected_location_cells": len(cells),
                   "unavailable_or_duplicate_item_location_cells": sum(not row["available"] for row in cells)}
    else:
        raise ValueError("Unknown procedural domain")
    for key in totals:
        outcome = numeric(value.get(key), expected[key], available)
        scores[key + "_exact"] = float(outcome["exact"])
        scores[key + "_capped_relative_error"] = outcome["capped_relative_absolute_error"]
        errors[key + "_absolute_error"] = outcome["absolute_error"]
    serialized = json.dumps(actual, separators=(",", ":")) if available else None
    scores["primary_strict_exact_match_recomputed"] = float(available and json_exact_grade(serialized, json.dumps(expected)))
    return {"scores": scores, "absolute_errors": errors, "details": details}


def aggregate(rows):
    """Within-family averages first; every expected row remains in denominator."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["config_id"], row["domain"], row["role"], row["regime"])].append(row)
        grouped[(row["config_id"], row["domain"], row["role"], "all_regimes")].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        families = defaultdict(list)
        for row in items:
            families[row["family_id"]].append(row)
        metrics = {}
        for metric in items[0]["scores"]:
            metrics[metric] = mean(mean(row["scores"][metric] for row in family) for family in families.values())
        error_report = {}
        for metric in items[0]["absolute_errors"]:
            family_errors = [mean(row["absolute_errors"][metric] for row in family)
                             if all(row["absolute_errors"][metric] is not None for row in family) else None
                             for family in families.values()]
            all_available = all(error is not None for error in family_errors)
            error_report[metric] = {"mean_over_all_expected_families": mean(family_errors) if all_available else None,
                                    "expected_families": len(family_errors),
                                    "families_with_any_unavailable_field": sum(error is None for error in family_errors),
                                    "missing_error_interpretation": "Raw absolute error has no numeric prediction to compare when unavailable; no valid-only mean is substituted. Field exactness remains zero and capped relative error is one."}
        result.append({"config_id": key[0], "domain": key[1], "role": key[2], "regime": key[3],
                       "expected_examples": len(items), "expected_families": len(families),
                       "missing_or_transport_failures": sum(row["availability"] == "missing_or_transport_failure" for row in items),
                       "completed_invalid_formats": sum(row["availability"] == "completed_invalid_format" for row in items),
                       "family_mean_metrics": metrics, "absolute_error_diagnostics": error_report})
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_v2(output):
    base = ROOT / "output/research/production_eval/experiments_v2"
    primary = json.loads((base / "analysis/outcome_analysis.json").read_text())
    source = base / "synthetic/normalized_tasks.jsonl"
    tasks = read_jsonl(source)
    task_ids = {row["example_id"] for row in tasks}
    configs = primary["config_ids"]
    responses, provenance = {}, []
    for spec in primary["input_provenance"]["response_files"]:
        path = ROOT / spec["path"]
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != spec["sha256"]:
            raise ValueError("Core response source changed")
        provenance.append({"path": str(path.relative_to(ROOT)), "sha256": h})
        for row in read_jsonl(path):
            if row["example_id"] not in task_ids:
                continue
            key = (row["config_id"], row["example_id"])
            if key in responses:
                raise ValueError("Duplicate config/example response")
            responses[key] = row
    rows = []
    for config in configs:
        for task in tasks:
            if hashlib.sha256(task["input_text"].encode()).hexdigest() != task["input_hash"]:
                raise ValueError("Task input hash mismatch")
            response = responses.get((config, task["example_id"]))
            if response is not None and response["input_hash"] != task["input_hash"]:
                raise ValueError("Response and source input disagree")
            availability = "missing_or_transport_failure" if response is None or response["status"] != "success" else \
                           "completed_invalid_format" if response["format_status"] != "valid" else "available"
            actual = response.get("answer") if availability == "available" else None
            result = diagnose(actual, json.loads(task["answer"]), task["task_family"], available=availability == "available")
            rows.append({"config_id": config, "example_id": task["example_id"], "family_id": task["family_id"],
                         "input_hash": task["input_hash"], "domain": task["task_family"], "role": task["split"],
                         "regime": task["metadata"]["shift_regime"], "variant": task["metadata"]["variant"],
                         "availability": availability, **result})
    report = {"status": "posthoc_secondary_field_diagnostics", "created_at": datetime.now(timezone.utc).isoformat(),
              "primary_exact_match_changed": False, "selection_or_training_changed": False,
              "rows": rows, "groups": aggregate(rows), "source_responses": provenance,
              "tasks_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "definitions": {
                  "sets": "Precision/recall/F1 compare unique string IDs; duplicates are separately flagged and never increase overlap. Lexical order is a separate condition.",
                  "inventory_counts": "Each expected item/location has one vote within a task. Missing or duplicated predicted items fail both locations. Extra items are separately flagged.",
                  "internal_consistency": "Reported total equals sum over every reported stock row, including extras/duplicates; this is bookkeeping consistency, not correctness.",
                  "absolute_error": "Reported only as a full-family mean when every required numeric prediction exists. Unavailable fields are not dropped or replaced by zero error.",
                  "capped_relative_error": "min(1,abs(prediction-truth)/max(1,abs(truth))); unavailable numeric fields receive1. This added diagnostic is post hoc.",
                  "weighting": "Equal expected-task weight within family, then equal family weight within each config/domain/role/regime. All-regimes pools families and is separate from regime-specific results.",
                  "unavailable": "Missing, transport-failed and format-invalid responses receive zero field credit in this secondary availability-aware summary. Primary analysis continues to distinguish unknown infrastructure outcomes.",
              }, "scope": "Post-hoc explanatory metrics on controlled synthetic tasks. No model reruns, parser repair, human gold claims, selection changes or significance claims."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "field_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Diagnósticos secundarios por campo", "", "Análisis posterior a los resultados. El exact match principal permanece intacto. Promedios por familia; los formatos inválidos y resultados no disponibles reciben cero crédito por campo.", "",
             "| Config | Dominio | Rol | Exacto | F1 de IDs | Orden | Total exacto | Conteos por ubicación | Consistencia interna |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["groups"]:
        if row["regime"] != "all_regimes":
            continue
        m = row["family_mean_metrics"]
        def pct(key):
            return "—" if key not in m else f"{100*m[key]:.1f}%"
        prefix = "selected_id" if row["domain"] == "structured_records" else "item"
        lines.append(f"| {row['config_id']} | {row['domain']} | {row['role']} | {pct('primary_strict_exact_match_recomputed')} | {pct(prefix+'_set_f1')} | {pct(prefix+'_order_compliance')} | {pct('total_units_exact')} | {pct('expected_item_location_count_accuracy')} | {pct('reported_total_equals_sum_of_reported_stock')} |")
    lines.extend(["", "Un F1 de IDs alto puede coexistir con cifras incorrectas. La consistencia interna comprueba las cifras publicadas entre sí; no comprueba su concordancia con la fuente. Duplicados, extras, errores absolutos y todos los regímenes aparecen en el JSON.",
                  "", "Si falta un campo numérico, el error absoluto medio de todas las familias queda indefinido: no se publica una media favorable calculada solo sobre respuestas válidas. La exactitud mantiene el denominador completo; el error relativo acotado asigna error1 al campo no disponible.", ""])
    (output / "field_diagnostics.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "groups": len(report["groups"]), "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/research/production_eval/experiments_v2/field_diagnostics")
    args = parser.parse_args()
    run_v2(args.output)


if __name__ == "__main__":
    main()
