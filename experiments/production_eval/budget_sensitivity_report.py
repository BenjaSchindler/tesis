"""Descriptive report for the frozen construction output-budget study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.analyze_outcomes_v2 import domain, family_members, grade_response
from experiments.production_eval.budget_sensitivity import DEFAULT_BASE, analyze_experiment, load_prepared, sha


def report(directory):
    analysis = analyze_experiment(directory)
    plan, rows = load_prepared(directory)
    ids = {row["example_id"] for row in rows}
    paths = {1024: Path(plan["baseline"]["output_path"]),
             **{budget: directory / f"responses_b{budget}.jsonl" for budget in (2048, 4096)}}
    responses = {budget: {row["example_id"]: row for row in inference.read_jsonl(path) if row["example_id"] in ids}
                 for budget, path in paths.items()}
    grades = {budget: {row["example_id"]: grade_response(row, responses[budget].get(row["example_id"])) for row in rows}
              for budget in paths}
    family_rows, summary, usage, stability = [], [], {}, {}
    for budget, path in paths.items():
        attempts = [row for row in inference.read_jsonl(Path(str(path) + ".attempts.jsonl")) if row["example_id"] in ids]
        usage[budget] = {"attempts": len(attempts), "api_usd": sum(row["charged_nanodollars"] for row in attempts) / 1e9}
        for name in ("prompt_tokens", "completion_tokens"):
            values = [row["usage"][name] for row in attempts if isinstance(row.get("usage"), dict)
                      and type(row["usage"].get(name)) is int and row["usage"][name] >= 0]
            usage[budget][name] = {"reported_attempts": len(values), "unknown_attempts": len(attempts) - len(values),
                                   "sum_reported": sum(values), "mean_reported": mean(values) if values else None,
                                   "median_reported": median(values) if values else None}
        invocations = inference.read_jsonl(directory / f"invocations_b{budget}.jsonl") if budget != 1024 else []
        usage[budget]["wall_seconds"] = sum(row["wall_seconds_this_invocation"] for row in invocations) if invocations else None
        usage[budget]["wall_time_scope"] = "Separate local invocations over exactly this subset." if budget != 1024 else "Unavailable for this subset: these requests were interleaved with the larger 1350-request baseline run."
        for group in sorted({domain(row) for row in rows}) + ["balanced_mixture"]:
            selected = [row for row in rows if group == "balanced_mixture" or domain(row) == group]
            families = family_members(selected)
            complete = [[grades[budget][row["example_id"]] for row in members] for members in families.values()
                        if all(grades[budget][row["example_id"]]["observed"] for row in members)]
            observed = [grades[budget][row["example_id"]] for row in selected if grades[budget][row["example_id"]]["observed"]]
            summary.append({"budget": budget, "dataset": group, "expected_rows": len(selected),
                            "observed_rows": len(observed), "unknown_rows": len(selected) - len(observed),
                            "complete_families": len(complete), "expected_families": len(families),
                            "family_accuracy": mean(mean(int(g["correct"]) for g in family) for family in complete) if complete else None,
                            "outer_json_or_schema_failures": sum(responses[budget].get(row["example_id"], {}).get("status") == "success"
                                and responses[budget][row["example_id"]].get("format_status") != "valid" for row in selected),
                            "answer_format_failures_including_outer": sum(not g["format_valid"] for g in observed)})
        if budget == 1024:
            continue
        nonbinding = []
        changed = []
        for row in rows:
            identifier = row["example_id"]
            a, b = responses[1024].get(identifier), responses[budget].get(identifier)
            if not a or not b:
                continue
            if a.get("raw_output_text") != b.get("raw_output_text"):
                changed.append(identifier)
                au, bu = a.get("usage") or {}, b.get("usage") or {}
                if a.get("finish_reason") == b.get("finish_reason") == "stop" and type(au.get("completion_tokens")) is int and type(bu.get("completion_tokens")) is int and max(au["completion_tokens"], bu["completion_tokens"]) < 1024:
                    nonbinding.append(identifier)
        stability[budget] = {"changed_raw_outputs": len(changed), "changed_despite_both_stop_under_1024": len(nonbinding),
                             "changed_canonical_predictions_among_nonbinding": sum(grades[1024][i]["canonical_prediction"] != grades[budget][i]["canonical_prediction"] for i in nonbinding),
                             "nonbinding_example_ids": nonbinding,
                             "interpretation": "Changes with both outputs finishing below 1024 show execution variability; observed accuracy differences cannot be attributed solely to a binding output-token ceiling."}
        for family, members in sorted(family_members(rows).items()):
            member_ids = [row["example_id"] for row in members]
            observed = all(grades[b][identifier]["observed"] for b in (1024, budget) for identifier in member_ids)
            a = mean(int(grades[1024][i]["correct"]) for i in member_ids) if observed else None
            b = mean(int(grades[budget][i]["correct"]) for i in member_ids) if observed else None
            delta = b - a if observed else None
            family_rows.append({"budget": budget, "family_id": family, "dataset": domain(members[0]),
                                "descendants": len(members), "paired_complete": observed,
                                "baseline_accuracy": a, "variant_accuracy": b, "delta": delta,
                                "outcome": "unknown" if not observed else "improved" if delta > 0 else "regressed" if delta < 0 else "tie"})
    result = {"status": "descriptive_construction_budget_sensitivity", "created_at_utc": inference.utc_now(),
              "scope": plan["scope"], "plan_sha256": sha(directory / "plan.json"),
              "summary": summary, "usage": usage, "nonbinding_output_variation": stability,
              "paired_family_deltas": family_rows, "paired_comparisons": {budget: value["paired_comparisons"] for budget, value in analysis["variants"].items()},
              "limitations": ["The 100 families and 250 requests are a post hoc construction subset from an older frozen pool, not a new held-out test.",
                              "Synthetic descendants remain within families; the balanced score gives each domain one-quarter weight.",
                              "Means and medians of token usage condition on reported usage; parser-payload failures retain explicit unknown counts.",
                              "Baseline subset wall time cannot be compared with the two standalone subset runs.",
                              "Nonbinding output differences indicate execution variability despite temperature 0; an identical-budget repeat is needed to estimate that component.",
                              "No significance tests or general model-superiority claims are made."]}
    (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (directory / "family_deltas.json").write_text(json.dumps(family_rows, indent=2, allow_nan=False) + "\n")
    lines = ["# Sensibilidad al presupuesto de salida", "", "Comparación exploratoria sobre 100 familias de construcción y sus 250 solicitudes. No es una nueva prueba independiente ni deriva de producción.", "",
             "| Dominio | Tokens máximos | Exactitud por familia | Familias completas | Fallos JSON/esquema | Fallos de formato de respuesta, incluidos los anteriores |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in summary:
        score = "pendiente" if item["family_accuracy"] is None else f"{100 * item['family_accuracy']:.1f}%"
        lines.append(f"| {item['dataset']} | {item['budget']} | {score} | {item['complete_families']}/{item['expected_families']} | {item['outer_json_or_schema_failures']} | {item['answer_format_failures_including_outer']} |")
    lines += ["", "La mezcla balanceada promedia familias, con un cuarto del peso para cada dominio. Los dos tipos de fallos de formato se solapan; no deben sumarse.", "",
              "| Tokens máximos | Tokens de salida: media / mediana reportadas | Intentos sin uso de salida conocido | Tiempo de ejecución del subconjunto | Coste API |",
              "|---:|---:|---:|---:|---:|"]
    for budget, item in usage.items():
        token = item["completion_tokens"]
        measured = "pendiente" if token["mean_reported"] is None else f"{token['mean_reported']:.1f} / {token['median_reported']:.1f}"
        wall = "No separable del experimento original" if item["wall_seconds"] is None else f"{item['wall_seconds']:.1f} s"
        lines.append(f"| {budget} | {measured} | {token['unknown_attempts']} | {wall} | US${item['api_usd']:.3f} |")
    lines += ["", "Los promedios de tokens usan solo intentos con uso reportado. El coste API cero no implica coste computacional cero; la suma de latencias de solicitudes incluye espera en cola.", "",
              "| Comparación contra 1024 | Familias emparejadas | Diferencia de exactitud | Familias mejoran / empeoran / empatan | Cambios de salida aunque ambos terminan antes de 1024 |",
              "|---|---:|---:|---:|---:|"]
    for budget in (2048, 4096):
        comparison = next(r for r in result["paired_comparisons"][str(budget)] if r["dataset"] == "balanced_mixture")
        counts = {name: sum(r["budget"] == budget and r["outcome"] == name for r in family_rows) for name in ("improved", "regressed", "tie")}
        delta = "pendiente" if comparison["paired_accuracy_delta"] is None else f"{100 * comparison['paired_accuracy_delta']:+.1f} pp"
        lines.append(f"| {budget} | {comparison['paired_complete_families']}/100 | {delta} | {counts['improved']} / {counts['regressed']} / {counts['tie']} | {stability[budget]['changed_despite_both_stop_under_1024']} |")
    lines += ["", "Las salidas cambiaron también en solicitudes donde ninguno de los dos límites fue alcanzado. Por ello, las diferencias observadas mezclan el presupuesto y variación de ejecución; no demuestran un efecto causal del límite. Una repetición con el mismo presupuesto ayudaría a medir esa variación.", "",
              "Se mantuvieron exactamente la entrada, el modelo, el perfil work, el esquema sin respuestas de referencia, la semilla, la temperatura y el límite de contexto. No se aplicaron atajos para tareas cortas. Los resultados faltantes permanecen desconocidos; las comparaciones requieren todos los descendientes de una familia. No se calculó significación estadística.", "",
              "Los deltas de cada familia están en `family_deltas.json`; las comparaciones, usos y verificaciones detalladas están en `report.json`, `analysis.json` y `plan.json`.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_BASE / "budget_sensitivity")
    result = report(parser.parse_args().directory.resolve())
    print(json.dumps({"status": result["status"], "family_delta_rows": len(result["paired_family_deltas"])}))
