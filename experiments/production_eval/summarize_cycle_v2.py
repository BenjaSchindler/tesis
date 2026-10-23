"""Spanish synthesis and exportable scientific figures from completed v2 runs.

Reads analysis/results/attempt ledgers; never grades, selects, or changes them.
Usage:
  python -m experiments.production_eval.summarize_cycle_v2 --analysis ANALYSIS.json \
    --responses qwen_direct.jsonl --responses qwen_work.jsonl \
    --responses gemini_direct.jsonl --output REPORT_DIRECTORY
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import tempfile
from typing import Any


DOMAINS = ("gsm8k", "bbh_boolean_expressions", "structured_records", "inventory_tracking")
DOMAIN_NAMES = {"gsm8k": "GSM8K", "bbh_boolean_expressions": "BBH booleano",
                "structured_records": "Registros estructurados", "inventory_tracking": "Inventario",
                "balanced_mixture": "Mezcla artificial equilibrada"}
CONFIG_NAMES = {"qwen_direct": "Qwen · directo", "qwen_work": "Qwen · trabajo explícito", "gemini_direct": "Gemini · directo"}
REGIME_NAMES = {"random_benchmark_split": "Referencia aleatoria", "stationary": "Estacionaria",
                "more_complex": "Más compleja", "changed_template": "Otra plantilla",
                "equal_domain_reference_mixture": "Referencia: ¼ por dominio"}
MIDDLE_BUDGETS = {"gsm8k": 50, "bbh_boolean_expressions": 25, "structured_records": 10,
                  "inventory_tracking": 10, "balanced_mixture": 40}
POLICY_NAMES = {"uniform_random": "Aleatorio", "stratified_random": "Estratificado",
                "random_plus_targeted": "Aleatorio + longitud", "minhash_farthest_first": "MinHash",
                "centroid_proximity": "Centroide", "facility_location": "Cobertura"}
REP_NAMES = {"mpnet_head": "MPNet inicio", "mpnet_head_tail": "MPNet extremos",
             "mpnet_mean_chunks": "MPNet bloques", "qwen3_document": "Qwen3 emb."}
COLORS = ("#0072B2", "#D55E00", "#009E73")
CONFIG_COLORS = dict(zip(CONFIG_NAMES, COLORS))


def regime_order(value: str) -> tuple[int, str]:
    order = ("random_benchmark_split", "stationary", "more_complex", "changed_template", "equal_domain_reference_mixture")
    return (order.index(value) if value in order else len(order), value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps need a timezone")
    return result


def pct(value: float | None) -> str:
    return "pendiente" if value is None else f"{100 * value:.1f}%"


def pp(value: float | None, signed: bool = False) -> str:
    return "pendiente" if value is None else format(100 * value, "+.2f" if signed else ".2f")


def selector_label(row: dict[str, Any]) -> str:
    label = POLICY_NAMES.get(row["policy"], row["policy"])
    representation = row.get("embedding_variant")
    return label + (" · " + REP_NAMES.get(representation, representation) if representation else "")


def ledger_totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    charges, settled = {}, set()
    for event in events:
        call = event["call_id"]
        if event["event"] == "reserve":
            amount = event["reservation_nanodollars"]
            if call in charges or type(amount) is not int or amount < 0:
                raise ValueError("Malformed/duplicate reservation")
            charges[call] = amount
        elif event["event"] == "settle":
            amount = event["charged_nanodollars"]
            if call not in charges or call in settled or type(amount) is not int or amount < 0:
                raise ValueError("Malformed/duplicate settlement")
            charges[call] = amount
            settled.add(call)
        else:
            raise ValueError("Unknown ledger event")
    return {"ledger_accounted_usd": sum(charges.values()) / 1e9,
            "reserved_attempts": len(charges), "unsettled_attempts": len(charges) - len(settled),
            "accounting_note": "Final settlements plus full outstanding reservations; an estimate/accounting bound, not an invoice."}


def summarize_usage(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {name: [] for name in ("input_tokens", "reported_output_tokens", "reported_reasoning_tokens", "total_tokens")}
    for attempt in attempts:
        usage = attempt.get("usage") or {}
        detail = usage.get("completion_tokens_details") or {}
        aliases = {"input_tokens": usage.get("prompt_token_count", usage.get("prompt_tokens")),
                   "reported_output_tokens": usage.get("candidates_token_count", usage.get("completion_tokens")),
                   "reported_reasoning_tokens": usage.get("thoughts_token_count", detail.get("reasoning_tokens")),
                   "total_tokens": usage.get("total_token_count", usage.get("total_tokens"))}
        for key, value in aliases.items():
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid reported token count")
                counters[key].append(value)
    return {key: {"known_sum": sum(values) if values else None, "attempts_with_measurement": len(values),
                  "attempts_without_measurement": len(attempts) - len(values)} for key, values in counters.items()}


def summarize_resources(paths: list[Path], analysis: dict[str, Any], attempt_overrides: dict[str, Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_config, attempts_by_config, ledger_by_config = defaultdict(list), defaultdict(list), defaultdict(list)
    sources, fallback, missing_ledgers, identities = [], Counter(), Counter(), set()
    for path in paths:
        terminal = read_jsonl(path)
        if not terminal:
            raise ValueError("Empty terminal file")
        configs = {row["config_id"] for row in terminal}
        if len(configs) != 1:
            raise ValueError("One config is required per terminal file")
        config = next(iter(configs))
        for row in terminal:
            key = (config, row["example_id"])
            if key in identities or row.get("terminal") is not True:
                raise ValueError("Duplicate or nonterminal inference row")
            identities.add(key)
        by_config[config].extend(terminal)
        attempt_path, ledger_path = Path(str(path) + ".attempts.jsonl"), Path(str(path) + ".ledger.jsonl")
        if attempt_overrides and config in attempt_overrides:
            attempt_path = attempt_overrides[config]
        attempts = read_jsonl(attempt_path) if attempt_path.exists() else terminal
        if not attempt_path.exists():
            fallback[config] += 1
        call_ids = [row["call_id"] for row in attempts]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Duplicate attempt call ID")
        attempts_by_config[config].extend(attempts)
        if ledger_path.exists():
            ledger_by_config[config].append(ledger_totals(read_jsonl(ledger_path)))
        else:
            missing_ledgers[config] += 1
        sources.append({"terminal": {"path": str(path), "sha256": digest(path)},
                        "attempts": {"path": str(attempt_path), "sha256": digest(attempt_path)} if attempt_path.exists() else None,
                        "ledger": {"path": str(ledger_path), "sha256": digest(ledger_path)} if ledger_path.exists() else None})
    expected, observed_analysis, grader_format_failures = Counter(), Counter(), Counter()
    for row in analysis["group_summary"]:
        expected[row["config_id"]] += row["expected_examples"]
        observed_analysis[row["config_id"]] += row["observed_examples"]
        grader_format_failures[row["config_id"]] += row["format_failures"]
    if set(by_config) != set(analysis["config_ids"]):
        raise ValueError("Terminal configs differ from analysis configs")
    result = []
    for config in analysis["config_ids"]:
        terminal, attempts = by_config[config], attempts_by_config[config]
        success = sum(row["status"] == "success" for row in terminal)
        if success != observed_analysis[config]:
            raise ValueError("Terminal outcome coverage differs from analyzed coverage")
        starts = [timestamp(row["started_at_utc"]) for row in attempts if row.get("started_at_utc")]
        ends = [timestamp(row["completed_at_utc"]) for row in attempts if row.get("completed_at_utc")]
        latencies = [row["latency_seconds"] for row in attempts if isinstance(row.get("latency_seconds"), (int, float))]
        complete_ledger = not missing_ledgers[config]
        ledgers = ledger_by_config[config]
        terminal_charge = sum(row.get("charged_nanodollars", 0) for row in terminal) / 1e9
        finishes = Counter(str(reason) for row in terminal for reason in (row.get("finish_reason") if isinstance(row.get("finish_reason"), list) else [row.get("finish_reason")]))
        result.append({"config_id": config, "provider": sorted({row.get("provider", "unknown") for row in terminal}),
                       "actual_model_versions": sorted({row["actual_model_version"] for row in terminal if row.get("actual_model_version")}),
                       "expected_requests": expected[config], "terminal_requests": len(terminal), "received_model_responses": success,
                       "unknown_outcomes": expected[config] - success, "transport_status_counts": dict(Counter(row["status"] for row in terminal)),
                       "format_status_counts": dict(Counter(row["format_status"] for row in terminal)), "finish_reason_counts": dict(finishes),
                       "grader_answer_format_failures": grader_format_failures[config],
                       "attempt_count": sum(row["reserved_attempts"] for row in ledgers) if complete_ledger else len(attempts),
                       "available_attempt_records": len(attempts), "attempt_log_complete": not fallback[config],
                       "ledger_accounted_usd": sum(row["ledger_accounted_usd"] for row in ledgers) if complete_ledger else None,
                       "terminal_only_charge_usd_partial": None if complete_ledger else terminal_charge,
                       "unsettled_attempts": sum(row["unsettled_attempts"] for row in ledgers) if complete_ledger else None,
                       "accounting_basis_counts": dict(Counter(row.get("accounting_basis", "unknown") for row in attempts)),
                       "token_usage_all_available_attempts": summarize_usage(attempts),
                       "first_attempt_utc": min(starts).isoformat() if starts else None,
                       "last_attempt_utc": max(ends).isoformat() if ends else None,
                       "observed_wall_span_seconds": (max(ends) - min(starts)).total_seconds() if starts and ends else None,
                       "sum_attempt_latencies_seconds": sum(latencies) if latencies else None,
                       "median_attempt_latency_seconds": median(latencies) if latencies else None,
                       "cost_note": "Local API charge excludes hardware/electricity; ledger totals include retries and outstanding reservations. Tokenizers and concurrency differ."})
    declared = analysis.get("input_provenance", {}).get("response_files")
    if declared is not None and sorted(row["sha256"] for row in declared) != sorted(digest(path) for path in paths):
        raise ValueError("Terminal files changed after the supplied analysis snapshot")
    return result, {"sources": sources, "terminal_identity_count": len(identities)}


def middle_budget_rows(analysis: dict[str, Any], candidate: str = "qwen_work") -> list[dict[str, Any]]:
    rows = []
    selectors = analysis.get("selector_estimation", {})
    for value in selectors.get("aggregates", []):
        if value["anchor_config"] != "qwen_direct" or value["candidate_config"] != candidate:
            continue
        if value["budget_families"] == MIDDLE_BUDGETS.get(value["dataset"]):
            rows.append(value)
    return sorted(rows, key=lambda row: (list(MIDDLE_BUDGETS).index(row["dataset"]), selector_label(row)))


def save_figure(fig, output: Path, stem: str) -> dict[str, str]:
    paths = {}
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths[extension] = str(path.resolve())
    return paths


def make_figures(analysis: dict[str, Any], selections: list[dict[str, Any]], output: Path, selector_candidate: str = "qwen_work") -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "production_eval_matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "axes.titleweight": "bold"})
    configs = analysis["config_ids"]
    figures = {}
    fig, ax = plt.subplots(figsize=(11.5, 5.4), layout="constrained")
    lookup = {(row["dataset"], row["config_id"]): row for row in analysis["group_summary"] if row["role"] == "reference"}
    x, width = np.arange(len(DOMAINS)), .24
    for index, config in enumerate(configs):
        values = [lookup.get((dataset, config), {}).get("full_population_family_mean_accuracy") for dataset in DOMAINS]
        positions = x + (index - (len(configs) - 1) / 2) * width
        bars = ax.bar(positions, [100 * value if value is not None else 0 for value in values], width,
                      color=CONFIG_COLORS.get(config, COLORS[index % len(COLORS)]), label=CONFIG_NAMES.get(config, config))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, "N/D" if value is None else f"{100 * value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [DOMAIN_NAMES[domain] for domain in DOMAINS])
    ax.set_ylim(0, 111)
    ax.set_ylabel("Exactitud media por familia (%)")
    ax.set_title("Referencias independientes de la selección")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    figures["reference_accuracy"] = save_figure(fig, output, "reference_accuracy")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, layout="constrained")
    regimes = ("construction", "stationary", "more_complex", "changed_template")
    for ax, dataset in zip(axes, DOMAINS[2:]):
        for index, config in enumerate(configs):
            source = next(row for row in analysis["group_summary"] if row["dataset"] == dataset and row["role"] == "construction" and row["config_id"] == config)
            reference = {row["regime"]: row for row in analysis["regime_summary"] if row["dataset"] == dataset and row["role"] == "reference" and row["config_id"] == config}
            selected = [source] + [reference[reg] for reg in regimes[1:]]
            values = [row["full_population_family_mean_accuracy"] for row in selected]
            bars = ax.bar(np.arange(4) + (index - (len(configs) - 1) / 2) * width,
                   [100 * value if value is not None else np.nan for value in values], width,
                   color=CONFIG_COLORS.get(config, COLORS[index % len(COLORS)]), label=CONFIG_NAMES.get(config, config))
            for bar, value in zip(bars, values):
                if value is None or value == 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, 1, "N/D" if value is None else "0", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(4), ["Construcción\n25 familias", "Ref. estacionaria\n9 familias", "Ref. compleja\n8 familias", "Ref. otra plantilla\n8 familias"], fontsize=9)
        ax.set_title(DOMAIN_NAMES[dataset])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Exactitud media por familia (%)")
    fig.suptitle("Comparación de distribuciones sintéticas controladas", weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figures["controlled_regimes"] = save_figure(fig, output, "controlled_regimes")
    plt.close(fig)

    if selections:
        fig, axes = plt.subplots(3, 2, figsize=(17, 18), layout="constrained")
        for ax, dataset in zip(axes.flat, MIDDLE_BUDGETS):
            rows = [row for row in selections if row["dataset"] == dataset]
            paired = [row for row in analysis.get("selector_estimation", {}).get("paired_run_results", [])
                      if row["dataset"] == dataset and row["anchor_config"] == "qwen_direct" and row["candidate_config"] == selector_candidate]
            constant = paired and all(row.get("full_construction_family_delta_is_constant") is True for row in paired)
            if constant:
                zero_targets = all(row["full_construction_delta"] == 0 and all(ref["full_reference_delta"] == 0 for ref in row["references"]) for row in paired)
                ax.set_facecolor("#eeeeee")
                ax.set_title(f"{DOMAIN_NAMES[dataset]} · {MIDDLE_BUDGETS[dataset]} familias", fontsize=11, color="#666666")
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(.5, .55, "Δ = 0 en todas las familias de construcción" if zero_targets else "Δ constante en construcción", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, weight="bold", color="#555555")
                ax.text(.5, .37, "Todos los selectores normalizados reproducen ese Δ.\nEl error cero no demuestra una geometría mejor.\n\nDominio no informativo para comparar selectores.",
                        transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#666666", linespacing=1.7)
                continue
            labels = [selector_label(row) for row in rows]
            reference_regimes = sorted({reg for row in rows for reg in row["reference_mean_absolute_errors"]}, key=regime_order)
            series = ["construction"] + reference_regimes
            for index, key in enumerate(series):
                values = [row["mean_construction_absolute_error"] if key == "construction" else row["reference_mean_absolute_errors"].get(key) for row in rows]
                y = np.arange(len(rows)) + (index - (len(series) - 1) / 2) * .12
                ax.scatter([100 * value if value is not None else np.nan for value in values], y, s=22,
                           color=("#555555", "#0072B2", "#D55E00", "#009E73")[index],
                           marker=("o", "s", "^", "D")[index],
                           label="Construcción" if key == "construction" else REGIME_NAMES.get(key, key))
            ax.set_yticks(range(len(rows)), labels, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(left=-.4)
            ax.set_xlabel("Error absoluto medio de Δ (pp)")
            ax.set_title(f"{DOMAIN_NAMES[dataset]} · {MIDDLE_BUDGETS[dataset]} familias", fontsize=11)
            ax.grid(axis="x", alpha=.2)
            ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .9, f"Comparación: Qwen directo → {CONFIG_NAMES[selector_candidate]}\n\nMedias sobre 10 semillas para métodos aleatorios.\nUna selección para cada método determinista.\n\nCada régimen mantiene su propia referencia.\nNo se muestran intervalos de confianza.\nLas semillas reutilizan los mismos resultados.\n\nMenor error no equivale a descubrir más fallos.\nLa mezcla es artificial, con ¼ por dominio.", va="top", fontsize=11, linespacing=1.7)
        fig.suptitle("Fidelidad de los selectores en presupuestos medios", fontsize=17, weight="bold")
        figures["selector_errors"] = save_figure(fig, output, "selector_errors_middle_budget")
        plt.close(fig)
    return figures


def render_report(analysis: dict[str, Any], resources: list[dict[str, Any]], selections: list[dict[str, Any]],
                  figures: dict[str, Any], complete: bool, reconciliation_notes: list[dict[str, Any]] | None = None,
                  selector_candidate: str = "qwen_work") -> str:
    missing_configs = sorted(set(CONFIG_NAMES) - set(analysis["config_ids"]))
    partial_reason = "faltan configuraciones: " + ", ".join(CONFIG_NAMES[config] for config in missing_configs) if missing_configs else "faltan solicitudes terminales"
    lines = ["# Resultados del ciclo de experimentos", "", "**" + ("Ejecución terminal completa." if complete else "Informe parcial: " + partial_reason + ".") + "** Comparación descriptiva de modelos, formatos de respuesta y selección de casos con respuestas verificables. Estos resultados corresponden a benchmarks públicos y tareas sintéticas controladas.", ""]
    total_expected = sum(row["expected_requests"] for row in resources)
    total_received = sum(row["received_model_responses"] for row in resources)
    lines.append(f"Se recibieron **{total_received:,} respuestas de modelo de {total_expected:,} solicitudes previstas para las {len(resources)} configuraciones incluidas**. Los errores de formato de respuestas recibidas cuentan como incorrectos; los fallos de transporte y las llamadas ausentes quedan desconocidos.")
    lines.extend(["", "## Exactitud en las referencias", "", "Cada familia pesa lo mismo dentro de su dominio. Las 100 variantes sintéticas por dominio corresponden a 25 familias, con cuatro variantes cada una.", "",
                  "| Dominio | Configuración | Familias completas / previstas | Exactitud de la referencia completa |",
                  "|---|---|---:|---:|"])
    for row in analysis["group_summary"]:
        if row["role"] == "reference" and row["dataset"] in DOMAINS:
            lines.append(f"| {DOMAIN_NAMES[row['dataset']]} | {CONFIG_NAMES.get(row['config_id'], row['config_id'])} | {row['complete_families']} / {row['expected_families']} | {pct(row['full_population_family_mean_accuracy'])} |")
    if "reference_accuracy" in figures:
        lines.extend(["", f"![Exactitud de las referencias]({figures['reference_accuracy']['png']})", ""])
    lines.extend(["## Efecto de pedir trabajo explícito", "", "Δ = Qwen con trabajo explícito − Qwen directo. Un valor positivo favorece el primero; estas diferencias no establecen superioridad general.", "",
                  "| Dominio | Familias pareadas completas | Δ referencia (pp) | Familias con Δ distinto de cero |",
                  "|---|---:|---:|---:|"])
    for row in analysis["paired_comparisons"]:
        if row["role"] == "reference" and row["anchor_config"] == "qwen_direct" and row["candidate_config"] == "qwen_work":
            lines.append(f"| {DOMAIN_NAMES[row['dataset']]} | {row['paired_complete_families']} / {row['expected_families']} | {pp(row['full_population_family_mean_delta'], True)} | {row.get('paired_families_with_nonzero_delta', 'no registrado')} |")
    lines.extend(["", "Ambos formatos comparten el límite de salida del experimento. Pedir trabajo explícito cambia cómo se utiliza ese límite; la comparación debe leerse junto con tokens y motivos de finalización.", "", "## Referencias sintéticas por régimen", "",
                  "Los regímenes tienen 9 familias estacionarias, 8 más complejas y 8 con otra plantilla por dominio. Se comparan por separado. Usan familias distintas: una diferencia observada no identifica por sí sola el efecto causal de la plantilla."])
    if "controlled_regimes" in figures:
        lines.extend(["", f"![Regímenes sintéticos controlados]({figures['controlled_regimes']['png']})"])
    lines.extend(["", "| Dominio | Régimen | Δ Qwen trabajo − directo (pp) | Familias pareadas completas |", "|---|---|---:|---:|"])
    for row in analysis["paired_regime_comparisons"]:
        if row["role"] == "reference" and row["dataset"] in DOMAINS[2:] and row["anchor_config"] == "qwen_direct" and row["candidate_config"] == "qwen_work":
            lines.append(f"| {DOMAIN_NAMES[row['dataset']]} | {REGIME_NAMES[row['regime']]} | {pp(row['full_population_family_mean_delta'], True)} | {row['paired_complete_families']} / {row['expected_families']} |")
    # Show the target movement itself, independently of the selected estimate.
    # This prevents a reference error from being attributed only to selection.
    target_transfer = {}
    for row in analysis.get("selector_estimation", {}).get("paired_run_results", []):
        if row["dataset"] not in DOMAINS[2:] or row["anchor_config"] != "qwen_direct" or row["candidate_config"] not in {"qwen_work", "gemini_direct"}:
            continue
        for ref in row["references"]:
            key = (row["dataset"], row["candidate_config"], ref["regime"])
            values = (row["full_construction_delta"], ref["full_reference_delta"], ref["construction_to_reference_shift"])
            if key in target_transfer and target_transfer[key] != values:
                raise ValueError("Full target differs across selectors on the same frozen domain")
            target_transfer[key] = values
    if target_transfer:
        lines.extend(["", "### Cambio del objetivo completo", "", "Cada Δ usa Qwen directo como ancla. Esta tabla muestra cuánto cambia el objetivo completo entre construcción y referencia, antes de atribuir errores a un selector.", "",
                      "| Dominio | Candidato | Referencia | Δ construcción (pp) | Δ referencia (pp) | Referencia − construcción (pp) |",
                      "|---|---|---|---:|---:|---:|"])
        for (dataset, config, reg), (before, after, movement) in sorted(target_transfer.items()):
            lines.append(f"| {DOMAIN_NAMES[dataset]} | {CONFIG_NAMES[config]} | {REGIME_NAMES[reg]} | {pp(before, True)} | {pp(after, True)} | {pp(movement, True)} |")
    demonstration = next((row for row in analysis.get("selector_estimation", {}).get("paired_run_results", []) if row["dataset"] == "structured_records" and row["anchor_config"] == "qwen_direct" and row["candidate_config"] == "gemini_direct" and row["policy"] == "facility_location" and row["embedding_variant"] == "mpnet_head" and row["budget_families"] == 10), None)
    if demonstration:
        harder = next((ref for ref in demonstration["references"] if ref["regime"] == "more_complex"), None)
        if harder and demonstration["construction_absolute_error"] is not None and harder["reference_absolute_error"] is not None:
            lines.extend(["", f"Un ejemplo concreto es cobertura con MPNet inicio, 10 familias de registros estructurados: al estimar Gemini − Qwen directo, el error es **{pp(demonstration['construction_absolute_error'])} pp en construcción** y **{pp(harder['reference_absolute_error'])} pp en la referencia más compleja**. El objetivo completo pasa de {pp(demonstration['full_construction_delta'], True)} a {pp(harder['full_reference_delta'], True)} pp. Una buena aproximación de construcción no garantiza anticipar una distribución distinta."])
    lines.extend(["", "## Qué tan bien estiman el cambio los subconjuntos", "", f"Se usan los presupuestos medios fijados: GSM8K 50, BBH 25, cada dominio sintético 10 y mezcla 40 familias. El objetivo mostrado es estimar Δ de Qwen directo a {CONFIG_NAMES[selector_candidate]}. Cada columna de referencia conserva su régimen; no se promedian regímenes de forma implícita.", "",
                  "Las cifras de métodos aleatorios promedian 10 semillas de adquisición sobre los mismos resultados. Los métodos deterministas tienen una selección por representación. No son 10 estudios independientes y no se presentan intervalos de confianza."])
    if "selector_errors" in figures:
        lines.extend(["", f"![Errores de los selectores]({figures['selector_errors']['png']})"])
    constant_domains = sorted({row["dataset"] for row in analysis.get("selector_estimation", {}).get("paired_run_results", [])
                               if row["anchor_config"] == "qwen_direct" and row["candidate_config"] == selector_candidate and row.get("full_construction_family_delta_is_constant") is True})
    if constant_domains:
        lines.extend(["", "**Objetivos constantes:** " + ", ".join(DOMAIN_NAMES[domain] for domain in constant_domains) + ". En esos pools todos los selectores de pesos normalizados reproducen el mismo Δ; un error cero no permite distinguir su calidad."])
    for dataset in MIDDLE_BUDGETS:
        rows = [row for row in selections if row["dataset"] == dataset]
        if not rows:
            continue
        regimes = sorted({reg for row in rows for reg in row["reference_mean_absolute_errors"]}, key=regime_order)
        lines.extend(["", f"### {DOMAIN_NAMES[dataset]}", "",
                      "| Selector | Error construcción (pp) | " + " | ".join(REGIME_NAMES.get(reg, reg) + " (pp)" for reg in regimes) + " |",
                      "|---|---:|" + "---:|" * len(regimes)])
        for row in rows:
            lines.append("| " + selector_label(row) + " | " + pp(row["mean_construction_absolute_error"]) + " | " + " | ".join(pp(row["reference_mean_absolute_errors"].get(reg)) for reg in regimes) + " |")
    lines.extend(["", "El error frente a referencia reúne la aproximación del selector y la diferencia entre construcción y referencia. Hallar más casos fallidos es otro objetivo: el análisis completo guarda rendimiento de hallazgo sin ponderar y estimación poblacional HT por separado. La política dirigida usa longitud de entrada, no dificultad verificada.", "",
                  "## Cobertura y recursos", "", "| Configuración | Respuestas recibidas / previstas | Intentos registrados | Contabilidad API (US$) | Tokens totales informados | Intervalo observado (min) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in resources:
        tokens = row["token_usage_all_available_attempts"]["total_tokens"]
        token_text = "no disponible" if tokens["known_sum"] is None else f"{tokens['known_sum']:,}" + (f" (faltan {tokens['attempts_without_measurement']} intentos)" if tokens["attempts_without_measurement"] else "")
        cost = "incompleta" if row["ledger_accounted_usd"] is None else f"{row['ledger_accounted_usd']:.4f}"
        span = "no disponible" if row["observed_wall_span_seconds"] is None else f"{row['observed_wall_span_seconds'] / 60:.1f}"
        lines.append(f"| {CONFIG_NAMES.get(row['config_id'], row['config_id'])} | {row['received_model_responses']} / {row['expected_requests']} | {row['attempt_count']} | {cost} | {token_text} | {span} |")
    lines.extend(["", "| Configuración | JSON/esquema externo inválido | Formato de respuesta rechazado por el corrector |", "|---|---:|---:|"])
    for row in resources:
        invalid_outer = sum(count for status, count in row["format_status_counts"].items() if status not in {"valid", "not_received"})
        lines.append(f"| {CONFIG_NAMES.get(row['config_id'], row['config_id'])} | {invalid_outer} | {row['grader_answer_format_failures']} |")
    lines.append("\nLas columnas son distintas: un JSON válido puede contener una cadena numérica con texto extra que el corrector rechaza. Los dos casos reconciliados de Qwen directo son fallos del parser externo, no necesariamente todos los fallos de formato de respuesta.")
    lines.extend(["", "La contabilidad conserva todos los intentos y reservas pendientes cuando sus ledgers están disponibles; puede incluir cotas conservadoras, no una factura. US$0 en local significa ausencia de cargo API: hardware y electricidad no están valorizados. El intervalo va del primer inicio al último término e incluye solapamientos, pausas y posibles reintentos; no es tiempo exclusivo de GPU. Los proveedores usan tokenizadores distintos.", "",
                  "El servidor local usa batching continuo con cuatro slots y caché de prefijos. Las latencias incluyen colas y efectos de caché; no constituyen un benchmark de throughput por llamada en despliegue. Semillas y hashes facilitan la trazabilidad, pero no garantizan salidas idénticas bit a bit. Los resultados observados se conservan como outcomes fijos del experimento.", "",
                  "El mismo presupuesto de familias puede implicar diferente número de respuestas en la mezcla, porque las familias sintéticas tienen cuatro variantes. No se midió tiempo de revisores humanos.", "", "## Alcance de los hallazgos", "",
                  "Estos resultados permiten estudiar selección, aproximación de métricas y robustez ante variantes con respuesta verificable. Los benchmarks públicos pueden estar contaminados por preentrenamiento; las tareas procedurales cubren una distribución diseñada. No prueban predicción del tráfico de producción un mes después, no constituyen un golden set validado por humanos y sus referencias se usan exclusivamente para evaluación.", "",
                  "Los JSON de síntesis y las tablas CSV conservan cobertura, recursos y errores por régimen. Las figuras incluyen PNG y SVG exportables.", ""])
    if reconciliation_notes:
        lines.extend(["## Procesamiento de respuestas truncadas", ""])
        for note in reconciliation_notes:
            lines.append(f"Se usó el archivo canónico documentado en [{Path(note['path']).name}]({Path(note['path']).resolve()}). La reconciliación revisó {note['changed_terminal_count']} resultados terminales: el servidor había convertido salidas del modelo truncadas por presupuesto en errores HTTP. Se conserva la primera salida de modelo completada, incluso inválida; un reintento posterior correcto no sustituye ese fallo. Los intentos y su contabilidad originales permanecen conservados ({note['source_attempt_count']} intentos para {note['canonical_outcome_count']} resultados).")
        lines.append("")
    return "\n".join(lines)


def write_selector_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["dataset", "budget_families", "policy", "embedding_variant", "acquisition_runs", "target", "mean_absolute_error_pp"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for target, value in {"construction": row["mean_construction_absolute_error"], **row["reference_mean_absolute_errors"]}.items():
                writer.writerow({**{key: row.get(key) for key in fields[:5]}, "target": target,
                                 "mean_absolute_error_pp": None if value is None else 100 * value})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--attempts", action="append", default=[], metavar="CONFIG=PATH", help="Original attempt log when canonical output has a different basename")
    parser.add_argument("--reconciliation", type=Path, action="append", default=[])
    parser.add_argument("--selector-candidate", choices=list(CONFIG_NAMES), default="qwen_work")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Label the report partial rather than requiring every terminal request")
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    overrides = {}
    for value in args.attempts:
        config, path = value.split("=", 1)
        if config in overrides:
            raise ValueError("Duplicate attempt override")
        overrides[config] = Path(path)
    resources, provenance = summarize_resources(args.responses, analysis, overrides)
    complete = all(row["terminal_requests"] == row["expected_requests"] for row in resources) and set(analysis["config_ids"]) == set(CONFIG_NAMES)
    if not complete and not args.allow_partial:
        raise ValueError("Terminal execution is incomplete; wait for completion or explicitly use --allow-partial")
    if args.selector_candidate not in analysis["config_ids"] and not args.allow_partial:
        raise ValueError("Requested selector comparison is absent from the supplied analysis")
    selections = middle_budget_rows(analysis, args.selector_candidate)
    reconciliation_notes = []
    for path in args.reconciliation:
        record = json.loads(path.read_text())
        if record.get("canonical_sha256") not in {digest(source) for source in args.responses}:
            raise ValueError("Reconciliation canonical hash does not match a supplied response file")
        if record.get("original_attempts_sha256") not in {digest(source) for source in overrides.values()}:
            raise ValueError("Reconciliation original attempts must be supplied explicitly and hash-match")
        reconciliation_notes.append({"path": str(path), "sha256": digest(path),
                                     "changed_terminal_count": len(record["changed_terminals"]),
                                     "source_attempt_count": record["number_source_attempts"],
                                     "canonical_outcome_count": record["number_canonical_outcomes"]})
    args.output.mkdir(parents=True, exist_ok=True)
    figures = make_figures(analysis, selections, args.output, args.selector_candidate)
    (args.output / "cycle_report.md").write_text(render_report(analysis, resources, selections, figures, complete, reconciliation_notes, args.selector_candidate))
    write_selector_csv(args.output / "selector_middle_budget.csv", selections)
    summary = {"status": "complete_terminal_execution" if complete else "partial_terminal_execution", "resources": resources,
               "middle_budget_selectors": selections, "figures": figures, "response_provenance": provenance, "reconciliation": reconciliation_notes,
               "analysis_source": {"path": str(args.analysis), "sha256": digest(args.analysis)}, "script_sha256": digest(Path(__file__)),
               "interpretation": "Descriptive public benchmark and controlled synthetic evaluation; no production forecasting, human gold, confidence-interval or equal-cost claim."}
    (args.output / "cycle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "report": str((args.output / "cycle_report.md").resolve()),
                      "figures": len(figures), "middle_budget_rows": len(selections)}))


if __name__ == "__main__":
    main()
