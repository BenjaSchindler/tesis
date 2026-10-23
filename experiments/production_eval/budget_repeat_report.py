"""Descriptive repeat diagnostics; token IDs are re-tokenized returned content."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.budget_repeat_control import load_control
from experiments.production_eval.budget_sensitivity import DEFAULT_BASE, load_prepared, sha
from experiments.production_eval.repeat_pair_metrics import pair_metrics


def read_bound(path, config, expected, allow_extra=False):
    result = {}
    for response in inference.read_jsonl(path):
        identifier = response["example_id"]
        if identifier not in expected:
            if allow_extra: continue
            raise ValueError("Response outside frozen subset")
        item = expected[identifier]
        if identifier in result or response["config_id"] != config["config_id"] or response["input_hash"] != item["input_hash"] or response["cache_key"] != item["cache_key"]:
            raise ValueError("Duplicate or changed response identity")
        result[identifier] = response
    attempts = [r for r in inference.read_jsonl(Path(str(path) + ".attempts.jsonl")) if r["example_id"] in expected]
    first_completed = {}
    attempts_by_call = {}
    for attempt in attempts:
        identifier = attempt["example_id"]
        if attempt["call_id"] in attempts_by_call or attempt["cache_key"] != expected[identifier]["cache_key"]:
            raise ValueError("Invalid attempt binding")
        attempts_by_call[attempt["call_id"]] = attempt
        if attempt["status"] == "success":
            first_completed.setdefault(identifier, attempt)
    for identifier, response in result.items():
        if attempts_by_call.get(response["call_id"]) != response:
            raise ValueError("Terminal differs from preserved attempt")
        if identifier in first_completed and response != first_completed[identifier]:
            raise ValueError("First completed model output was not retained")
        if response.get("raw_response") is not None and response["response_sha256"] != inference.digest(inference.canonical(response["raw_response"])):
            raise ValueError("Raw response hash mismatch")
    return result, attempts


def usage(attempts):
    summary = {"attempts": len(attempts), "api_usd": sum(r["charged_nanodollars"] for r in attempts) / 1e9}
    for field in ("prompt_tokens", "completion_tokens"):
        values = [(r.get("usage") or {}).get(field) for r in attempts]
        known = [value for value in values if type(value) is int and value >= 0]
        summary[field] = {"known": len(known), "unknown": len(values) - len(known), "sum_known": sum(known),
                          "mean_known": mean(known) if known else None, "median_known": median(known) if known else None}
    return summary


def common_prefix(a, b):
    index = 0
    while index < min(len(a), len(b)) and a[index] == b[index]:
        index += 1
    return index


def token_cache(path, responses, config, allow_tokenize):
    cached = {}
    tokenizer_identity = {"model_revision": config["model_revision"], "add_special": False, "parse_special": False}
    for item in inference.read_jsonl(path):
        key = item["key"]
        if key in cached or item["tokenizer_identity"] != tokenizer_identity or item["token_sha256"] != inference.digest(inference.canonical(item["token_ids"])):
            raise ValueError("Invalid tokenization cache")
        cached[key] = item
    texts = {inference.digest(text): text for mapping in responses.values() for row in mapping.values()
             if isinstance(text := row.get("raw_output_text"), str)}
    missing = [(digest, text) for digest, text in texts.items() if digest not in cached]
    if missing and not allow_tokenize:
        raise ValueError("Missing returned-text tokenizations; run once with --tokenize using local CPU endpoint")
    if missing:
        import requests
        with requests.Session() as session:
            session.trust_env = False
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            for digest, text in missing:
                result = session.post(config["base_url"].rstrip("/") + "/tokenize", json={"content": text, "add_special": False, "parse_special": False}, timeout=(10, 60))
                result.raise_for_status()
                tokens = result.json()["tokens"]
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise ValueError("Unexpected tokenizer response")
                item = {"key": digest, "raw_text_sha256": digest, "tokenizer_identity": tokenizer_identity,
                        "token_ids": tokens, "token_sha256": inference.digest(inference.canonical(tokens)),
                        "created_at_utc": inference.utc_now(), "provenance": "Re-tokenized returned content, not the actual emitted generation token sequence."}
                inference.RunJournal.append(path, item)
                cached[digest] = item
    return cached


def augment_hashes(metrics, a, b, cache):
    for pair in metrics["pairs"]:
        if not pair["raw_text_comparable"]:
            pair["hash_comparison"] = None
            continue
        identifier = pair["example_id"]
        ta, tb = a[identifier]["raw_output_text"], b[identifier]["raw_output_text"]
        ca, cb = cache[inference.digest(ta)], cache[inference.digest(tb)]
        ia, ib = ca["token_ids"], cb["token_ids"]
        pair["hash_comparison"] = {
            "text_sha256_a": ca["raw_text_sha256"], "text_sha256_b": cb["raw_text_sha256"],
            "retokenized_token_sha256_a": ca["token_sha256"], "retokenized_token_sha256_b": cb["token_sha256"],
            "retokenized_token_sequence_equal": ca["token_sha256"] == cb["token_sha256"],
            "retokenized_token_count_a": len(ia), "retokenized_token_count_b": len(ib),
            "common_prefix_characters": common_prefix(ta, tb), "common_prefix_retokens": common_prefix(ia, ib),
            "first_128_characters_sha256_a": inference.digest(ta[:128]),
            "first_128_characters_sha256_b": inference.digest(tb[:128]),
            "both_at_least_32_retokens": min(len(ia), len(ib)) >= 32,
            "first_32_retokens_sha256_a": inference.digest(inference.canonical(ia[:32])),
            "first_32_retokens_sha256_b": inference.digest(inference.canonical(ib[:32]))}
    for summary in metrics["summaries"]:
        selected = [p for p in metrics["pairs"] if summary["dataset"] == "balanced_mixture" or p["dataset"] == summary["dataset"]]
        for key, pairs in (("all_pairs", selected), ("conditional_both_stop_under_1024", [p for p in selected if p["both_stop_under_1024"]])):
            hashes = [pair["hash_comparison"] for pair in pairs if pair["hash_comparison"] is not None]
            long = [item for item in hashes if item["both_at_least_32_retokens"]]
            summary[key]["returned_text_hash_comparison"] = {
                "comparable_pairs": len(hashes), "identical_retokenized_sequences": sum(item["retokenized_token_sequence_equal"] for item in hashes),
                "equal_first_128_character_hashes": sum(item["first_128_characters_sha256_a"] == item["first_128_characters_sha256_b"] for item in hashes),
                "both_at_least_32_retokens": len(long),
                "equal_first_32_retoken_hashes_when_both_long_enough": sum(item["first_32_retokens_sha256_a"] == item["first_32_retokens_sha256_b"] for item in long),
                "median_common_prefix_retokens": median(item["common_prefix_retokens"] for item in hashes) if hashes else None}


def report(study, allow_tokenize=False):
    plan, rows = load_prepared(study)
    configs = {"original1024": plan["baseline"]["config"], **{f"budget{budget}": plan["variants"][str(budget)]["config"] for budget in (2048, 4096)}}
    paths = {"original1024": Path(plan["baseline"]["output_path"]), **{f"budget{budget}": study / f"responses_b{budget}.jsonl" for budget in (2048, 4096)}}
    amendments = {}
    for repeat in (1, 2):
        directory = study / f"repeat_control_{repeat}"
        if not (directory / "amendment.json").exists(): continue
        control, control_rows = load_control(directory)
        if control_rows != rows: raise ValueError("Repeat changed inputs")
        name = f"repeat1024_{repeat}"
        configs[name], paths[name], amendments[name] = control["config"], directory / "responses.jsonl", control
    responses, sources, usages = {}, {}, {}
    for name, path in paths.items():
        expected = {row["example_id"]: inference.prepare(row, configs[name]) for row in rows}
        indexed, attempts = read_bound(path, configs[name], expected, name == "original1024")
        responses[name] = indexed
        sources[name] = {"path": str(path), "sha256": sha(path) if path.exists() else None,
                         "attempts_sha256": sha(str(path) + ".attempts.jsonl") if Path(str(path) + ".attempts.jsonl").exists() else None,
                         "first_completed_model_output_check": "passed", "terminal_rows": len(indexed)}
        usages[name] = usage(attempts)
        if name in amendments:
            if any(row["started_at_utc"] <= amendments[name]["created_at_utc"] for row in attempts):
                raise ValueError("Control call predates amendment")
            invocations = inference.read_jsonl(path.parent / "invocations.jsonl")
            usages[name]["standalone_wall_seconds"] = sum(row["wall_seconds_this_invocation"] for row in invocations)
    cache_path = study / "returned_text_tokens.jsonl"
    cache = token_cache(cache_path, responses, configs["original1024"], allow_tokenize)
    comparisons = {}
    for name in configs:
        if name == "original1024": continue
        metrics = pair_metrics(rows, responses["original1024"], responses[name])
        augment_hashes(metrics, responses["original1024"], responses[name], cache)
        comparisons[f"original1024_vs_{name}"] = metrics
    if "repeat1024_2" in responses:
        metrics = pair_metrics(rows, responses["repeat1024_1"], responses["repeat1024_2"])
        augment_hashes(metrics, responses["repeat1024_1"], responses["repeat1024_2"], cache)
        comparisons["repeat1024_1_vs_repeat1024_2"] = metrics
    result = {"created_at_utc": inference.utc_now(), "scope": "Descriptive execution-repeat diagnostic on the same post hoc construction pool.",
              "study_plan_sha256": sha(study / "plan.json"), "amendments": {name: {"created_at_utc": value["created_at_utc"], "rationale": value["rationale"]} for name, value in amendments.items()},
              "source_outputs": sources, "usage": usages, "comparisons": comparisons,
              "token_cache_sha256": sha(cache_path), "tokenization_provenance": "Server /tokenize applied to returned raw_output_text with add_special=false, parse_special=false. These are re-tokenized content IDs, not captured emitted token IDs, and can differ from usage counts.",
              "limitations": ["Exact HTTP generation bodies are equal between original1024 and repeats; config bookkeeping changes only cache identity.",
                              "max_tokens is not included in the model-visible prompt. Nonbinding differences alone do not identify their cause; identical-config disagreements measure observed execution variation.",
                              "Original1024 was interleaved with 1350 requests; repeats and budget variants use this 250-request subset. Runtime scheduling/context cache conditions were not randomized or standardized.",
                              "Both-stop/under1024 conditions on observed outcomes and excludes unknown usage. It is a diagnostic subset, not the primary target.",
                              "Invalid formats are observed wrong outcomes; both-invalid pairs are not counted as prediction agreement. Unknown outcomes are not imputed.",
                              "A repeat measures pairwise disagreement, not a sampling distribution of run variance. Its score difference cannot be subtracted to identify a causal budget effect.",
                              "The 100-family mixture gives each of four domains one-quarter weight. Raw 250-row agreement weights synthetic domains more heavily."]}
    (study / "repeat_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Control de repetición con 1.024 tokens", "", "Mismos 250 casos y 100 familias de construcción; análisis posterior de variación de ejecución. No es validación independiente.", "",
             "| Comparación | Exactitud familias A / B | Pares observados | Corrección cambia | Ambos formatos válidos: predicción cambia / pares | Texto cambia / pares |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in comparisons.items():
        summary = next(s for s in metrics["summaries"] if s["dataset"] == "balanced_mixture")
        all_pairs = summary["all_pairs"]
        lines.append(f"| {name} | {100*summary['family_accuracy_a']:.1f}% / {100*summary['family_accuracy_b']:.1f}% | {all_pairs['both_observed']} | {all_pairs['correctness_disagreements']} | {all_pairs['changed_canonical_predictions_both_valid']} / {all_pairs['both_valid_prediction_pairs']} | {all_pairs['raw_text_changed']} / {all_pairs['raw_text_comparable_pairs']} |")
    lines += ["", "| Comparación | Ambos terminan stop y <1024: pares | Cambios de corrección | Cambios de predicción / ambos válidos | Secuencias retokenizadas idénticas | Prefijo de 32 retokens igual / pares de longitud suficiente |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in comparisons.items():
        sub = next(s for s in metrics["summaries"] if s["dataset"] == "balanced_mixture")["conditional_both_stop_under_1024"]
        hashes = sub["returned_text_hash_comparison"]
        lines.append(f"| {name} | {sub['rows']} | {sub['correctness_disagreements']} | {sub['changed_canonical_predictions_both_valid']} / {sub['both_valid_prediction_pairs']} | {hashes['identical_retokenized_sequences']} | {hashes['equal_first_32_retoken_hashes_when_both_long_enough']} / {hashes['both_at_least_32_retokens']} |")
    lines += ["", "Los tokens de esta tabla proceden de volver a tokenizar el texto devuelto; no son una captura de los IDs realmente emitidos. Los conteos de uso reportados se conservan por separado, con sus desconocidos.", "",
              "La repetición envía exactamente el mismo cuerpo HTTP al modelo. El límite de salida no forma parte del prompt. Los desacuerdos de una repetición demuestran variación observada de ejecución bajo esta configuración, pero no identifican su mecanismo ni permiten descontarla para estimar un efecto causal del presupuesto.", "",
              "El subconjunto stop/<1024 depende de los resultados y excluye usos desconocidos. Las comparaciones principales conservan todos los resultados observados y exigen familias completas; un formato inválido es un fallo observado, no un dato faltante. Ambos formatos inválidos no se cuentan como respuestas coincidentes.", "",
              "El JSON adjunto incluye transiciones CC/CW/WC/WW, métricas por dominio, cada delta de familia, hashes y prefijos por par, costos y datos de auditoría. La exactitud por familia da un cuarto del peso a cada dominio; las coincidencias sobre 250 filas pesan más los dominios sintéticos.", ""]
    (study / "repeat_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_BASE / "budget_sensitivity")
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    result = report(args.study.resolve(), args.tokenize)
    print(inference.canonical({"comparisons": list(result["comparisons"]), "outputs": {key: value["terminal_rows"] for key, value in result["source_outputs"].items()}}))
