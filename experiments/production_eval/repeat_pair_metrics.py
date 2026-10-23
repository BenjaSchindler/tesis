"""Pure paired diagnostics with explicit observed/valid/nonbinding denominators."""
from collections import Counter
from statistics import mean

from experiments.production_eval.analyze_outcomes_v2 import domain, family_members, grade_response


def positive_usage(response):
    value = (response.get("usage") or {}).get("completion_tokens")
    return value if type(value) is int and value >= 0 else None


def pair_metrics(rows, responses_a, responses_b):
    pairs = []
    for row in rows:
        identifier = row["example_id"]
        a, b = responses_a.get(identifier, {}), responses_b.get(identifier, {})
        ga, gb = grade_response(row, a or None), grade_response(row, b or None)
        both = ga["observed"] and gb["observed"]
        va, vb = ga["format_valid"], gb["format_valid"]
        transition = None if not both else "both_valid" if va and vb else "valid_to_invalid" if va else "invalid_to_valid" if vb else "both_invalid"
        pred_changed = ga["canonical_prediction"] != gb["canonical_prediction"] if transition == "both_valid" else None
        correctness = ("C" if ga["correct"] else "W") + ("C" if gb["correct"] else "W") if both else None
        ta, tb = positive_usage(a), positive_usage(b)
        text_a, text_b = a.get("raw_output_text"), b.get("raw_output_text")
        text_received = isinstance(text_a, str) and isinstance(text_b, str)
        nonbinding = both and a.get("finish_reason") == b.get("finish_reason") == "stop" and ta is not None and tb is not None and max(ta, tb) < 1024
        pairs.append({"example_id": identifier, "family_id": row["family_id"], "dataset": domain(row),
                      "observed_a": ga["observed"], "observed_b": gb["observed"], "both_observed": both,
                      "correct_a": ga["correct"], "correct_b": gb["correct"],
                      "answer_format_transition": transition, "canonical_prediction_changed": pred_changed,
                      "correctness_transition": correctness, "correctness_changed": correctness in ("CW", "WC") if both else None,
                      "raw_text_comparable": text_received,
                      "raw_text_changed": text_a != text_b if text_received else None,
                      "finish_reason_a": a.get("finish_reason"), "finish_reason_b": b.get("finish_reason"),
                      "completion_tokens_a": ta, "completion_tokens_b": tb,
                      "confirmed_length_finish_a": a.get("finish_reason") == "length",
                      "confirmed_length_finish_b": b.get("finish_reason") == "length",
                      "at_or_above_1024_a": ta >= 1024 if ta is not None else None,
                      "at_or_above_1024_b": tb >= 1024 if tb is not None else None,
                      "both_stop_under_1024": bool(nonbinding)})
    summaries = []
    indexed = {pair["example_id"]: pair for pair in pairs}
    for group in sorted({domain(row) for row in rows}) + ["balanced_mixture"]:
        selected = [row for row in rows if group == "balanced_mixture" or domain(row) == group]
        family_pairs = []
        for family, members in family_members(selected).items():
            descendants = [indexed[row["example_id"]] for row in members]
            complete = all(pair["both_observed"] for pair in descendants)
            a = mean(int(pair["correct_a"]) for pair in descendants) if complete else None
            b = mean(int(pair["correct_b"]) for pair in descendants) if complete else None
            family_pairs.append({"family_id": family, "descendants": len(descendants), "paired_complete": complete,
                                 "accuracy_a": a, "accuracy_b": b, "delta": b - a if complete else None})
        complete = [pair for pair in family_pairs if pair["paired_complete"]]
        group_pairs = [indexed[row["example_id"]] for row in selected]
        def summarize(subset):
            observed = [pair for pair in subset if pair["both_observed"]]
            formats = Counter(pair["answer_format_transition"] for pair in observed)
            correction = Counter(pair["correctness_transition"] for pair in observed)
            comparable = [pair for pair in subset if pair["raw_text_comparable"]]
            valid = [pair for pair in observed if pair["answer_format_transition"] == "both_valid"]
            return {"rows": len(subset), "both_observed": len(observed),
                    "unknown_a": sum(not pair["observed_a"] for pair in subset), "unknown_b": sum(not pair["observed_b"] for pair in subset),
                    "format_transitions": dict(formats), "correctness_transitions": dict(correction),
                    "correctness_disagreements": correction["CW"] + correction["WC"],
                    "correctness_agreements": correction["CC"] + correction["WW"],
                    "both_valid_prediction_pairs": len(valid),
                    "changed_canonical_predictions_both_valid": sum(pair["canonical_prediction_changed"] for pair in valid),
                    "raw_text_comparable_pairs": len(comparable),
                    "raw_text_changed": sum(pair["raw_text_changed"] for pair in comparable),
                    "usage_unknown_a": sum(pair["completion_tokens_a"] is None for pair in subset),
                    "usage_unknown_b": sum(pair["completion_tokens_b"] is None for pair in subset)}
        summaries.append({"dataset": group, "all_pairs": summarize(group_pairs),
                          "conditional_both_stop_under_1024": summarize([pair for pair in group_pairs if pair["both_stop_under_1024"]]),
                          "expected_families": len(family_pairs), "complete_families": len(complete),
                          "family_accuracy_a": mean(pair["accuracy_a"] for pair in complete) if complete else None,
                          "family_accuracy_b": mean(pair["accuracy_b"] for pair in complete) if complete else None,
                          "conditional_complete_family_delta": mean(pair["delta"] for pair in complete) if complete else None,
                          "full_family_delta": mean(pair["delta"] for pair in complete) if complete and len(complete) == len(family_pairs) else None,
                          "families_improve": sum(pair["delta"] > 0 for pair in complete),
                          "families_regress": sum(pair["delta"] < 0 for pair in complete),
                          "families_tie": sum(pair["delta"] == 0 for pair in complete), "family_deltas": family_pairs})
    return {"pairs": pairs, "summaries": summaries}
