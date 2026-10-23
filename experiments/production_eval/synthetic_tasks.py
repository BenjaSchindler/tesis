"""Reproducible, verifier-backed tasks for controlled synthetic shift studies.

This is a procedural task distribution, not evidence of real production drift.
Each split contains 25 independent families per domain and four descendants per
family. Descendants must never be independently split or treated as independent
replicates. No language-model labels, network access or generated code execution
are used. Answers are cross-checked by separate Python and SQL implementations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "controlled_tasks_v1"
DEFAULT_SEED = 20260914
DOMAINS = ("structured_records", "inventory_tracking")
VARIANTS = ("base", "paraphrase", "counterfactual", "counterfactual_paraphrase")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derived_seed(*parts: Any) -> int:
    return int(digest("|".join(str(part) for part in parts))[:16], 16)


def records_oracle(problem: dict[str, Any]) -> dict[str, Any]:
    """Direct row-wise implementation of the inclusion predicate and totals."""
    selected = [r for r in problem["records"] if r["team"] == problem["target_team"]
                and r["active"] is True and r["void"] is False]
    return {"selected_ids": sorted(r["id"] for r in selected),
            "total_units": sum(r["units"] for r in selected),
            "total_value": sum(r["units"] * r["unit_price"] for r in selected)}


def records_sql_oracle(problem: dict[str, Any]) -> dict[str, Any]:
    """Independent relational implementation; query is static and parameterized."""
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE records (id TEXT, team TEXT, active INT, void INT, units INT, price INT)")
        db.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                       [(r["id"], r["team"], int(r["active"]), int(r["void"]), r["units"], r["unit_price"])
                        for r in problem["records"]])
        ids = [r[0] for r in db.execute(
            "SELECT id FROM records WHERE team = ? AND active = 1 AND void = 0 ORDER BY id",
            (problem["target_team"],))]
        units, value = db.execute(
            "SELECT COALESCE(SUM(units), 0), COALESCE(SUM(units * price), 0) "
            "FROM records WHERE team = ? AND active = 1 AND void = 0",
            (problem["target_team"],)).fetchone()
    return {"selected_ids": ids, "total_units": units, "total_value": value}


def inventory_oracle(problem: dict[str, Any]) -> dict[str, Any]:
    """Sequential simulation including a nonnegative stock invariant."""
    state = {r["item"]: {"east": r["east"], "west": r["west"]} for r in problem["initial"]}
    for op in problem["operations"]:
        if op["cancelled"]:
            continue
        target = state[op["item"]]
        if op["kind"] == "receive":
            target[op["location"]] += op["quantity"]
        elif op["kind"] == "ship":
            target[op["location"]] -= op["quantity"]
        elif op["kind"] == "move":
            target[op["location"]] -= op["quantity"]
            target[op["destination"]] += op["quantity"]
        else:
            raise ValueError("Unknown operation")
        if min(target.values()) < 0:
            raise ValueError("Generated operation makes stock negative")
    stock = [{"item": item, **state[item]} for item in sorted(state)]
    return {"stock": stock, "total_units": sum(r["east"] + r["west"] for r in stock)}


def inventory_sql_oracle(problem: dict[str, Any]) -> dict[str, Any]:
    """Independent signed-event ledger, aggregated without sequential mutation."""
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE initial (item TEXT, east INT, west INT)")
        db.execute("CREATE TABLE operations (kind TEXT, item TEXT, location TEXT, destination TEXT, quantity INT, cancelled INT)")
        db.executemany("INSERT INTO initial VALUES (?, ?, ?)",
                       [(r["item"], r["east"], r["west"]) for r in problem["initial"]])
        db.executemany("INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?)",
                       [(r["kind"], r["item"], r["location"], r.get("destination"), r["quantity"], int(r["cancelled"]))
                        for r in problem["operations"]])
        result = db.execute("""
            WITH events AS (
              SELECT item, 'east' location, east delta FROM initial
              UNION ALL SELECT item, 'west', west FROM initial
              UNION ALL SELECT item, location, quantity FROM operations WHERE kind='receive' AND cancelled=0
              UNION ALL SELECT item, location, -quantity FROM operations WHERE kind IN ('ship','move') AND cancelled=0
              UNION ALL SELECT item, destination, quantity FROM operations WHERE kind='move' AND cancelled=0
            )
            SELECT item, SUM(CASE WHEN location='east' THEN delta ELSE 0 END),
                         SUM(CASE WHEN location='west' THEN delta ELSE 0 END)
            FROM events GROUP BY item ORDER BY item
        """).fetchall()
    stock = [{"item": item, "east": east, "west": west} for item, east, west in result]
    return {"stock": stock, "total_units": sum(east + west for _, east, west in result)}


ORACLES = {"structured_records": (records_oracle, records_sql_oracle),
           "inventory_tracking": (inventory_oracle, inventory_sql_oracle)}


def build_records(data_seed: int, entity_seed: int, regime: str) -> tuple[dict, dict, dict]:
    rng = random.Random(data_seed)
    prefix = digest(str(entity_seed))[:10]
    count = 28 if regime == "more_complex" else 8
    teams = [f"team_{prefix}_{i}" for i in range(3)]
    records = []
    for i in range(count):
        records.append({"id": f"row_{prefix}_{i:02d}", "team": rng.choice(teams),
                        "active": rng.choice([True, True, False]),
                        "void": rng.choice([True, False, False]),
                        "units": rng.randint(1, 17), "unit_price": rng.randint(2, 23)})
    # Exactly qualifying, inactive, voided and wrong-team examples are guaranteed.
    for index, (team, active, void) in enumerate([
            (teams[0], True, False), (teams[0], False, False),
            (teams[0], True, True), (teams[1], True, False)]):
        records[index].update(team=team, active=active, void=void)
    target_id = records[0]["id"]
    rng.shuffle(records)
    base = {"target_team": teams[0], "records": records}
    changed = copy.deepcopy(base)
    target = next(r for r in changed["records"] if r["id"] == target_id)
    before = target["units"]
    target["units"] += 1
    mutation = {"entity_id": target_id, "field": "units", "before": before,
                "after": target["units"], "expected_total_units_delta": 1,
                "expected_total_value_delta": target["unit_price"]}
    return base, changed, mutation


def build_inventory(data_seed: int, entity_seed: int, regime: str) -> tuple[dict, dict, dict]:
    rng = random.Random(data_seed)
    prefix = digest(str(entity_seed))[:10]
    count, steps = (7, 36) if regime == "more_complex" else (3, 10)
    initial = [{"item": f"item_{prefix}_{i:02d}", "east": rng.randint(15, 30),
                "west": rng.randint(15, 30)} for i in range(count)]
    state = {r["item"]: {"east": r["east"], "west": r["west"]} for r in initial}
    operations = []
    for i in range(steps):
        item = rng.choice(initial)["item"]
        location = rng.choice(["east", "west"])
        # Prefix guarantees one of each operation and at least one cancellation.
        kind = ["receive", "ship", "move"][i] if i < 3 else rng.choice(["receive", "ship", "move"])
        quantity = rng.randint(1, max(1, min(9, state[item][location])))
        if kind != "receive" and state[item][location] == 0:
            kind = "receive"
        op = {"step": i + 1, "kind": kind, "item": item, "location": location,
              "quantity": quantity, "cancelled": i == 3 or (i > 3 and rng.random() < .25)}
        if kind == "move":
            op["destination"] = "west" if location == "east" else "east"
        operations.append(op)
        if not op["cancelled"]:
            if kind == "receive":
                state[item][location] += quantity
            else:
                state[item][location] -= quantity
                if kind == "move":
                    state[item][op["destination"]] += quantity
    rng.shuffle(initial)
    base = {"initial": initial, "operations": operations}
    changed = copy.deepcopy(base)
    target = changed["initial"][0]
    before = target["east"]
    target["east"] += 3
    mutation = {"entity_id": target["item"], "field": "initial.east", "before": before,
                "after": target["east"], "expected_total_units_delta": 3}
    return base, changed, mutation


def render(domain: str, problem: dict, paraphrase: bool, regime: str, template_seed: int) -> str:
    rng = random.Random(template_seed)
    if domain == "structured_records":
        rule = (
            "Retain a record exactly when its team equals target_team, active is true, and void is false. "
            "For retained records return selected_ids in ascending lexical order, total_units as the sum of units, "
            "and total_value as the sum of units multiplied by unit_price. Exclude all other records."
            if not paraphrase else
            "Discard rows from other teams, inactive rows, and voided rows. Among the remaining rows matching "
            "target_team, alphabetize the IDs into selected_ids; add their units into total_units; add each "
            "row's units times its unit_price into total_value."
        )
        schema = '{"selected_ids":["id",...],"total_units":integer,"total_value":integer}'
        label = "Source records"
    else:
        rule = (
            "Start from initial stock and process operations in step order. A cancelled operation has no effect. "
            "receive adds quantity at location; ship subtracts quantity at location; move subtracts quantity "
            "at location and adds it at destination for the same item. Return stock for every item, sorted "
            "lexically by item, with its final east and west counts. total_units is the sum across all items "
            "and both locations."
            if not paraphrase else
            "Track each item's east and west stock from the initial values. Apply the numbered steps in sequence, "
            "skipping every step whose cancelled flag is true. For receive, increase the named location; for ship, "
            "decrease it; for move, take quantity from location and place that quantity in destination. Output "
            "all final item counts in stock, ordered alphabetically by item, plus total_units over both locations."
        )
        schema = '{"stock":[{"item":"id","east":integer,"west":integer},...],"total_units":integer}'
        label = "Inventory ledger"
    rule += " Use integer arithmetic. Output one JSON object with exactly the requested keys and no commentary or markdown."
    # Independent template RNG controls innocuous label wording/serialization.
    label = rng.choice([label, "Input data", "Complete input"])
    data = json.dumps(problem, sort_keys=False, ensure_ascii=False, indent=None if paraphrase else 2)
    if regime == "changed_template":
        return f"DATA FIRST\n<{label.lower().replace(' ', '_')}>\n{data}\n</{label.lower().replace(' ', '_')}>\n\n" \
               f"REQUIRED RESULT SHAPE\n{schema}\n\nPROCESSING RULES\n{rule}"
    return f"{rule}\n\nRequired JSON shape: {schema}\n\n{label}:\n{data}"


def json_exact_grade(response: str, expected_answer: str) -> bool:
    """Strict JSON-value grade; no fences, duplicate keys, NaN or bool-as-int."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("Non-finite constant")

    try:
        actual = json.loads(response, object_pairs_hook=object_pairs, parse_constant=reject_constant)
        expected = json.loads(expected_answer)
    except (ValueError, TypeError, json.JSONDecodeError):
        return False

    def same(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b

    return same(actual, expected)


def generate(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rows = []
    for domain in DOMAINS:
        for split in ("construction", "reference"):
            for family_index in range(25):
                regime = "stationary" if split == "construction" or family_index < 9 else \
                         "more_complex" if family_index < 17 else "changed_template"
                family_id = digest(f"{VERSION}|{seed}|{domain}|{split}|{family_index}")
                seeds = {kind: derived_seed(family_id, kind) for kind in ("data", "entities", "template_family")}
                builder = build_records if domain == "structured_records" else build_inventory
                base, changed, mutation = builder(seeds["data"], seeds["entities"], regime)
                base_id = digest(f"{family_id}|base")
                for variant in VARIANTS:
                    counterfactual = variant.startswith("counterfactual")
                    paraphrase = variant.endswith("paraphrase")
                    problem = changed if counterfactual else base
                    input_text = render(domain, problem, paraphrase, regime, seeds["template_family"])
                    primary, secondary = ORACLES[domain]
                    answer = primary(problem)
                    if answer != secondary(problem):
                        raise AssertionError(f"Independent oracle disagreement: {family_id}")
                    rows.append({
                        "example_id": digest(f"{family_id}|{variant}"), "input_text": input_text,
                        "input_hash": digest(input_text), "task_family": domain,
                        "dataset": VERSION, "split": split, "answer": canonical(answer),
                        "answer_type": "json", "data_role": "development_construction" if split == "construction" else "controlled_reference",
                        "family_id": family_id, "parent_id": None if variant == "base" else base_id,
                        "metadata": {"generator_version": VERSION, "generator_seed": seed,
                                     "domain": domain, "family_index": family_index, "variant": variant,
                                     "shift_regime": regime, "shift_interpretation": "controlled_synthetic_not_real_temporal_drift",
                                     "template_family_id": digest(f"{family_id}|template_family"),
                                     "template_style": "data_first" if regime == "changed_template" else "instruction_first",
                                     "substream_seeds": seeds, "source_problem": problem,
                                     "mutation": mutation if counterfactual else None,
                                     "difficulty": {"entities": len(problem.get("records", problem.get("initial", []))),
                                                    "steps": len(problem.get("operations", [])),
                                                    "input_characters": len(input_text)},
                                     "answer_provenance": "python_oracle_agrees_with_independent_sql_oracle",
                                     "human_annotation": False, "model_annotation": False,
                                     "grading": "strict_json_schema_and_values_no_extra_keys"}})
    return rows


def validate(rows: list[dict]) -> dict:
    ids = [row["example_id"] for row in rows]
    hashes = [row["input_hash"] for row in rows]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise AssertionError("Duplicate ID or input")
    families = {}
    oracle_checks = 0
    for row in rows:
        if row["input_hash"] != digest(row["input_text"]):
            raise AssertionError("Input hash mismatch")
        primary, secondary = ORACLES[row["task_family"]]
        expected = primary(row["metadata"]["source_problem"])
        if expected != secondary(row["metadata"]["source_problem"]) or canonical(expected) != row["answer"]:
            raise AssertionError("Answer disagreement")
        if not json_exact_grade(row["answer"], row["answer"]):
            raise AssertionError("Grading rejected canonical answer")
        oracle_checks += 1
        families.setdefault(row["family_id"], []).append(row)
    cf_checks = paraphrase_checks = 0
    for siblings in families.values():
        if len(siblings) != 4 or len({r["split"] for r in siblings}) != 1:
            raise AssertionError("Family count or split isolation failed")
        by_variant = {r["metadata"]["variant"]: r for r in siblings}
        if set(by_variant) != set(VARIANTS):
            raise AssertionError("Missing variant")
        if by_variant["base"]["answer"] == by_variant["counterfactual"]["answer"]:
            raise AssertionError("Counterfactual did not change answer")
        cf_checks += 1
        for first, second in (("base", "paraphrase"), ("counterfactual", "counterfactual_paraphrase")):
            if by_variant[first]["answer"] != by_variant[second]["answer"]:
                raise AssertionError("Paraphrase changed answer")
            paraphrase_checks += 1
    return {"rows": len(rows), "families": len(families), "independent_oracle_agreements": oracle_checks,
            "answer_changing_counterfactual_pairs": cf_checks, "answer_preserving_paraphrase_pairs": paraphrase_checks,
            "counts_by_domain_split_regime": dict(sorted(Counter(
                f"{r['task_family']}|{r['split']}|{r['metadata']['shift_regime']}" for r in rows).items())),
            "all_inputs_unique": True, "families_split_isolated": True}


def write_dataset(output: Path, seed: int = DEFAULT_SEED) -> dict:
    rows = generate(seed)
    checks = validate(rows)
    output.mkdir(parents=True, exist_ok=True)
    files = {"normalized_tasks.jsonl": rows,
             **{f"{domain}.jsonl": [r for r in rows if r["task_family"] == domain] for domain in DOMAINS},
             **{f"{split}.jsonl": [r for r in rows if r["split"] == split] for split in ("construction", "reference")}}
    file_hashes = {}
    for name, records in files.items():
        content = "".join(canonical(row) + "\n" for row in records)
        (output / name).write_text(content, encoding="utf-8")
        file_hashes[name] = {"sha256": digest(content), "rows": len(records)}
    manifest = {"generator_version": VERSION, "seed": seed, "checks": checks, "files": file_hashes,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "external_dependencies": [], "api_calls": 0, "estimated_api_cost_usd": 0,
                "unit_of_independence": "family_id; four descendants per family",
                "template_seed_policy": "Disjoint family-specific entity/data/template RNG seeds across splits. Stationary uses the same base rendering distribution; changed_template moves data before the instruction and changes delimiters.",
                "limitations": ["Controlled procedural tasks do not measure real temporal production drift.",
                                "There are 25 independent families per domain/split, not 100 independent tasks.",
                                "SQL agreement verifies answer computation; it does not prove all possible prompt interpretations equivalent.",
                                "Reference shift proportions are experimental settings, not estimates of a deployment distribution."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/research/production_eval/experiments_v2/synthetic"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
