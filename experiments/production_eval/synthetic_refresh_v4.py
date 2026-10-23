"""Prepare draft v4 data only: paired input presentations under isolated shifts.

No model imports, inference, embeddings, training or final protocol freeze.
The seed contract is an existing draft_protocol.json written before generation.
Inputs, gold answers and source programs are stored in separate JSONL files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval import synthetic_tasks as original

VERSION = "gold_refresh_v4"
BASE = ROOT / "output/research/production_eval/experiments_v4"
PRIOR_FILES = (
    ROOT / "output/research/production_eval/experiments_v2/synthetic/normalized_tasks.jsonl",
    ROOT / "output/research/production_eval/experiments_v3/tasks.jsonl",
)
INPUT_FIELDS = frozenset({"example_id", "family_id", "partition", "domain", "group_id",
                          "complexity", "presentation", "input_text", "input_hash",
                          "observable_counts"})
GROUPS = (
    ("structured_records_simple", "structured_records", "simple", {"records": 8}),
    ("inventory_tracking_simple", "inventory_tracking", "simple", {"items": 3, "operations": 10}),
    ("structured_records_complex", "structured_records", "complex", {"records": 28}),
    ("inventory_tracking_complex", "inventory_tracking", "complex", {"items": 7, "operations": 36}),
)
RULES = {
    "structured_records": {
        "A": "Retain a record exactly when its team equals target_team, active is true, and void is false. "
             "Return selected_ids in ascending lexical order, total_units as the sum of retained units, "
             "and total_value as the sum of retained units multiplied by unit_price. Exclude all other records.",
        "B": "Discard records from other teams, inactive records, and voided records. For the remaining "
             "records matching target_team, alphabetize their IDs into selected_ids, add their units into "
             "total_units, and add each record's units times unit_price into total_value.",
    },
    "inventory_tracking": {
        "A": "Start from initial stock and process operations in step order. A cancelled operation has no effect. "
             "receive adds quantity at location; ship subtracts quantity at location; move subtracts quantity "
             "at location and adds it at destination for the same item. Return stock for every item, sorted "
             "lexically by item, with its final east and west counts. total_units is the sum across all "
             "items and both locations.",
        "B": "Track every item's east and west stock from the initial values. Apply the numbered steps "
             "in sequence, skipping every step whose cancelled flag is true. For receive, increase the "
             "named location; for ship, decrease it; for move, take quantity from location and place "
             "that quantity in destination for the same item. Output all final item counts in stock, "
             "ordered alphabetically by item, plus total_units over both locations.",
    },
}
SHAPES = {
    "structured_records": '{"selected_ids":["id",...],"total_units":integer,"total_value":integer}',
    "inventory_tracking": '{"stock":[{"item":"id","east":integer,"west":integer},...],"total_units":integer}',
}
OUTPUT_RULE = "Use integer arithmetic. Output one JSON object with exactly the requested keys and no commentary or markdown."
TABLE_RULE = "Source tables use tab-separated columns. true and false are Boolean values. A dash in destination means no destination."
SCHEMAS = {
    "structured_records": {
        "type": "object", "additionalProperties": False,
        "required": ["selected_ids", "total_units", "total_value"],
        "properties": {"selected_ids": {"type": "array", "items": {"type": "string"}},
                       "total_units": {"type": "integer"},
                       "total_value": {"type": "integer"}},
    },
    "inventory_tracking": {
        "type": "object", "additionalProperties": False,
        "required": ["stock", "total_units"],
        "properties": {"stock": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["item", "east", "west"],
            "properties": {"item": {"type": "string"}, "east": {"type": "integer"},
                           "west": {"type": "integer"}}}},
                       "total_units": {"type": "integer"}},
    },
}


def canonical(value):
    return original.canonical(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def strict_json(text):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate JSON source key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite source value: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def exact_keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("Unexpected source program fields")


def identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) is None:
        raise ValueError("Invalid identifier")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer, including Boolean-as-integer")


def boolean(value):
    if type(value) is not bool:
        raise ValueError("Invalid Boolean")


def validate_program(domain, program):
    """Type/shape/order checks independent of either answer oracle."""
    if domain == "structured_records":
        exact_keys(program, ("target_team", "records"))
        identifier(program["target_team"])
        rows = program["records"]
        if type(rows) is not list or not rows:
            raise ValueError("Missing records")
        for row in rows:
            exact_keys(row, ("id", "team", "active", "void", "units", "unit_price"))
            identifier(row["id"]); identifier(row["team"])
            boolean(row["active"]); boolean(row["void"])
            integer(row["units"], 1); integer(row["unit_price"], 1)
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate record IDs")
        return {"records": len(rows)}
    if domain != "inventory_tracking":
        raise ValueError("Unknown domain")
    exact_keys(program, ("initial", "operations"))
    if type(program["initial"]) is not list or not program["initial"] or \
            type(program["operations"]) is not list or not program["operations"]:
        raise ValueError("Missing inventory source rows")
    for row in program["initial"]:
        exact_keys(row, ("item", "east", "west"))
        identifier(row["item"]); integer(row["east"]); integer(row["west"])
    item_ids = {row["item"] for row in program["initial"]}
    if len(item_ids) != len(program["initial"]):
        raise ValueError("Duplicate inventory items")
    # Do not sort a damaged source: its visible event sequence is part of input.
    for index, op in enumerate(program["operations"], 1):
        kind = op.get("kind") if type(op) is dict else None
        keys = {"step", "kind", "item", "location", "quantity", "cancelled"}
        if kind == "move":
            keys.add("destination")
        exact_keys(op, keys)
        integer(op["step"], 1); integer(op["quantity"], 1); boolean(op["cancelled"])
        if op["step"] != index or kind not in ("receive", "ship", "move") or op["item"] not in item_ids:
            raise ValueError("Invalid operation, item or step order")
        if op["location"] not in ("east", "west"):
            raise ValueError("Invalid location")
        if kind == "move" and (op["destination"] not in ("east", "west") or op["destination"] == op["location"]):
            raise ValueError("Invalid move destination")
    return {"items": len(program["initial"]), "operations": len(program["operations"])}


def render_tables(domain, program):
    if domain == "structured_records":
        lines = ["target_team\t" + program["target_team"], "RECORDS",
                 "id\tteam\tactive\tvoid\tunits\tunit_price"]
        for r in program["records"]:
            lines.append("\t".join([r["id"], r["team"], str(r["active"]).lower(), str(r["void"]).lower(),
                                     str(r["units"]), str(r["unit_price"])]))
    else:
        lines = ["INITIAL STOCK", "item\teast\twest"]
        lines.extend("\t".join([r["item"], str(r["east"]), str(r["west"])]) for r in program["initial"])
        lines.extend(["", "OPERATIONS", "step\tkind\titem\tlocation\tdestination\tquantity\tcancelled"])
        for r in program["operations"]:
            lines.append("\t".join([str(r["step"]), r["kind"], r["item"], r["location"], r.get("destination", "-"),
                                     str(r["quantity"]), str(r["cancelled"]).lower()]))
    return "\n".join(lines)


def render_input(domain, program, presentation):
    validate_program(domain, program)
    if presentation == "A":
        return (f"TASK {domain}\n{RULES[domain]['A']}\n{OUTPUT_RULE}\n\nRequired JSON shape: {SHAPES[domain]}\n\n"
                f"BEGIN SOURCE JSON\n{json.dumps(program, ensure_ascii=False, indent=2)}\nEND SOURCE JSON")
    if presentation == "B":
        return (f"TASK {domain}\nBEGIN SOURCE TABLES\n{render_tables(domain, program)}\nEND SOURCE TABLES\n\n"
                f"{TABLE_RULE}\n{RULES[domain]['B']}\n{OUTPUT_RULE}\n\nRequired JSON shape: {SHAPES[domain]}")
    raise ValueError("Unknown presentation")


def parse_integer(text):
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise ValueError("Invalid TSV integer")
    return int(text)


def parse_boolean(text):
    if text not in ("true", "false"):
        raise ValueError("Invalid TSV Boolean")
    return text == "true"


def columns(line, size):
    result = line.split("\t")
    if len(result) != size or any(not value for value in result):
        raise ValueError("Malformed TSV row")
    return result


def parse_tables(domain, text):
    """Parse typed source fields directly; never calls render_tables or oracles."""
    lines = text.split("\n")
    if domain == "structured_records":
        if len(lines) < 4 or lines[1:3] != ["RECORDS", "id\tteam\tactive\tvoid\tunits\tunit_price"]:
            raise ValueError("Invalid records table headers")
        heading = columns(lines[0], 2)
        if heading[0] != "target_team":
            raise ValueError("Missing target team")
        records = []
        for line in lines[3:]:
            r = columns(line, 6)
            records.append({"id": r[0], "team": r[1], "active": parse_boolean(r[2]), "void": parse_boolean(r[3]),
                            "units": parse_integer(r[4]), "unit_price": parse_integer(r[5])})
        return {"target_team": heading[1], "records": records}
    if len(lines) < 7 or lines[:2] != ["INITIAL STOCK", "item\teast\twest"] or lines.count("OPERATIONS") != 1:
        raise ValueError("Invalid inventory table headers")
    marker = lines.index("OPERATIONS")
    if marker < 4 or lines[marker - 1] != "" or lines[marker + 1] != "step\tkind\titem\tlocation\tdestination\tquantity\tcancelled":
        raise ValueError("Invalid operations table header")
    initial, operations = [], []
    for line in lines[2:marker - 1]:
        r = columns(line, 3)
        initial.append({"item": r[0], "east": parse_integer(r[1]), "west": parse_integer(r[2])})
    for line in lines[marker + 2:]:
        r = columns(line, 7)
        op = {"step": parse_integer(r[0]), "kind": r[1], "item": r[2], "location": r[3],
              "quantity": parse_integer(r[5]), "cancelled": parse_boolean(r[6])}
        if r[1] == "move":
            op["destination"] = r[4]
        elif r[4] != "-":
            raise ValueError("Unexpected destination on a non-move operation")
        operations.append(op)
    return {"initial": initial, "operations": operations}


def parse_input(text):
    """Recover only visible source; exact framing also detects changed rules."""
    if type(text) is not str or not text.startswith("TASK "):
        raise ValueError("Missing task header")
    header, remainder = text.split("\n", 1)
    domain = header[5:]
    if domain not in RULES:
        raise ValueError("Unknown task header")
    if remainder.startswith("BEGIN SOURCE TABLES\n"):
        presentation = "B"
        parts = remainder.split("\nEND SOURCE TABLES\n\n")
        if len(parts) != 2:
            raise ValueError("Invalid tables framing")
        source, trailer = parts
        expected = f"{TABLE_RULE}\n{RULES[domain]['B']}\n{OUTPUT_RULE}\n\nRequired JSON shape: {SHAPES[domain]}"
        if trailer != expected:
            raise ValueError("Changed processing rules or output specification")
        program = parse_tables(domain, source[len("BEGIN SOURCE TABLES\n"):])
    else:
        presentation = "A"
        prefix = f"{RULES[domain]['A']}\n{OUTPUT_RULE}\n\nRequired JSON shape: {SHAPES[domain]}\n\nBEGIN SOURCE JSON\n"
        if not remainder.startswith(prefix) or not remainder.endswith("\nEND SOURCE JSON"):
            raise ValueError("Changed processing rules or JSON framing")
        source = remainder[len(prefix):-len("\nEND SOURCE JSON")]
        program = strict_json(source)
    validate_program(domain, program)
    return domain, presentation, program


def validate_protocol(protocol):
    if protocol.get("generator_namespace") != VERSION or protocol.get("master_seed") != 2026091504:
        raise ValueError("Seed contract differs from the pre-generation draft")
    if protocol.get("final_design_frozen") is not False or protocol.get("inference_enabled") is not False:
        raise ValueError("This preparer only accepts draft, inference-disabled protocols")
    expected_groups = [{"group_id": g, "domain": d, "complexity": c, **counts} for g, d, c, counts in GROUPS]
    if protocol["groups"] != expected_groups or protocol["families_per_group"] != {"fixture": 4, "construction": 50, "future": 100}:
        raise ValueError("Draft group counts changed")
    if protocol["presentations_by_partition"] != {"fixture": ["A", "B"], "construction": ["A"], "future": ["A", "B"]}:
        raise ValueError("Draft presentation allocation changed")
    if len(protocol["scenarios"]) != 4 or any(len(s["group_weights"]) != 4 or abs(sum(s["group_weights"]) - 1) > 1e-12
                                             or any(w <= 0 for w in s["group_weights"]) for s in protocol["scenarios"]):
        raise ValueError("Invalid scenario weights")


def generate(protocol):
    validate_protocol(protocol)
    inputs, gold, programs = [], [], []
    for partition, count in protocol["families_per_group"].items():
        for group, domain, complexity, expected_counts in GROUPS:
            for index in range(count):
                family = original.digest(f"{VERSION}|{protocol['master_seed']}|{partition}|{group}|{index}")
                builder = original.build_records if domain == "structured_records" else original.build_inventory
                program, _, _ = builder(original.derived_seed(family, "data"), original.derived_seed(family, "entities"),
                                        "more_complex" if complexity == "complex" else "stationary")
                counts = validate_program(domain, program)
                if counts != expected_counts:
                    raise ValueError("Builder violates declared complexity")
                primary, secondary = original.ORACLES[domain]
                answer = primary(program)
                if answer != secondary(program):
                    raise ValueError("Python and SQL disagree")
                program_hash = original.digest(canonical(program))
                common = {"family_id": family, "partition": partition, "domain": domain,
                          "group_id": group, "complexity": complexity}
                programs.append({**common, "program": program, "program_hash": program_hash,
                                 "family_index": index,
                                 "substream_seeds": {kind: original.derived_seed(family, kind) for kind in ("data", "entities")}})
                for presentation in protocol["presentations_by_partition"][partition]:
                    example_id = original.digest(family + "|" + presentation)
                    text = render_input(domain, program, presentation)
                    inputs.append({**common, "example_id": example_id, "presentation": presentation,
                                   "input_text": text, "input_hash": original.digest(text), "observable_counts": counts})
                    gold.append({"example_id": example_id, "family_id": family, "answer": canonical(answer),
                                 "answer_hash": original.digest(canonical(answer)), "program_hash": program_hash,
                                 "label_provenance": "independent_python_and_sql_oracles_agree", "human_annotation": False})
    return inputs, gold, programs


def extract_prior_program(row):
    if "source_problem" in row.get("metadata", {}):
        return row["metadata"]["source_problem"]
    # Earlier v3 inputs contain one full JSON program plus a non-JSON shape hint.
    # Parse candidate JSON objects from text, accepting only an exact source root.
    candidates = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", row["input_text"]):
        try:
            value, _ = decoder.raw_decode(row["input_text"], match.start())
        except ValueError:
            continue
        if type(value) is dict and set(value) in ({"target_team", "records"}, {"initial", "operations"}):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("Cannot reconstruct exactly one prior source program")
    return candidates[0]


def audit_prior_disjoint(inputs, programs, prior_files):
    families = {r["family_id"] for r in programs}
    input_hashes = {r["input_hash"] for r in inputs}
    program_hashes = {r["program_hash"] for r in programs}
    evidence = []
    for path in prior_files:
        prior = load_jsonl(path)
        old_families = {r["family_id"] for r in prior}
        old_inputs = {original.digest(r["input_text"]) for r in prior}
        old_programs = {original.digest(canonical(extract_prior_program(r))) for r in prior}
        if families & old_families or input_hashes & old_inputs or program_hashes & old_programs:
            raise ValueError("v4 overlaps prior families, exact inputs or source programs")
        evidence.append({"path": str(Path(path).resolve()), "sha256": sha(path), "rows": len(prior),
                         "families": len(old_families), "source_programs": len(old_programs),
                         "family_id_intersection": 0, "input_hash_intersection": 0, "program_hash_intersection": 0})
    return evidence


def validate_dataset(protocol, inputs, gold, programs):
    validate_protocol(protocol)
    if any(set(row) != INPUT_FIELDS for row in inputs):
        raise ValueError("Input file contains unexpected fields, including possible gold leakage")
    if len({r["example_id"] for r in inputs}) != len(inputs) or len({r["input_hash"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate inputs/IDs")
    by_gold = {r["example_id"]: r for r in gold}
    by_program = {r["family_id"]: r for r in programs}
    if len(by_gold) != len(gold) or len(by_program) != len(programs) or set(by_gold) != {r["example_id"] for r in inputs}:
        raise ValueError("Gold/program join has missing or duplicate identities")
    if len({r["program_hash"] for r in programs}) != len(programs):
        raise ValueError("Programs duplicated across families")
    for source in programs:
        index = source["family_index"]
        partition = source["partition"]
        if partition not in protocol["families_per_group"] or type(index) is not int or not 0 <= index < protocol["families_per_group"][partition]:
            raise ValueError("Source family index exceeds its seed allocation")
        seeds = {kind: original.derived_seed(source["family_id"], kind) for kind in ("data", "entities")}
        if source["substream_seeds"] != seeds:
            raise ValueError("Source seeds changed")
        builder = original.build_records if source["domain"] == "structured_records" else original.build_inventory
        regenerated, _, _ = builder(seeds["data"], seeds["entities"],
                                    "more_complex" if source["complexity"] == "complex" else "stationary")
        if canonical(regenerated) != canonical(source["program"]):
            raise ValueError("Source program differs from its declared deterministic seed")
    families = defaultdict(list)
    for row in inputs:
        family = row["family_id"]
        if family not in by_program:
            raise ValueError("Missing source family")
        source = by_program[family]
        domain, presentation, parsed = parse_input(row["input_text"])
        if domain != row["domain"] or presentation != row["presentation"] or canonical(parsed) != canonical(source["program"]):
            raise ValueError("Rendered input does not reconstruct its original program")
        if any(row[key] != source[key] for key in ("partition", "domain", "group_id", "complexity")):
            raise ValueError("Family crosses partitions/groups")
        expected_family = original.digest(f"{VERSION}|{protocol['master_seed']}|{row['partition']}|{row['group_id']}|{source['family_index']}")
        if family != expected_family or row["example_id"] != original.digest(family + "|" + presentation):
            raise ValueError("ID violates declared seed derivation")
        counts = validate_program(domain, parsed)
        expected = next(c for g, _, _, c in GROUPS if g == row["group_id"])
        if counts != expected or row["observable_counts"] != counts:
            raise ValueError("Observable counts differ from visible input or group")
        if original.digest(row["input_text"]) != row["input_hash"] or original.digest(canonical(parsed)) != source["program_hash"]:
            raise ValueError("Input/program hash mismatch")
        primary, secondary = original.ORACLES[domain]
        expected_answer = primary(parsed)
        if expected_answer != secondary(parsed):
            raise ValueError("Parsed-input oracles disagree")
        target = by_gold[row["example_id"]]
        if target["family_id"] != family or target["program_hash"] != source["program_hash"] or \
                target["answer"] != canonical(expected_answer) or target["answer_hash"] != original.digest(target["answer"]):
            raise ValueError("Gold differs from parsed input answer")
        if target["human_annotation"] is not False or target["label_provenance"] != "independent_python_and_sql_oracles_agree":
            raise ValueError("Incorrect label provenance")
        families[family].append(row)
    if set(families) != set(by_program):
        raise ValueError("Unreferenced source families")
    expected_family_counts = {(partition, group): count for partition, count in protocol["families_per_group"].items()
                              for group, _, _, _ in GROUPS}
    actual_counts = Counter((r["partition"], r["group_id"]) for r in programs)
    if actual_counts != expected_family_counts:
        raise ValueError("Family counts changed")
    paired = Counter()
    for members in families.values():
        partition = members[0]["partition"]
        if sorted(r["presentation"] for r in members) != sorted(protocol["presentations_by_partition"][partition]):
            raise ValueError("Missing/extra family presentation")
        if len(members) == 2:
            if len({by_gold[r["example_id"]]["answer"] for r in members}) != 1:
                raise ValueError("Presentation changes answer")
            paired[partition] += 1
    research_inputs = sum(r["partition"] != "fixture" for r in inputs)
    fixture_inputs = len(inputs) - research_inputs
    if research_inputs != protocol["expected_research_inputs_per_system"] or \
            fixture_inputs != protocol["expected_fixture_inputs_per_system"]:
        raise ValueError("Planned call count differs from actual inputs")
    return {"inputs": len(inputs), "research_inputs": research_inputs, "fixture_inputs": fixture_inputs,
            "families": len(programs), "family_counts": {f"{a}|{b}": n for (a, b), n in sorted(actual_counts.items())},
            "paired_program_and_answer_checks": dict(paired), "independent_parsed_input_oracle_agreements": len(inputs),
            "input_fields_exact_allowlist": sorted(INPUT_FIELDS), "source_order_preserved": True,
            "within_v4_family_input_program_disjointness": True, "model_calls_executed": 0,
            "planned_research_model_calls": research_inputs * len(protocol["draft_systems"]),
            "planned_fixture_model_calls": fixture_inputs * len(protocol["draft_systems"])}


def prepare(output=BASE):
    output = Path(output)
    protocol_path = output / "draft_protocol.json"
    if not protocol_path.is_file():
        raise ValueError("Write the seed contract before generating any draft data")
    destination = output / "draft_dataset"
    if destination.exists():
        raise FileExistsError("Draft data already exists; refusing overwrite")
    protocol_hash_before = sha(protocol_path)
    protocol = strict_json(protocol_path.read_text())
    inputs, gold, programs = generate(protocol)
    checks = validate_dataset(protocol, inputs, gold, programs)
    checks["prior_disjointness"] = audit_prior_disjoint(inputs, programs, PRIOR_FILES)
    if sha(protocol_path) != protocol_hash_before:
        raise ValueError("Seed contract changed during preparation")
    # Stage complete artifacts; no in-place modifications to previous datasets.
    stage = Path(tempfile.mkdtemp(prefix=".draft_dataset_", dir=output))
    payloads = {"inputs.jsonl": inputs, "gold.jsonl": gold, "programs.jsonl": programs}
    for partition in protocol["families_per_group"]:
        payloads[f"inputs_{partition}.jsonl"] = [r for r in inputs if r["partition"] == partition]
    evidence = {}
    for name, rows in payloads.items():
        path = stage / name
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        evidence[name] = {"sha256": sha(path), "rows": len(rows)}
    (stage / "schemas.json").write_text(json.dumps(SCHEMAS, indent=2) + "\n")
    evidence["schemas.json"] = {"sha256": sha(stage / "schemas.json"), "domains": len(SCHEMAS)}
    manifest = {"status": "draft_dataset_validated_pending_root_review", "final_design_frozen": False,
                "inference_enabled": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator_namespace": VERSION, "master_seed": protocol["master_seed"],
                "preexisting_seed_protocol_sha256": protocol_hash_before, "files": evidence, "checks": checks,
                "source_files": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                                 (Path(__file__), Path(original.__file__), Path(__file__).with_name("test_synthetic_refresh_v4.py"))],
                "model_calls_executed": 0, "selection_performed": False, "embeddings_computed": False,
                "unit_of_independence": "family; A and B are paired presentations of exactly the same source program",
                "input_gold_separation": "Only inputs*.jsonl may feed input-only selection/inference. Join gold.jsonl/programs.jsonl only in explicitly authorized grading/oracle arms.",
                "limitations": ["Draft requires root review and final configuration before any inference.",
                                "A/B changes a defined instruction/serialization package, not underlying program or arithmetic.",
                                "Synthetic scenarios and weights do not measure real production time or prevalence.",
                                "Logical file separation is a pipeline contract, not a cryptographic barrier to oracle access.",
                                "No model, human correctness or annotation-cost evidence was collected."]}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "README.md").write_text(
        "# Borrador de datos v4\n\nValidado por parsers y oráculos; pendiente de revisión del diseño. No hay inferencia ni protocolo final congelado.\n\n"
        "`inputs_*.jsonl` contiene solo entradas y metadatos observables. `gold.jsonl` y `programs.jsonl` están separados. "
        "A usa JSON; B tablas TSV. El parser reconstruye el mismo programa de cada presentación y conserva el orden de inventario.\n\n"
        "200 familias de construcción, 400 futuras con A/B y 16 fixtures con A/B: 1.000 entradas de investigación y 32 de fixtures por sistema. "
        "Para tres sistemas: 3.000 llamadas de investigación más 96 si se ejecutan todos los fixtures. Llamadas ejecutadas: 0.\n")
    os.rename(stage, destination)
    return {"destination": str(destination.resolve()), "status": manifest["status"], "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
