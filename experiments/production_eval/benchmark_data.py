"""Acquire two bounded, pinned public benchmarks and score final answer strings.

This is a retrospective benchmark experiment, not production traffic or temporal
validation.  Roles are fixed without model outputs.  No inference is performed.

Run once with network access:
    python -m experiments.production_eval.benchmark_data
Existing manifests are verified offline on subsequent runs (no silent resplit).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_OUTPUT = Path("output/research/production_eval/experiments_v2/benchmarks")
SEED = 20260914
MAX_FILE_BYTES = 2_000_000
MAX_DOWNLOAD_BYTES = 10_000_000
SOURCES = (
    {"dataset": "gsm8k", "repo": "openai/grade-school-math", "branch": "master",
     "data_path": "grade_school_math/data/test.jsonl", "split": "test",
     "roles": {"development": 100, "construction": 300, "reference": 300},
     "paper": "https://arxiv.org/abs/2110.14168"},
    {"dataset": "bbh_boolean_expressions", "repo": "suzgunmirac/BIG-Bench-Hard", "branch": "main",
     "data_path": "bbh/boolean_expressions.json", "split": "published_bbh_task",
     "roles": {"development": 50, "construction": 100, "reference": 100},
     "paper": "https://arxiv.org/abs/2210.09261"},
)

# Locale-independent decimal notation only; commas must group thousands.
# No substring extraction, exponents, units, rounding, NaN, or fractions.
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?\Z", re.ASCII)


def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def canonical_answer(answer: Any, answer_type: str) -> str | None:
    """Return exact canonical form or None; accepting a substring is forbidden."""
    if not isinstance(answer, str) or len(answer) > 256:
        return None
    value = answer.strip()
    if answer_type == "boolean":
        return value.capitalize() if value.casefold() in {"true", "false"} else None
    if answer_type != "decimal":
        raise ValueError(f"Unsupported answer type: {answer_type}")
    if not NUMBER.fullmatch(value):
        return None
    number = Decimal(value.replace(",", ""))
    # Avoid Decimal.normalize(): its context precision may round long answers.
    if number == 0:
        return "0"
    rendered = format(number, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def check_answer(answer: Any, expected: str, answer_type: str) -> dict[str, Any]:
    """Score the string in JSON {'answer': string}, never arbitrary model text.

    Invalid model format is an incorrect outcome plus an explicit format flag;
    malformed ground truth raises, so a dataset error cannot become a model error.
    """
    gold = canonical_answer(expected, answer_type)
    if gold is None:
        raise ValueError("Invalid supplied reference answer")
    predicted = canonical_answer(answer, answer_type)
    return {"correct": predicted is not None and predicted == gold,
            "valid_format": predicted is not None, "canonical_answer": predicted,
            "error": None if predicted is not None else "invalid_final_answer_format"}


def verify_boolean_expression(text: str) -> str:
    """Evaluate the benchmark's tiny Boolean grammar by AST, without eval()."""
    if not text.endswith(" is") or len(text) > 4096:
        raise ValueError("Unexpected Boolean prompt format")
    tree = ast.parse(text[:-3], mode="eval")

    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            # Visit every child even if Python would short-circuit, so disallowed
            # expressions cannot hide behind True or / False and.
            values = [visit(child) for child in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        raise ValueError("Expression outside the literal Boolean grammar")

    return str(visit(tree))


def normalized_row(source: dict[str, Any], commit: str, index: int,
                   text: str, answer: str, answer_type: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty/nonstring input")
    canonical = canonical_answer(answer, answer_type)
    if canonical is None:
        raise ValueError("Source answer cannot be parsed strictly")
    source_id = f"{source['data_path']}:{index}"
    family = sha256(" ".join(text.split()).casefold())
    return {"example_id": sha256(f"{source['dataset']}|{commit}|{source_id}"),
            "input_text": text, "input_hash": sha256(text),
            "task_family": "arithmetic_word_problem" if answer_type == "decimal" else "boolean_reasoning",
            "dataset": source["dataset"], "split": source["split"],
            "answer": canonical, "answer_type": answer_type,
            "source_id": source_id, "source_revision": commit,
            "data_role": "unassigned", "family_id": family,
            "label_provenance": "published_reference_answer",
            "label_independently_verified": answer_type == "boolean"}


def parse_source(source: dict[str, Any], commit: str, payload: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source["dataset"] == "gsm8k":
        raw = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        rows = []
        for index, item in enumerate(raw):
            if set(item) != {"question", "answer"} or not isinstance(item["answer"], str):
                raise ValueError("Unexpected GSM8K schema")
            parts = item["answer"].split("####")
            if len(parts) != 2:
                raise ValueError("GSM8K source must contain one final-answer marker")
            rows.append(normalized_row(source, commit, index, item["question"], parts[1].strip(), "decimal"))
        return rows, {"source_count": len(rows), "source_schema": "JSONL question:string, answer:string with #### final decimal",
                      "independent_answer_verification": "Not performed; published GSM8K labels are used, possible source errors remain."}
    raw = json.loads(payload)
    if not isinstance(raw.get("examples"), list):
        raise ValueError("Unexpected BBH schema")
    rows = []
    for index, item in enumerate(raw["examples"]):
        if set(item) != {"input", "target"}:
            raise ValueError("Unexpected BBH record schema")
        if verify_boolean_expression(item["input"]) != item["target"]:
            raise ValueError(f"Published Boolean label disagrees with grammar evaluator at index {index}")
        rows.append(normalized_row(source, commit, index, item["input"], item["target"], "boolean"))
    return rows, {"source_count": len(rows), "source_schema": "JSON object with examples:[{input:string,target:string}] and canary",
                  "independent_answer_verification": "All source answers verified by restricted literal Boolean AST evaluator.",
                  "source_canary": raw.get("canary")}


def assign_roles(rows: list[dict[str, Any]], counts: dict[str, int], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One representative per exact normalized-input family; hash order is stable.

    Hash(seed,dataset,family) acts as a reproducible pseudorandom permutation,
    independent of source ordering, answer labels, and later model outcomes.
    This protects exact duplicates, not unknown paraphrase/template families.
    """
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Invalid role counts")
    families: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda row: row["example_id"]):
        previous = families.get(row["family_id"])
        if previous and previous["answer"] != row["answer"]:
            raise ValueError("Duplicate input has conflicting supplied answers")
        families.setdefault(row["family_id"], row)
    ordered = sorted(families.values(), key=lambda row: sha256(f"{seed}|{row['dataset']}|{row['family_id']}"))
    if sum(counts.values()) > len(ordered):
        raise ValueError(f"Requested {sum(counts.values())} unique families but only {len(ordered)} exist")
    selected = []
    cursor = 0
    for role, count in counts.items():
        selected.extend({**row, "data_role": role} for row in ordered[cursor:cursor + count])
        cursor += count
    return selected, {"selected_count": len(selected), "unique_normalized_input_families": len(families),
                      "duplicates_removed": len(rows) - len(families), "unused_unique_families": len(families) - len(selected),
                      "role_counts": dict(Counter(row["data_role"] for row in selected))}


class BoundedDownloader:
    def __init__(self) -> None:
        self.total_bytes = 0

    def get(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "thesis-verifiable-benchmark-research/1.0", "Accept": "application/vnd.github+json"})
        with urlopen(request, timeout=45) as response:
            payload = response.read(min(MAX_FILE_BYTES, MAX_DOWNLOAD_BYTES - self.total_bytes) + 1)
        self.total_bytes += len(payload)
        if len(payload) > MAX_FILE_BYTES or self.total_bytes > MAX_DOWNLOAD_BYTES:
            raise ValueError("Bounded download size exceeded")
        return payload


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def verify_existing(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    for file in manifest["files"]:
        if sha256((output / file["path"]).read_bytes()) != file["sha256"]:
            raise ValueError(f"Integrity failure in {file['path']}")
    return {"status": "verified_existing_offline", "record_count": manifest["record_count"],
            "normalized_sha256": manifest["normalized_sha256"]}


def prepare(output: Path, seed: int = SEED) -> dict[str, Any]:
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["role_assignment"]["seed"] != seed:
            raise ValueError("Existing frozen roles have a different seed; choose another output directory")
        return verify_existing(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output without final manifest; preserve partial files and use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    downloader = BoundedDownloader()
    datasets, files, all_rows = [], [], []
    for source in SOURCES:
        repo, branch = source["repo"], source["branch"]
        commit_url = f"https://api.github.com/repos/{repo}/commits/{branch}"
        commit_payload = downloader.get(commit_url)
        commit = json.loads(commit_payload)["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Unexpected commit SHA")
        source_dir = Path("sources") / source["dataset"]
        revision_path = source_dir / "revision.json"
        write_bytes(output / revision_path, commit_payload)
        files.append({"path": str(revision_path), "url": commit_url, "sha256": sha256(commit_payload), "bytes": len(commit_payload)})
        downloaded: dict[str, bytes] = {}
        for remote in (source["data_path"], "LICENSE", "README.md"):
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{remote}"
            payload = downloader.get(url)
            path = source_dir / Path(remote).name
            write_bytes(output / path, payload)
            files.append({"path": str(path), "url": url, "sha256": sha256(payload), "bytes": len(payload)})
            downloaded[remote] = payload
        if not downloaded["LICENSE"].startswith(b"MIT License"):
            raise ValueError("Repository license differs from the verified MIT source")
        rows, schema_summary = parse_source(source, commit, downloaded[source["data_path"]])
        selected, role_summary = assign_roles(rows, source["roles"], seed)
        all_rows.extend(selected)
        datasets.append({**source, "resolved_commit": commit, "license": "MIT (repository LICENSE retained)",
                         "repository_url": f"https://github.com/{repo}/tree/{commit}", **schema_summary, **role_summary})
    if len({row["input_hash"] for row in all_rows}) != len(all_rows):
        raise ValueError("Cross-dataset duplicate exact inputs found")
    blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_rows).encode("utf-8")
    path = Path("normalized_records.jsonl")
    write_bytes(output / path, blob)
    files.append({"path": str(path), "sha256": sha256(blob), "bytes": len(blob)})
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_without_model_outputs", "record_count": len(all_rows), "normalized_sha256": sha256(blob),
        "datasets": datasets, "files": files,
        "role_assignment": {"seed": seed, "method": "Sort unique normalized-input families by SHA256(seed|dataset|family_id), take fixed role counts in declared order.",
                            "role_order": ["development", "construction", "reference"],
                            "model_output_independent": True, "split_type": "random_retrospective_not_temporal",
                            "family_definition": "SHA256(whitespace-collapsed, casefolded input); does not detect paraphrases or all shared templates."},
        "checker": {"module": "experiments.production_eval.benchmark_data", "entrypoint": "check_answer(answer, expected, answer_type)",
                    "code_sha256": sha256(Path(__file__).read_bytes()), "version": 1,
                    "answer_contract": "JSON object {'answer': string}; caller passes only its string value. Decimals allow optional sign, valid thousands commas, decimal fraction. Boolean permits case variants. No prose/units/alternatives/fractions/exponents.",
                    "invalid_format_counts_as_incorrect": True, "score_interpretation": "Strict final-answer correctness; format compliance reported separately."},
        "network": {"downloaded_bytes": downloader.total_bytes, "max_file_bytes": MAX_FILE_BYTES, "max_total_bytes": MAX_DOWNLOAD_BYTES},
        "limitations": [
            "Both are established public benchmarks; possible training contamination of current models is unknown and cannot be excluded by a new random split.",
            "Neither dataset is production traffic. No timestamps or production-drift claim are introduced.",
            "GSM8K labels are supplied published references; no independent human verification of every word problem was performed.",
            "BBH Boolean labels were independently recomputed with a restricted Boolean grammar. This verifies labels, not representativeness or difficulty.",
            "BBH contains a benchmark canary warning against training-corpus inclusion; this preparation is for evaluation only and retains the canary in the original source and manifest.",
            "Random reference roles are held out from selector construction; adaptive repeated use would make them development data. Freeze analyses before first reference evaluation.",
            "Exact normalized-input family separation does not establish semantic or template independence."
        ]}
    write_bytes(output / "manifest.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    return {"status": manifest["status"], "record_count": len(all_rows), "normalized_sha256": sha256(blob),
            "downloaded_bytes": downloader.total_bytes, "datasets": [{"dataset": item["dataset"], "commit": item["resolved_commit"],
            "source_count": item["source_count"], "role_counts": item["role_counts"], "duplicates_removed": item["duplicates_removed"]} for item in datasets]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
