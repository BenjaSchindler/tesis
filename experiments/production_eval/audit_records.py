"""Read-only, fail-closed loading of one frozen machine feasibility audit.

Failed or schema-invalid attempts stay in the original JSONL. They are checked
for request provenance but omitted from the returned successful annotations.
An unsuccessful later attempt never erases an earlier valid success.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    from . import api_audit
except ImportError:  # Direct sibling imports when a script is invoked by path.
    import api_audit


def load_audits(path: Path | str, requests: Iterable[dict]) -> dict[str, dict]:
    """Return the last valid success per input hash, without changing any file.

All attempts must belong to the supplied request pool and to one model/rubric/
settings configuration. The current code's rubric and full request hash are
recomputed, so an old audit cannot silently annotate revised input text. Every
successful response must report the same actual model version. An empty file
returns an empty mapping; a missing or malformed file is an error, not evidence
that the audit completed with zero eligible requests.

Errors describe only the failed check and optional line number, never request
text, generated text, credentials, or raw exception details.
"""
    expected, identifiers = {}, {}
    for number, request in enumerate(requests, 1):
        if not isinstance(request, dict):
            raise ValueError(f"Invalid normalized request at position {number}")
        value, source_hash, identifier = (
            request.get("input_text"), request.get("input_hash"), request.get("example_id")
        )
        if (not isinstance(value, str) or not isinstance(identifier, str) or not identifier
                or source_hash != api_audit.digest(value)):
            raise ValueError(f"Invalid normalized request provenance at position {number}")
        if identifier in identifiers and identifiers[identifier] != source_hash:
            raise ValueError("One request identifier maps to different input hashes")
        identifiers[identifier] = source_hash
        expected.setdefault(source_hash, request)

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("Audit file is missing or unreadable") from None

    successes, actual_versions = {}, set()
    frozen_config = None
    current_rubric_hash = api_audit.digest(api_audit.RUBRIC)
    prepared = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            raise ValueError(f"Malformed audit JSON at line {number}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Invalid audit record at line {number}")
        source_hash = row.get("input_hash")
        if not isinstance(source_hash, str) or source_hash not in expected:
            raise ValueError(f"Audit input hash is outside the request pool at line {number}")
        identifier = row.get("example_id")
        if not isinstance(identifier, str) or identifiers.get(identifier) != source_hash:
            raise ValueError(f"Audit identifier and input hash disagree at line {number}")
        model, rubric_hash, settings = (
            row.get("requested_model"), row.get("rubric_sha256"), row.get("settings")
        )
        if (not isinstance(model, str) or not model or rubric_hash != current_rubric_hash
                or not isinstance(settings, dict)):
            raise ValueError(f"Audit model, rubric or settings provenance is invalid at line {number}")
        config = api_audit.canonical({"requested_model": model, "rubric_sha256": rubric_hash,
                                      "settings": settings})
        if frozen_config is None:
            frozen_config = config
        elif frozen_config != config:
            raise ValueError(f"Audit mixes frozen model/rubric/settings configurations at line {number}")
        if source_hash not in prepared:
            try:
                prepared[source_hash] = api_audit.prepare(expected[source_hash], model, settings)
            except (KeyError, ValueError, TypeError):
                raise ValueError(f"Audit request settings cannot be reconstructed at line {number}") from None
        key = prepared[source_hash]["cache_key"]
        if row.get("cache_key") != key or row.get("request_sha256") != key:
            raise ValueError(f"Audit full request hash does not match at line {number}")
        if row.get("annotation_provenance") != "machine_generated_unvalidated":
            raise ValueError(f"Audit annotation provenance label is invalid at line {number}")
        status = row.get("status")
        if status not in {"success", "schema_invalid", "api_error"}:
            raise ValueError(f"Unknown audit status at line {number}")
        if status != "success":
            # These are unknowns, not negative task-validity labels. Importantly,
            # the configuration checks above also cover unsuccessful attempts.
            continue
        try:
            annotation = api_audit.validate_annotation(api_audit.canonical(row.get("annotation")))
        except (ValueError, TypeError):
            raise ValueError(f"Successful audit annotation has an invalid schema at line {number}") from None
        if type(row.get("eligible_machine_prediction")) is not bool or row["eligible_machine_prediction"] != api_audit.eligible(annotation):
            raise ValueError(f"Audit eligibility disagrees with the frozen rule at line {number}")
        version = row.get("actual_model_version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"Successful audit lacks an actual model version at line {number}")
        actual_versions.add(version)
        if len(actual_versions) > 1:
            raise ValueError(f"Successful audits mix actual model versions at line {number}")
        raw = row.get("raw_response")
        if not isinstance(raw, dict) or row.get("response_sha256") != api_audit.digest(api_audit.canonical(raw)):
            raise ValueError(f"Successful audit response hash does not match at line {number}")
        successes[source_hash] = row
    return successes
