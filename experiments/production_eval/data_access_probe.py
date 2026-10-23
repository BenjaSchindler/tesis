#!/usr/bin/env python3
"""Bounded, revision-checked public WildChat acquisition for feasibility review.

This uses public GETs only. It makes no model calls, labels no outcomes, and
never downloads a Parquet shard. Local raw records must remain local review
material: stdout contains metadata only. A saved row is the *full* source row,
so later code must use the first fixed prefix rather than replaying later turns.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time

import requests

DATASET = "allenai/WildChat-4.8M"
REVISION = "c827c6df8fcf008219ffaffa4d1dd77491099367"
ROW_COUNT = 3_199_860
BASE = "https://datasets-server.huggingface.co"
PARAMS = {"dataset": DATASET, "config": "default", "split": "train"}
REQUEST_INTERVAL = 4.0
RATE_LOCK = threading.Lock()
NEXT_REQUEST_TIME = 0.0


def await_request_slot():
    global NEXT_REQUEST_TIME
    with RATE_LOCK:
        delay = max(0, NEXT_REQUEST_TIME - time.monotonic())
        if delay:
            time.sleep(delay)
        NEXT_REQUEST_TIME = time.monotonic() + REQUEST_INTERVAL


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_get(url, params=None, expected_revision=None):
    """Retry transient GET failures; reject stale snapshots or oversized rows."""
    for attempt in range(8):
        try:
            await_request_slot()
            with requests.get(url, params=params, timeout=(15, 45), stream=True) as response:
                if response.status_code == 429:
                    # Never rotate identity or endpoint to evade the public limit.
                    # A shared pause also prevents the other worker from flooding it.
                    with RATE_LOCK:
                        global NEXT_REQUEST_TIME
                        NEXT_REQUEST_TIME = max(NEXT_REQUEST_TIME, time.monotonic() + 60)
                    print(json.dumps({"event": "public_rate_limit", "attempt": attempt + 1,
                                      "minimum_pause_seconds": 60}), flush=True)
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                if expected_revision and response.headers.get("x-revision") != expected_revision:
                    raise ValueError("Dataset-server revision mismatch or absent x-revision")
                chunks, count = [], 0
                for chunk in response.iter_content(64 * 1024):
                    count += len(chunk)
                    if count > 12_000_000:
                        raise ValueError("Single JSON response exceeded the 12 MB safety bound")
                    chunks.append(chunk)
                body = b"".join(chunks)
                return json.loads(body), {
                    "url": response.url,
                    "retrieved_at": utc_now(),
                    "x_revision": response.headers.get("x-revision"),
                    "decoded_response_bytes": len(body),
                    "response_sha256": hashlib.sha256(body).hexdigest(),
                    "attempts": attempt + 1,
                }
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 12))


def check_source_revision():
    source, evidence = json_get(f"https://huggingface.co/api/datasets/{DATASET}")
    if source["sha"] != REVISION or source.get("gated") is not False:
        raise ValueError("Expected pinned, ungated public dataset revision is unavailable")
    return {
        "sha": source["sha"], "lastModified": source["lastModified"],
        "gated": source["gated"], **evidence,
    }


def sample_design(sample_count, seed):
    rng = random.Random(seed)
    for i in range(sample_count):
        start, stop = i * ROW_COUNT // sample_count, (i + 1) * ROW_COUNT // sample_count
        size = stop - start
        yield {
            "stratum_id": i, "row_index_start": start, "row_index_stop_exclusive": stop,
            "stratum_population_size": size, "source_row_index": rng.randrange(start, stop),
            "inclusion_probability": 1 / size, "design_weight": size,
        }


def fetch_one(design):
    payload, evidence = json_get(
        BASE + "/rows", {**PARAMS, "offset": design["source_row_index"], "length": 1}, REVISION
    )
    if payload.get("partial") is not False or payload.get("num_rows_total") != ROW_COUNT:
        raise ValueError("Rows endpoint was partial or had an unexpected population count")
    if len(payload.get("rows", [])) != 1:
        raise ValueError("Expected exactly one source row")
    record = payload["rows"][0]
    if record["row_idx"] != design["source_row_index"]:
        raise ValueError("Returned row index differed from requested sample unit")
    return {
        "dataset": DATASET, "dataset_revision": REVISION, "sampling": design,
        "retrieval": evidence, "row_idx": record["row_idx"],
        "truncated_cells": record.get("truncated_cells", []), "row": record["row"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-interval", type=float, default=4.0)
    parser.add_argument("--method", choices=["rows", "parquet"], default="rows")
    parser.add_argument("--max-transfer-bytes", type=int, default=500_000_000)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/research/production_eval/execution"))
    args = parser.parse_args()
    if not 1 <= args.sample_count <= 300 or not 1 <= args.workers <= 6 or args.request_interval < 4:
        parser.error("Use 1–300 source rows and 1–6 concurrent public requests")
    global REQUEST_INTERVAL
    REQUEST_INTERVAL = args.request_interval
    if args.method == "parquet":
        parquet_acquire(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_sample.jsonl"
    checkpoint_path = args.output_dir / "raw_sample.partial.jsonl"
    manifest_path = args.output_dir / "data_access_manifest.json"
    if raw_path.exists() or manifest_path.exists():
        parser.error("Output already exists; use a new directory to preserve prior evidence")
    started = utc_now()
    source_before = check_source_revision()
    designs = list(sample_design(args.sample_count, args.seed))
    records = []
    if checkpoint_path.exists():
        records = [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line]
        expected = {d["source_row_index"]: d for d in designs}
        for record in records:
            if (record["dataset_revision"] != REVISION or record["sampling"] != expected.get(record["row_idx"])
                    or record["retrieval"]["x_revision"] != REVISION):
                raise ValueError("Existing checkpoint does not match this exact sample design and revision")
        if len({r["row_idx"] for r in records}) != len(records):
            raise ValueError("Duplicate records in checkpoint")
    completed_ids = {r["row_idx"] for r in records}
    with os.fdopen(os.open(checkpoint_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(fetch_one, design) for design in designs
                       if design["source_row_index"] not in completed_ids]
            failures = []
            for completed in as_completed(pending):
                try:
                    record = completed.result()
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
                    continue
                records.append(record)
                checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                checkpoint.flush()
                if len(records) % 10 == 0:
                    print(json.dumps({"downloaded_records": len(records), "of": args.sample_count}), flush=True)
            if failures:
                raise RuntimeError(f"{len(failures)} row GETs failed; completed rows are checkpointed. First: {failures[0]}")
    source_after = check_source_revision()
    records.sort(key=lambda x: x["row_idx"])
    with os.fdopen(os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "status": "complete", "started_at": started, "completed_at": utc_now(),
        "dataset": DATASET, "dataset_revision": REVISION,
        "population": "All released public conversation rows; not unique users or requests",
        "population_rows": ROW_COUNT, "sample_count": len(records), "seed": args.seed,
        "minimum_request_interval_seconds": REQUEST_INTERVAL,
        "sampling_method": "One uniformly random row from each exhaustive equal-width row-index stratum",
        "selection_uses_text_or_outcomes": False,
        "time_spread": "Row-index stratification only; calendar balance is not guaranteed or assumed",
        "source_before": source_before, "source_after": source_after,
        "all_row_response_revisions_match": all(r["retrieval"]["x_revision"] == REVISION for r in records),
        "truncated_record_count": sum(bool(r["truncated_cells"]) for r in records),
        "decoded_row_response_bytes": sum(r["retrieval"]["decoded_response_bytes"] for r in records),
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_file": str(raw_path),
        "gold_labels_generated": False, "paid_inference_calls": 0,
        "attribution": "Allen Institute for AI / WildChat authors; public release ODC-BY",
        "limitations": [
            "The public release excludes conversations flagged toxic; no full-traffic prevalence claim.",
            "Source assistant responses are unverified outputs, never golden answers.",
            "Conversation final timestamp is not the first request timestamp.",
            "Monthly counts from this 300-row sample are estimates, not a census.",
            "Shared-turn and duplicate-family dependence still requires audit.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoint_path.unlink()
    print(json.dumps({k:manifest[k] for k in ["status", "sample_count", "decoded_row_response_bytes", "truncated_record_count", "raw_file_sha256"]}), flush=True)


def parquet_acquire(args):
    """Read exactly the same predetermined sample through bounded projections."""
    from bisect import bisect_right
    import io
    import struct
    import pyarrow.parquet as pq
    from metadata_scan import RangeSource, TransferBudget, CachedRanges

    projected = [
        "conversation_hash", "model", "timestamp", "turn", "language", "toxic", "redacted", "hashed_ip",
        "conversation.list.element.content", "conversation.list.element.role",
        "conversation.list.element.timestamp", "conversation.list.element.created",
        "conversation.list.element.turn_identifier",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "raw_sample.jsonl"
    partial = args.output_dir / "raw_sample.partial.jsonl"
    manifest_path = args.output_dir / "data_access_manifest.json"
    if raw.exists() or manifest_path.exists():
        raise ValueError("Choose a new output directory; existing acquisition evidence is preserved")
    existing = [json.loads(line) for line in partial.read_text().splitlines() if line] if partial.exists() else []
    expected_designs = {d["source_row_index"]: d for d in sample_design(args.sample_count, args.seed)}
    for record in existing:
        if record["dataset_revision"] != REVISION or record["sampling"] != expected_designs.get(record["row_idx"]):
            raise ValueError("Checkpoint does not match the exact pinned sample design")
    completed_ids = {record["row_idx"] for record in existing}
    if len(completed_ids) != len(existing):
        raise ValueError("Duplicate source rows in checkpoint")
    started = utc_now()
    before = check_source_revision()
    budget = TransferBudget(args.max_transfer_bytes)
    cache_dir = args.output_dir / "parquet_footer_cache"
    cache_dir.mkdir(mode=0o700, exist_ok=True)

    def footer_for(i):
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-{i:05d}-of-00086.parquet"
        source = RangeSource(url, budget)
        cache_path = cache_dir / f"shard-{i:05d}.footer"
        cache_meta_path = cache_dir / f"shard-{i:05d}.json"
        if cache_path.exists() and cache_meta_path.exists():
            cached = cache_path.read_bytes()
            cached_meta = json.loads(cache_meta_path.read_text())
            if (cached_meta["url"] != url or hashlib.sha256(cached).hexdigest() != cached_meta["sha256"]
                    or cached[-4:] != b"PAR1"):
                raise ValueError("Pinned footer cache failed integrity check")
            source.size = cached_meta["size"]
            metadata = pq.read_metadata(io.BytesIO(b"PAR1" + cached))
            return i, source, cached_meta["footer_start"], cached, metadata
        tail = source.fetch(suffix=8)
        length = struct.unpack("<I", tail[:4])[0]
        if tail[4:] != b"PAR1" or not 0 < length < 5_000_000:
            raise ValueError("Invalid bounded Parquet footer")
        start = source.size - length - 8
        footer = source.fetch(start, source.size - 8)
        metadata = pq.read_metadata(io.BytesIO(b"PAR1" + footer + tail))
        with os.fdopen(os.open(cache_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as cached_file:
            cached_file.write(footer + tail)
        cache_meta_path.write_text(json.dumps({"url": url, "size": source.size, "footer_start": start,
                                              "sha256": hashlib.sha256(footer + tail).hexdigest(),
                                              "num_rows": metadata.num_rows}) + "\n")
        return i, source, start, footer + tail, metadata

    with ThreadPoolExecutor(max_workers=4) as pool:
        shards = sorted(pool.map(footer_for, range(86)), key=lambda x: x[0])
    sizes = [s[4].num_rows for s in shards]
    if sum(sizes) != ROW_COUNT:
        raise ValueError("Pinned Parquet population differs from the declared design")
    boundaries = [0]
    for size in sizes:
        boundaries.append(boundaries[-1] + size)
    selected_by_shard = {}
    for design in sample_design(args.sample_count, args.seed):
        if design["source_row_index"] in completed_ids:
            continue
        shard_index = bisect_right(boundaries, design["source_row_index"]) - 1
        local = design["source_row_index"] - boundaries[shard_index]
        metadata = shards[shard_index][4]
        group_boundaries = [0]
        for g in range(metadata.num_row_groups):
            group_boundaries.append(group_boundaries[-1] + metadata.row_group(g).num_rows)
        group = bisect_right(group_boundaries, local) - 1
        selected_by_shard.setdefault(shard_index, {}).setdefault(group, []).append(
            (local - group_boundaries[group], design, local)
        )
    specs, preflight_bytes = {}, budget.used
    for i, group_map in selected_by_shard.items():
        metadata = shards[i][4]
        columns = [j for j in range(metadata.num_columns) if metadata.schema.column(j).path in projected]
        if {metadata.schema.column(j).path for j in columns} != set(projected):
            raise ValueError("Projected schema mismatch")
        ranges = []
        for g in group_map:
            for j in columns:
                chunk = metadata.row_group(g).column(j)
                start = min(o for o in [chunk.dictionary_page_offset, chunk.data_page_offset]
                            if o is not None and o >= 4)
                ranges.append((start, start + chunk.total_compressed_size))
        merged = []
        for start, stop in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        specs[i] = merged
        preflight_bytes += sum(stop - start for start, stop in merged)
    print(json.dumps({"event": "parquet_preflight", "budget_bytes": budget.limit,
                      "minimum_transfer_bytes": preflight_bytes, "footer_bytes_downloaded": budget.used}), flush=True)
    if preflight_bytes > budget.limit:
        raise RuntimeError("Exact predetermined sample exceeds declared transfer cap; no text chunks fetched")
    records = list(existing)
    def dumps(value):
        return json.dumps(value, ensure_ascii=False, default=lambda x: x.isoformat())
    with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as checkpoint:
        for i in sorted(selected_by_shard):
            _, source, footer_start, footer_data, metadata = shards[i]
            with ThreadPoolExecutor(max_workers=4) as pool:
                segments = list(pool.map(lambda pair: (pair[0], source.fetch(*pair)), specs[i]))
            segments.append((footer_start, footer_data))
            reader = CachedRanges(source.size, segments)
            pf = pq.ParquetFile(reader, metadata=metadata, pre_buffer=False)
            for group, chosen in selected_by_shard[i].items():
                table = pf.read_row_group(group, columns=projected, use_threads=False)
                for offset, design, local in chosen:
                    row = table.slice(offset, 1).to_pylist()[0]
                    record = {
                        "dataset": DATASET, "dataset_revision": REVISION, "sampling": design,
                        "row_idx": design["source_row_index"], "truncated_cells": [], "row": row,
                        "retrieval": {"method": "pinned_parquet_projected_row_group", "url": source.url,
                                      "retrieved_at": utc_now(), "source_shard": i, "source_row_group": group,
                                      "source_row_index_in_shard": local, "projected_fields": projected,
                                      "footer_sha256": hashlib.sha256(footer_data).hexdigest(),
                                      "content_row_sha256": hashlib.sha256(dumps(row).encode()).hexdigest()},
                    }
                    # Normalize datetimes to stable ISO JSON now, not only on disk.
                    record = json.loads(dumps(record))
                    records.append(record)
                    checkpoint.write(dumps(record) + "\n")
                    checkpoint.flush()
            print(json.dumps({"downloaded_records": len(records), "of": args.sample_count,
                              "completed_shard": i, "transferred_bytes": budget.used}), flush=True)
    after = check_source_revision()
    records.sort(key=lambda x: x["row_idx"])
    with os.fdopen(os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        for record in records:
            output.write(dumps(record) + "\n")
    manifest = {
        "status": "complete", "started_at": started, "completed_at": utc_now(),
        "dataset": DATASET, "dataset_revision": REVISION, "population_rows": ROW_COUNT,
        "population": "Released conversation rows; not unique users or requests",
        "sample_count": len(records), "seed": args.seed,
        "sampling_method": "One uniformly random row from each exhaustive equal-width row-index stratum",
        "acquisition_method": "Pinned Parquet projected row groups via strict byte-range GETs",
        "source_before": before, "source_after": after, "projected_fields": projected,
        "truncated_record_count": 0, "maximum_transfer_bytes": budget.limit,
        "range_payload_bytes": budget.used, "range_get_attempts": budget.requests_count,
        "range_payload_scope": "This invocation only; earlier attempts are separately described",
        "resumed_record_count": len(existing),
        "preflight_minimum_bytes": preflight_bytes, "parquet_shard_row_counts": sizes,
        "raw_file": str(raw), "raw_file_bytes": raw.stat().st_size,
        "raw_file_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "gold_labels_generated": False, "paid_inference_calls": 0,
        "earlier_acquisition_attempts": "Any earlier attempts must be documented separately; transfer totals here cover this invocation only.",
        "limitations": ["No labels or eligibility inferred by retrieval.",
                        "Rows outside the sample share fetched Parquet pages but are not retained as sample records.",
                        "First-response normalization and duplicate audit remain downstream.",
                        "Public release excludes flagged-toxic conversations."]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    partial.unlink()
    print(json.dumps({"status": "complete", "sample_count": len(records),
                      "range_payload_bytes": budget.used, "raw_file_sha256": manifest["raw_file_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
