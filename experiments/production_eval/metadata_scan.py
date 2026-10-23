#!/usr/bin/env python3
"""Pinned WildChat metadata census through strictly bounded HTTP byte ranges.

Fetches only Parquet footers and role/timestamp/created/language/model chunks.
No text column is fetched. Shards are processed sequentially; chunk prefetch is
bounded within each shard. This counts released conversation rows, not eligible
tasks, independent users, gold outcomes, or unfiltered production traffic.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import time

import pyarrow.parquet as pq
import requests

DATASET = "allenai/WildChat-4.8M"
REVISION = "c827c6df8fcf008219ffaffa4d1dd77491099367"
SHARD_COUNT = 86
EXPECTED_SOURCE_ROWS = 3_199_860
MAX_BYTES = 300_000_000
COLUMNS = [
    "conversation.list.element.role", "conversation.list.element.timestamp",
    "conversation.list.element.created", "language", "model",
]
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def now():
    return datetime.now(timezone.utc).isoformat()


class TransferBudget:
    def __init__(self, limit, used=0, requests_count=0):
        self.limit, self.used, self.requests_count = limit, used, requests_count
        self.reserved = 0
        self.lock = threading.Lock()

    def reserve(self, count):
        with self.lock:
            if self.used + self.reserved + count > self.limit:
                raise RuntimeError("Cumulative network payload budget would be exceeded")
            self.reserved += count
            self.requests_count += 1

    def finish(self, expected, actual):
        with self.lock:
            self.reserved -= expected
            self.used += actual


class RangeSource:
    def __init__(self, url, budget):
        self.url, self.budget, self.size = url, budget, None
        # Resolve the pinned Hub URL once, then reuse its signed CDN URL in
        # memory. Resolving every column chunk needlessly exhausts shared Hub
        # rate limits. This ephemeral URL is never logged or persisted.
        self.download_url = url
        self.local = threading.local()

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def fetch(self, start=None, stop=None, suffix=None):
        expected = suffix if suffix is not None else stop - start
        if expected <= 0:
            raise ValueError("Empty or invalid range")
        requested = f"bytes=-{suffix}" if suffix is not None else f"bytes={start}-{stop - 1}"
        for attempt in range(3):
            actual = 0
            self.budget.reserve(expected)
            try:
                with self.session().get(
                    self.download_url, headers={"Range": requested, "Accept-Encoding": "identity"},
                    timeout=(20, 60), stream=True,
                ) as response:
                    # Do not consume bodies on rejected responses (especially a
                    # server that ignored Range and attempted a full shard).
                    if response.status_code != 206:
                        rate_headers = {
                            key: value for key, value in response.headers.items()
                            if key.lower() in {"retry-after", "ratelimit", "ratelimit-policy",
                                               "x-ratelimit-limit", "x-ratelimit-remaining",
                                               "x-ratelimit-reset"}
                        }
                        raise RuntimeError(
                            f"Expected HTTP206 for {requested}; got {response.status_code}; "
                            f"rate_headers={json.dumps(rate_headers, sort_keys=True)}"
                        )
                    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if not match:
                        raise RuntimeError("Missing or malformed Content-Range")
                    got_start, got_end, size = map(int, match.groups())
                    expected_start = size - suffix if suffix is not None else start
                    expected_end = size - 1 if suffix is not None else stop - 1
                    if (got_start, got_end) != (expected_start, expected_end):
                        raise RuntimeError("Content-Range differs from exact requested range")
                    if self.size is not None and size != self.size:
                        raise RuntimeError("Pinned source size changed during scan")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise RuntimeError("Unexpected encoded range response")
                    if int(response.headers.get("Content-Length", expected)) != expected:
                        raise RuntimeError("Content-Length differs from exact requested range")
                    chunks = []
                    for chunk in response.iter_content(chunk_size=min(65536, expected)):
                        actual += len(chunk)
                        if actual > expected:
                            raise RuntimeError("Response exceeded its exact range length")
                        chunks.append(chunk)
                    if actual != expected:
                        raise requests.ConnectionError("Truncated HTTP range response")
                    self.size = size
                    self.download_url = response.url
                    return b"".join(chunks)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
            finally:
                self.budget.finish(expected, actual)


class CachedRanges(io.RawIOBase):
    """Virtual file exposing only explicitly prefetched permitted byte ranges."""
    def __init__(self, size, segments):
        super().__init__()
        self.size, self.position = size, 0
        self.segments = sorted(segments)
        self.starts = [start for start, _ in self.segments]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, count=-1):
        stop = self.size if count < 0 else min(self.size, self.position + count)
        if stop <= self.position:
            return b""
        pieces = []
        while self.position < stop:
            slot = bisect_right(self.starts, self.position) - 1
            if slot < 0:
                raise RuntimeError("Arrow attempted an unapproved byte range")
            start, content = self.segments[slot]
            within = self.position - start
            if within >= len(content):
                raise RuntimeError("Arrow attempted to read a nonprojected column or gap")
            take = min(stop - self.position, len(content) - within)
            pieces.append(content[within:within + take])
            self.position += take
        return b"".join(pieces)


def parse_time(value, allow_unix=False):
    if value is None:
        return None
    try:
        if isinstance(value, datetime):
            result = value
        elif allow_unix and isinstance(value, (int, float)):
            result = datetime.fromtimestamp(value, tz=timezone.utc)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return None
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def update_counts(row, counts):
    counts["rows"] += 1
    language = row.get("language") or "<missing>"
    model = row.get("model") or "<missing>"
    counts["language_counts"][language] += 1
    counts["model_counts"][model] += 1
    conversation = row.get("conversation") or []
    first_user = next(((i, t) for i, t in enumerate(conversation) if t.get("role") == "user"), None)
    first_assistant = next(((i, t) for i, t in enumerate(conversation) if t.get("role") == "assistant"), None)
    if not first_user:
        counts["structure_counts"]["missing_user"] += 1
    if not first_assistant:
        counts["structure_counts"]["missing_assistant"] += 1
    if first_user and first_assistant and first_assistant[0] < first_user[0]:
        counts["structure_counts"]["assistant_before_first_user"] += 1
    user_time = parse_time(first_user[1].get("timestamp")) if first_user else None
    assistant_time = parse_time(first_assistant[1].get("timestamp")) if first_assistant else None
    created = parse_time(first_assistant[1].get("created"), allow_unix=True) if first_assistant else None
    counts["first_assistant_timestamp_month_counts"][assistant_time.strftime("%Y-%m") if assistant_time else "<missing_or_invalid>"] += 1
    counts["first_user_timestamp_month_counts"][user_time.strftime("%Y-%m") if user_time else "<missing_or_invalid>"] += 1
    if user_time:
        chosen, source = user_time, "first_user_timestamp"
    elif assistant_time:
        chosen, source = assistant_time, "first_assistant_timestamp_proxy"
    elif created:
        chosen, source = created, "first_assistant_created_proxy"
    else:
        chosen, source = None, "unavailable"
    month = chosen.strftime("%Y-%m") if chosen else "<missing_or_invalid>"
    counts["request_time_month_counts"][month] += 1
    counts["request_time_source_counts"][source] += 1
    counts["month_language_counts"][month + "|" + language] += 1
    if chosen:
        iso = chosen.isoformat()
        counts["minimum_request_time"] = min(counts.get("minimum_request_time") or iso, iso)
        counts["maximum_request_time"] = max(counts.get("maximum_request_time") or iso, iso)
    if assistant_time and created and abs((assistant_time - created).total_seconds()) > 60:
        counts["structure_counts"]["assistant_timestamp_created_disagree_over_60_seconds"] += 1
    if user_time and assistant_time and user_time > assistant_time:
        counts["structure_counts"]["first_user_timestamp_after_first_assistant"] += 1


COUNT_FIELDS = [
    "language_counts", "model_counts", "structure_counts",
    "first_assistant_timestamp_month_counts", "first_user_timestamp_month_counts",
    "request_time_month_counts", "request_time_source_counts", "month_language_counts",
]


def scan_shard(index, budget, workers):
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-{index:05d}-of-{SHARD_COUNT:05d}.parquet"
    before_bytes, before_requests = budget.used, budget.requests_count
    source = RangeSource(url, budget)
    tail = source.fetch(suffix=8)
    if tail[4:] != b"PAR1":
        raise RuntimeError("Missing Parquet trailer magic")
    footer_length = struct.unpack("<I", tail[:4])[0]
    if not 0 < footer_length < min(source.size - 8, 5_000_000):
        raise RuntimeError("Unexpected Parquet footer length")
    footer_start = source.size - footer_length - 8
    footer = source.fetch(footer_start, source.size - 8)
    metadata = pq.read_metadata(io.BytesIO(b"PAR1" + footer + tail))
    selected_columns = [i for i in range(metadata.num_columns)
                        if metadata.schema.column(i).path in COLUMNS]
    found = [metadata.schema.column(i).path for i in selected_columns]
    if set(found) != set(COLUMNS):
        raise RuntimeError(f"Pinned projection schema mismatch: {found}")
    ranges = []
    for group in range(metadata.num_row_groups):
        for column in selected_columns:
            chunk = metadata.row_group(group).column(column)
            offsets = [o for o in [chunk.dictionary_page_offset, chunk.data_page_offset] if o is not None and o >= 4]
            start = min(offsets)
            stop = start + chunk.total_compressed_size
            if stop > footer_start:
                raise RuntimeError("Column chunk extends into the footer")
            ranges.append((start, stop))
    # Merge only adjacent or overlapping permitted chunks; never bridge a gap
    # that could contain a text column.
    merged = []
    for start, stop in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    planned = sum(stop - start for start, stop in merged)
    if budget.used + planned > budget.limit:
        raise RuntimeError("Remaining byte budget cannot finish this shard's metadata projection")
    def fetch_range(pair):
        start, stop = pair
        return start, source.fetch(start, stop)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        segments = list(pool.map(fetch_range, merged))
    # Metadata explicitly supplies the source column offsets, so Arrow needs
    # only these projected ranges; the source footer remains available too.
    segments.extend([(footer_start, footer + tail)])
    reader = CachedRanges(source.size, segments)
    parquet = pq.ParquetFile(reader, metadata=metadata, pre_buffer=False)
    counts = {key: Counter() for key in COUNT_FIELDS}
    counts["rows"] = 0
    for group in range(metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=COLUMNS, use_threads=False)
        for row in table.to_pylist():
            update_counts(row, counts)
    if counts["rows"] != metadata.num_rows:
        raise RuntimeError("Projected rows do not match the source footer count")
    if sum(counts["request_time_month_counts"].values()) != metadata.num_rows:
        raise RuntimeError("Month counts failed row conservation")
    return {
        "shard_index": index, "source_url": url, "source_file_bytes": source.size,
        "footer_bytes": footer_length, "footer_sha256": hashlib.sha256(footer).hexdigest(),
        "row_groups": metadata.num_row_groups, "projected_columns": found,
        "projected_compressed_bytes": planned, "network_payload_bytes": budget.used - before_bytes,
        "http_range_requests": budget.requests_count - before_requests,
        "completed_at": now(), "counts": counts,
    }


def aggregate(shards):
    result = {key: Counter() for key in COUNT_FIELDS}
    result["rows"] = sum(shard["counts"]["rows"] for shard in shards)
    for shard in shards:
        for key in COUNT_FIELDS:
            result[key].update(shard["counts"][key])
    for key, fn in [("minimum_request_time", min), ("maximum_request_time", max)]:
        values = [s["counts"][key] for s in shards if s["counts"].get(key)]
        result[key] = fn(values) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/research/production_eval/execution/monthly_metadata.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-shards", type=int, default=SHARD_COUNT)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12 or not 1 <= args.limit_shards <= SHARD_COUNT:
        parser.error("Use 1–12 workers and 1–86 shards")
    if not 1 <= args.max_bytes <= MAX_BYTES:
        parser.error("The hard cumulative payload limit is 300,000,000 bytes")
    if args.output.exists():
        if not args.resume:
            parser.error("Output exists; use --resume to retain the cumulative budget and completed shards")
        report = json.loads(args.output.read_text())
        if report["dataset_revision"] != REVISION or report["dataset"] != DATASET:
            parser.error("Cannot resume a different source snapshot")
    else:
        report = {
            "schema_version": "1.0", "started_at": now(), "dataset": DATASET,
            "dataset_revision": REVISION, "total_source_shards": SHARD_COUNT,
            "population": "Released source conversation rows; not independent users or eligible tasks",
            "projection": COLUMNS, "source_text_columns_fetched": False,
            "timestamp_rule": "First user timestamp where parseable, else first assistant timestamp proxy, else first assistant created Unix-seconds proxy; unavailable retained",
            "timezone_rule": "Naive source datetimes interpreted as UTC; this normalization convention does not verify source timezone semantics",
            "primary_time_evidence": "Exact first-assistant timestamp month counts are also retained separately from the composite request-time proxy",
            "eligibility_assessed": False, "gold_outcomes_assessed": False,
            "network_payload_bytes": 0, "http_range_requests": 0,
            "maximum_network_payload_bytes": args.max_bytes,
            "budget_excludes": "HTTP headers, TLS framing, and rejected bodies that are not consumed; all consumed response payload including retries is counted",
            "shards": [],
        }
    budget = TransferBudget(args.max_bytes, report["network_payload_bytes"], report["http_range_requests"])
    completed = {shard["shard_index"] for shard in report["shards"]}
    if len(completed) != len(report["shards"]) or any(not 0 <= i < SHARD_COUNT for i in completed):
        parser.error("Resume checkpoint has duplicate or invalid shard indices")
    if any(f"/resolve/{REVISION}/" not in shard["source_url"] for shard in report["shards"]):
        parser.error("Resume checkpoint contains an unpinned or different source revision")
    def save(status):
        totals = aggregate(report["shards"])
        report.update(status=status, updated_at=now(), scanned_shards=len(report["shards"]),
                      network_payload_bytes=budget.used, http_range_requests=budget.requests_count,
                      counts=totals,
                      expected_source_rows=EXPECTED_SOURCE_ROWS,
                      unsuccessful_attempt_payload_bytes=(
                          budget.used - sum(s["network_payload_bytes"] for s in report["shards"])
                      ),
                      complete=(status == "complete" and len(report["shards"]) == SHARD_COUNT
                                and totals["rows"] == EXPECTED_SOURCE_ROWS))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    save("running")
    try:
        for index in range(args.limit_shards):
            if index in completed:
                continue
            shard = scan_shard(index, budget, args.workers)
            report["shards"].append(shard)
            report.pop("error", None)
            save("running")
            print(json.dumps({"scanned_shards": len(report["shards"]),
                              "rows": report["counts"]["rows"],
                              "network_payload_bytes": budget.used}), flush=True)
        if len(report["shards"]) == SHARD_COUNT and report["counts"]["rows"] != EXPECTED_SOURCE_ROWS:
            raise RuntimeError("All-shard row total differs from the pinned source population metadata")
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "at": now()}
        save("partial_error")
        raise
    save("complete" if len(report["shards"]) == SHARD_COUNT else "partial_requested_limit")


if __name__ == "__main__":
    main()
