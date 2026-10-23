"""Development-only ablation of long-request representations, with no API calls.

The experiment measures what a short encoder actually sees. It does not assign
quality labels or measure production-regression prediction. Original texts are
never rewritten; representation plans reference positions in their full token
sequence. Shared prefixes are fitted exclusively on the selected development
cohort and are a mechanical recurrence heuristic, not semantic boilerplate gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.production_eval.audit_records import load_audits
from experiments.production_eval.selection_methods import select

ENCODERS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
}
VARIANTS = ("head", "tail", "head_tail", "chunk_mean", "prefix_payload_mix")
PAIRS = (
    ("abstract_template", "ae3ecfa8d8cf755745b1698f8aece6dd3056441f302a86c9d923028f4f37bb4e", "9d44f170708c1bd93b9a0c23cd3e819489bc05f7443725d69368404c70a677b8"),
    ("extraction_template", "057b5af185584316f51e6e1859e2c31d561f57c0f306a92b33dc32f7ad3d8fde", "2f0340e7e9dc1566dda143cb4c5342c846e34ebcd19fbf504cd4841f1093fd41"),
    ("recipe_template", "205af5c3b7a68b7390c3b6461cd7d7693d1a0c2e68e13ac6416c918042be1de6", "0f059fefbe1a99062ca9c9c02f0804dd8bca0d4a8c33c4ebf2196380e64af67b"),
)


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def save_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def common_prefix(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def fit_prefixes(sequences, min_shared=64, min_residual=16):
    """Find exact branching prefixes shared by two construction inputs.

    Complete duplicate/prefix-only sequences are deliberately not templates.
    At least min_residual tokens must remain in both supporting examples. The
    rule sees input tokens only, never outcomes, source answers or future data.
    """
    prefixes = set()
    for i, left in enumerate(sequences):
        for right in sequences[i + 1:]:
            count = common_prefix(left, right)
            if min_shared <= count <= min(len(left), len(right)) - min_residual:
                prefixes.add(tuple(left[:count]))
    return sorted(prefixes, key=lambda value: (-len(value), value))


def matching_prefix(sequence, prefixes):
    return next((len(prefix) for prefix in prefixes if tuple(sequence[:len(prefix)]) == prefix), 0)


def make_plan(length, capacity, variant, prefix_length=0):
    """Return (source-position segments, normalized aggregation weights)."""
    if length < 1 or capacity < 2:
        raise ValueError("Need a nonempty token input and capacity at least two")
    if variant not in VARIANTS or not 0 <= prefix_length < length:
        raise ValueError("Invalid variant or prefix length")
    if variant in {"head", "tail", "head_tail"}:
        if length <= capacity:
            positions = list(range(length))
        elif variant == "head":
            positions = list(range(capacity))
        elif variant == "tail":
            positions = list(range(length - capacity, length))
        else:
            first = (capacity + 1) // 2
            positions = list(range(first)) + list(range(length - (capacity - first), length))
        return [positions], [1.0]

    def chunk(start, stop):
        return [list(range(at, min(at + capacity, stop))) for at in range(start, stop, capacity)]

    if variant == "chunk_mean" or prefix_length == 0:
        segments = chunk(0, length)
        return segments, [len(segment) / length for segment in segments]
    # Each side gets equal total mass, regardless of template/payload length.
    # The long template is retained rather than silently discarded.
    prefix, residual = chunk(0, prefix_length), chunk(prefix_length, length)
    weights = [0.5 * len(segment) / prefix_length for segment in prefix]
    weights += [0.5 * len(segment) / (length - prefix_length) for segment in residual]
    return prefix + residual, weights


def unit(array):
    return array / np.linalg.norm(array, axis=-1, keepdims=True).clip(1e-12)


def aggregate(piece_embeddings, groups, weights):
    return unit(np.stack([np.asarray(weight) @ piece_embeddings[group]
                          for group, weight in zip(groups, weights)])).astype(np.float32)


def encode_pieces(model, pieces, device, batch_size=32):
    import torch
    result = []
    with torch.inference_mode():
        for start in range(0, len(pieces), batch_size):
            data = [{"input_ids": piece, "attention_mask": [1] * len(piece)}
                    for piece in pieces[start:start + batch_size]]
            features = model.tokenizer.pad(data, padding=True, return_tensors="pt")
            features = {key: value.to(device) for key, value in features.items()}
            result.append(model(features)["sentence_embedding"].cpu().numpy())
    return np.concatenate(result)


def wrap_special_tokens(tokenizer, content):
    # Both pinned encoders use one CLS token followed by content and one SEP.
    # New Transformers tokenizers removed build_inputs_with_special_tokens;
    # derive and check the wrapper rather than hard-coding numeric IDs.
    wrapper = tokenizer("", add_special_tokens=True)["input_ids"]
    if wrapper != [tokenizer.cls_token_id, tokenizer.sep_token_id]:
        raise RuntimeError("Unsupported single-sequence special-token template")
    return [wrapper[0], *content, wrapper[1]]


def run_variant(model, rows, sequences, prefixes, variant, device, folder):
    import torch
    limit = model.max_seq_length
    capacity = limit - model.tokenizer.num_special_tokens_to_add(pair=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    ids = [row["example_id"] for row in rows]
    pieces, piece_map, groups, weights, token_records = [], {}, [], [], []
    signatures = defaultdict(list)
    prefix_lengths = []
    for row, sequence in zip(rows, sequences):
        prefix_length = matching_prefix(sequence, prefixes) if variant == "prefix_payload_mix" else 0
        prefix_lengths.append(prefix_length)
        segments, weight = make_plan(len(sequence), capacity, variant, prefix_length)
        group, source = [], []
        for segment in segments:
            actual = wrap_special_tokens(model.tokenizer, [sequence[i] for i in segment])
            assert len(actual) <= limit
            key = tuple(actual)
            if key not in piece_map:
                piece_map[key] = len(pieces)
                pieces.append(actual)
            group.append(piece_map[key])
            source.append(actual)
        positions = sorted({i for segment in segments for i in segment})
        signature = digest({"actual_tokens": source, "weights": weight})
        signatures[signature].append(row["example_id"])
        groups.append(group)
        weights.append(weight)
        token_records.append({
            "example_id": row["example_id"], "input_hash": row["input_hash"],
            "full_content_tokens": len(sequence), "retained_distinct_source_positions": len(positions),
            "omitted_content_tokens": len(sequence) - len(positions), "segment_indices": group,
            "segment_source_positions": segments, "segment_weights": weight,
            "prefix_tokens": prefix_length, "representation_token_hash": signature,
        })
    piece_embeddings = encode_pieces(model, pieces, device)
    embeddings = aggregate(piece_embeddings, groups, weights)
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) < 0.99):
        raise RuntimeError("Invalid normalized embeddings")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    path = folder / (variant + ".npz")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as stream:
        np.savez_compressed(stream, example_ids=np.asarray(ids), input_hashes=np.asarray([r["input_hash"] for r in rows]), embeddings=embeddings)
    save_json(folder / (variant + "_tokens.json"), {
        "actual_input_tokens_per_unique_encoded_piece": pieces,
        "rows": token_records,
        "scope": "Actual encoder token IDs are local research data; segments index each original full token sequence. No original input was modified.",
    })
    lengths = np.asarray([len(sequence) for sequence in sequences])
    retained = np.asarray([record["retained_distinct_source_positions"] for record in token_records])
    result = {
        "variant": variant, "max_seq_length_including_special_tokens": limit,
        "content_capacity": capacity, "requests_with_tokens_omitted": int(sum(retained < lengths)),
        "mean_fraction_source_tokens_retained": float(np.mean(retained / lengths)),
        "total_source_tokens": int(lengths.sum()), "total_retained_source_tokens": int(retained.sum()),
        "distinct_representation_token_sequences": len(signatures),
        "token_collision_groups": [group for group in signatures.values() if len(group) > 1],
        "prefix_matched_requests": int(sum(length > 0 for length in prefix_lengths)),
        "logical_segments": sum(len(group) for group in groups), "unique_encoded_segments": len(pieces),
        "physical_nonpadding_encoder_tokens_including_special_tokens": sum(len(piece) for piece in pieces),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
        "embedding_file": str(path.relative_to(ROOT)), "embedding_sha256": digest(path.read_bytes()),
    }
    if variant == "head":
        native = model.tokenize([row["input_text"] for row in rows])
        actual_native = [a[b.bool()].tolist() for a, b in zip(native["input_ids"], native["attention_mask"])]
        if actual_native != [pieces[group[0]] for group in groups]:
            raise RuntimeError("Reconstructed head tokenization differs from native SentenceTransformer")
        native_embedding = model.encode([row["input_text"] for row in rows], batch_size=32,
                                        show_progress_bar=False, convert_to_numpy=True)
        delta = float(np.max(np.abs(native_embedding - embeddings)))
        if delta > 5e-5:
            raise RuntimeError("Native head embedding verification failed")
        result["native_head_max_abs_embedding_error"] = delta
        result["native_head_exact_token_agreement"] = True
    return embeddings, result, token_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "output/research/production_eval/execution"
    parser.add_argument("--input", type=Path, default=base / "audit_requests.jsonl")
    parser.add_argument("--audit", type=Path, default=base / "audit.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/research/production_eval/experiments_v2/representations")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers.utils import logging
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    torch.set_num_threads(8)
    if args.device.startswith("cuda"):
        torch.cuda.init()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    annotations = load_audits(args.audit, rows)
    rows = sorted([row for row in rows if row["technical_eligible"]], key=lambda row: row["example_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for encoder, (name, revision) in ENCODERS.items():
        snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
        if not snapshot.is_dir():
            raise RuntimeError("Pinned cached model missing; downloads are prohibited")
        start = time.monotonic()
        model = SentenceTransformer(str(snapshot), device=args.device, local_files_only=True)
        model.eval()
        load_seconds = time.monotonic() - start
        for cohort in ("technical", "machine_eligible"):
            chosen = rows if cohort == "technical" else [row for row in rows if annotations[row["input_hash"]]["eligible_machine_prediction"]]
            ids = [row["example_id"] for row in chosen]
            index = {identifier: i for i, identifier in enumerate(ids)}
            texts = [row["input_text"].strip() for row in chosen]
            if model[0].do_lower_case:
                texts = [text.lower() for text in texts]
            sequences = model.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            prefixes = fit_prefixes(sequences)
            folder = args.output_dir / encoder / cohort
            folder.mkdir(parents=True, exist_ok=True)
            save_json(folder / "prefix_fit.json", {
                "construction_ids": ids, "construction_input_hashes": [row["input_hash"] for row in chosen],
                "min_shared_tokens": 64, "min_residual_tokens": 16,
                "prefixes": [{"token_ids": list(prefix), "token_sha256": digest(prefix),
                              "support_ids": [row["example_id"] for row, seq in zip(chosen, sequences) if tuple(seq[:len(prefix)]) == prefix]}
                             for prefix in prefixes],
                "warning": "Exact recurring prefixes may contain substantive shared content. They are retained with equal template/payload mass, not labeled as disposable boilerplate. All fitting inputs are development data.",
            })
            budget = 100 if cohort == "technical" else 15
            strata = [annotations[row["input_hash"]]["annotation"]["task_family"] for row in chosen]
            variants, embedding_cache, selection_cache = [], {}, {}
            for variant in VARIANTS:
                embedding, result, records = run_variant(model, chosen, sequences, prefixes, variant, args.device, folder)
                embedding_cache[variant] = embedding
                result["pair_diagnostics"] = []
                for label, left, right in PAIRS:
                    if left not in index or right not in index:
                        continue
                    a, b = index[left], index[right]
                    lcp = common_prefix(sequences[a], sequences[b])
                    result["pair_diagnostics"].append({
                        "pair": label, "ids": [left, right], "shared_prefix_content_tokens": lcp,
                        "cosine_similarity": float(np.clip(embedding[a] @ embedding[b], -1, 1)),
                        "actual_representation_tokens_equal": records[a]["representation_token_hash"] == records[b]["representation_token_hash"],
                        "left_differing_suffix_positions_seen": sum(pos >= lcp for segment in records[a]["segment_source_positions"] for pos in segment),
                        "right_differing_suffix_positions_seen": sum(pos >= lcp for segment in records[b]["segment_source_positions"] for pos in segment),
                    })
                result["selections"] = []
                for policy in ("centroid_proximity", "facility_location"):
                    start = time.monotonic()
                    selection = select(policy, ids, budget, embeddings=embedding, strata=strata, seed=17)
                    selected_ids = [ids[i] for i in selection.indices]
                    selection_cache[(variant, policy)] = set(selected_ids)
                    result["selections"].append({"policy": policy, "budget": budget,
                                                "elapsed_seconds": time.monotonic() - start,
                                                "selected_ids": selected_ids,
                                                "sample_weights": selection.sample_weights.tolist(),
                                                "task_family_counts": dict(Counter(strata[i] for i in selection.indices)),
                                                "metadata": selection.metadata})
                variants.append(result)
                print(json.dumps({"encoder": encoder, "cohort": cohort, "variant": variant,
                                  "collisions": len(chosen) - result["distinct_representation_token_sequences"],
                                  "omitted_requests": result["requests_with_tokens_omitted"],
                                  "seconds": round(result["elapsed_seconds"], 3)}), flush=True)
            for variant in variants:
                for selected in variant["selections"]:
                    base_selected = selection_cache[("head", selected["policy"])]
                    current_selected = set(selected["selected_ids"])
                    selected["intersection_with_native_head"] = len(base_selected & current_selected)
                    selected["jaccard_with_native_head"] = len(base_selected & current_selected) / len(base_selected | current_selected)
            report = {"encoder": encoder, "model": name, "revision": revision, "device": args.device,
                      "model_load_seconds": load_seconds, "cohort": cohort, "pool_size": len(chosen),
                      "budget": budget, "prefix_fit_scope": "this construction cohort only",
                      "variant_definitions": {
                          "head": "Native leading tokens, reconstructed and checked against native tokens/embeddings.",
                          "tail": "Last content-capacity tokens with model special tokens restored.",
                          "head_tail": "Half capacity from each end, joined without overlap; artificial adjacency is intentional.",
                          "chunk_mean": "Nonoverlapping full-input chunks; content-token-count-weighted mean of unit embeddings, then renormalize.",
                          "prefix_payload_mix": "Exact construction-pool shared branching prefix (>=64 tokens, >=16 residual). Weight mean prefix and mean residual equally; each side uses token-weighted chunks. No match: chunk_mean.",
                      }, "variants": variants}
            save_json(folder / "results.json", report)
            all_results.append(report)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary = {
        "status": "development_representation_ablation_completed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": digest(args.input.read_bytes()), "audit_sha256": digest(args.audit.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "api_calls": 0, "model_downloads": 0, "human_labels_created": 0, "outcome_labels_created": 0,
        "scope": "252 frozen technical development inputs and their31 machine-predicted eligible subset. Token retention, collision removal, compute and selection stability only. No quality, outcome validity, population, cross-time or selector-superiority claim.",
        "diagnostic_pairs_source": "execution/annotation_spotcheck.md; consciously selected development examples, not a validation set.",
        "physical_work_note": "Exact duplicate token pieces are encoded once per variant. Time excludes file writes, native verification, model loading and selectors; includes preparation, forward passes and aggregation. Peak allocation includes resident model. GPU warmup and order can affect time.",
        "results": all_results,
    }
    save_json(args.output_dir / "results.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir / "results.json")}), flush=True)


if __name__ == "__main__":
    main()
