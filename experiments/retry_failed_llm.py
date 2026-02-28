#!/usr/bin/env python3
"""
Clear failed LLM generations and their results so exp_multi_llm.py can retry them.

Identifies cache files with <THRESHOLD generated texts (content filter refusals),
deletes those cache files, and removes the corresponding results entries.
Then re-running exp_multi_llm.py will regenerate only the failed configs.
"""

import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache" / "llm_generations"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "multi_llm"

SEVERE_REFUSAL_THRESHOLD = 20

MULTI_LLM_MODELS = {
    "gemini-3-flash-preview", "gpt-5-mini", "claude-haiku-4-5",
    "kimi-k2.5", "glm-5"
}

# Only retry these datasets (the ones used in multi-LLM experiment)
EXPERIMENT_DATASETS = {
    "20newsgroups", "sms_spam", "hate_speech_davidson",
    "ag_news", "dbpedia14", "emotion", "20newsgroups_20class"
}


def is_experiment_dataset(dataset_name):
    for ds in EXPERIMENT_DATASETS:
        if dataset_name.startswith(ds + "_") or dataset_name == ds:
            return True
    return False


def main():
    # Step 1: Find failed cache files
    failed = []  # (model, dataset, class_name, cache_file_path)

    for model_dir in sorted(CACHE_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name not in MULTI_LLM_MODELS:
            continue
        model_name = model_dir.name

        for f in model_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)

            stored_model = data.get("model", "")
            if stored_model != model_name:
                continue

            dataset = data.get("dataset", "")
            if not is_experiment_dataset(dataset):
                continue

            n_texts = len(data.get("texts", []))
            if n_texts < SEVERE_REFUSAL_THRESHOLD:
                class_name = data.get("class_name", "?")
                failed.append((model_name, dataset, class_name, f, n_texts))

    if not failed:
        print("No failed generations found. Nothing to retry.")
        return

    print(f"Found {len(failed)} failed generations (< {SEVERE_REFUSAL_THRESHOLD} texts):\n")
    for model, dataset, cls, path, n in sorted(failed):
        print(f"  {model:25s} {dataset:35s} {cls:25s} ({n} texts)")

    # Step 2: Delete failed cache files
    print(f"\nDeleting {len(failed)} cache files...")
    for _, _, _, path, _ in failed:
        path.unlink()
        print(f"  Deleted {path.name}")

    # Step 3: Remove affected results
    # A result is affected if its (dataset, llm_model) combo had any failed class
    affected_combos = set()
    for model, dataset, cls, _, _ in failed:
        affected_combos.add((dataset, model))

    for results_file in ["final_results.json", "partial_results.json"]:
        rpath = RESULTS_DIR / results_file
        if not rpath.exists():
            continue

        with open(rpath) as f:
            data = json.load(f)

        original_count = len(data.get("results", []))
        kept = []
        removed = 0

        for r in data.get("results", []):
            combo = (r["dataset"], r["llm_model"])
            if r["augmentation_method"] == "soft_weighted" and combo in affected_combos:
                removed += 1
            else:
                kept.append(r)

        data["results"] = kept
        with open(rpath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  {results_file}: {original_count} -> {len(kept)} results ({removed} removed)")

    print(f"\n{'=' * 70}")
    print(f"Cleared {len(failed)} cache files and removed affected results.")
    print(f"Affected (dataset, model) combos: {len(affected_combos)}")
    print(f"\nNow re-run:  uv run python experiments/exp_multi_llm.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
