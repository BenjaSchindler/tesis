#!/usr/bin/env python3
"""
Pipelined Experiment Runner for Phase E

Problem: Embeddings are the bottleneck (~60% of time), but VRAM is underutilized
during LLM generation (which uses OpenAI API, not GPU).

Solution: Pipeline experiments so that:
- While Experiment N is doing LLM generation → Experiment N+1 computes embeddings
- Embeddings run sequentially (to maximize batch efficiency)
- LLM generation can overlap with next experiment's embeddings

Pipeline stages:
  STAGE 1: Embeddings (GPU) - Sequential, one at a time
  STAGE 2: Clustering + Anchors + LLM Generation (CPU/API) - Can overlap
  STAGE 3: Quality Gate + Training + Eval (GPU) - Sequential

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Exp 1: [EMBED]──────>[CLUSTER+LLM]──────>[TRAIN+EVAL]         │
  │  Exp 2:           [EMBED]──────>[CLUSTER+LLM]──────>[TRAIN+EVAL]│
  │  Exp 3:                    [EMBED]──────>[CLUSTER+LLM]─────>... │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    python3 pipelined_runner.py --experiments experiments.yaml
    python3 pipelined_runner.py --quick-test  # Test with 2 experiments
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import tempfile

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))


@dataclass
class Experiment:
    """Single experiment configuration."""
    name: str
    seed: int
    extra_args: str = ""
    config_type: str = "default"

    # Runtime state
    status: str = "pending"  # pending, embedding, generation, training, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    embeddings_ready: threading.Event = field(default_factory=threading.Event)
    results: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0


class PipelinedRunner:
    """Orchestrates experiments with pipelined execution."""

    def __init__(
        self,
        base_dir: str,
        results_dir: str,
        data_path: str = "../MBTI_500.csv",
        cache_dir: str = "embeddings_cache",
        batch_size: int = 128,
        max_concurrent_llm: int = 2,  # Max concurrent LLM generation
    ):
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_concurrent_llm = max_concurrent_llm

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Queues for pipeline stages
        self.embedding_queue: queue.Queue = queue.Queue()
        self.generation_queue: queue.Queue = queue.Queue()
        self.training_queue: queue.Queue = queue.Queue()

        # Semaphores for resource control
        self.gpu_lock = threading.Lock()  # Only one GPU-intensive task at a time
        self.llm_semaphore = threading.Semaphore(max_concurrent_llm)

        # Status tracking
        self.experiments: List[Experiment] = []
        self.completed = 0
        self.failed = 0
        self.running = True

        # Log file
        self.log_file = self.results_dir / "pipeline.log"

    def log(self, msg: str):
        """Thread-safe logging."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def add_experiment(self, name: str, seed: int, extra_args: str = "", config_type: str = "default"):
        """Add experiment to queue."""
        exp = Experiment(name=name, seed=seed, extra_args=extra_args, config_type=config_type)
        self.experiments.append(exp)
        self.embedding_queue.put(exp)

    def _run_embedding_stage(self, exp: Experiment) -> bool:
        """Stage 1: Compute embeddings (GPU intensive)."""
        exp.status = "embedding"
        exp.start_time = time.time()
        self.log(f"[{exp.name} s{exp.seed}] STAGE 1: Computing embeddings...")

        # Check cache first
        cache_key = f"{self.data_path}_{exp.seed}"
        cache_file = Path(self.cache_dir) / f"embeddings_seed{exp.seed}.npz"

        if cache_file.exists():
            self.log(f"[{exp.name} s{exp.seed}] Using cached embeddings")
            exp.embeddings_ready.set()
            return True

        # Compute embeddings with GPU lock
        with self.gpu_lock:
            try:
                # Run embedding computation
                cmd = [
                    "python3", "-c", f"""
import sys
sys.path.insert(0, 'core')
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from embedding_cache import EmbeddingCache
import re

DATA_PATH = "{self.data_path}"
SEED = {exp.seed}
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = {self.batch_size}

cache = EmbeddingCache("{self.cache_dir}")
df = pd.read_csv(DATA_PATH).rename(columns={{"posts": "text", "type": "label"}})

# Check cache
if all(cache.load(DATA_PATH, SEED, split, MODEL_NAME, 0.2, 0.15) is not None
       for split in ["train", "val", "test"]):
    print("CACHED")
    exit(0)

# Compute embeddings
embedder = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True)

train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=SEED, stratify=train_val_df["label"])

def normalize(text):
    text = re.sub(r"https?://\\S+", "", text)
    text = re.sub(r"\\|\\|\\|", " ", text)
    return text.strip()

for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if cache.load(DATA_PATH, SEED, name, MODEL_NAME, 0.2, 0.15) is None:
        print(f"Computing {{name}} embeddings...")
        texts = [normalize(t) for t in split_df["text"]]
        emb = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
        cache.save(emb, DATA_PATH, SEED, name, MODEL_NAME, 0.2, 0.15)

print("DONE")
"""
                ]

                result = subprocess.run(
                    cmd,
                    cwd=str(self.base_dir),
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour max
                )

                if result.returncode != 0:
                    self.log(f"[{exp.name} s{exp.seed}] Embedding failed: {result.stderr[:200]}")
                    return False

                self.log(f"[{exp.name} s{exp.seed}] Embeddings complete")
                exp.embeddings_ready.set()
                return True

            except Exception as e:
                self.log(f"[{exp.name} s{exp.seed}] Embedding error: {e}")
                return False

    def _run_generation_stage(self, exp: Experiment) -> bool:
        """Stage 2: Clustering + Anchor Selection + LLM Generation (uses API, not GPU)."""
        exp.status = "generation"
        self.log(f"[{exp.name} s{exp.seed}] STAGE 2: Generation (clustering + LLM)...")

        # Wait for embeddings
        if not exp.embeddings_ready.wait(timeout=7200):
            self.log(f"[{exp.name} s{exp.seed}] Timeout waiting for embeddings")
            return False

        # Run generation with LLM semaphore (allows concurrent API calls)
        with self.llm_semaphore:
            out_prefix = self.results_dir / f"{exp.name}_s{exp.seed}"

            cmd = f"""python3 -u core/runner_phase2.py \
                --data-path {self.data_path} --test-size 0.2 --random-seed {exp.seed} \
                --embedding-model sentence-transformers/all-mpnet-base-v2 --device cuda --embedding-batch-size {self.batch_size} \
                --llm-model gpt-4o-mini --max-clusters 3 --prompts-per-cluster 3 --prompt-mode mix \
                --use-ensemble-selection --use-val-gating --val-size 0.15 --val-tolerance 0.02 \
                --enable-anchor-gate --enable-anchor-selection \
                --anchor-selection-ratio 0.8 --anchor-outlier-threshold 1.5 \
                --use-class-description --use-f1-budget-scaling --f1-budget-thresholds 0.45 0.20 \
                --f1-budget-multipliers 0.0 0.5 1.0 \
                --min-classifier-confidence 0.10 --contamination-threshold 0.95 \
                --synthetic-weight 0.5 --synthetic-weight-mode flat \
                --use-hard-anchors --deterministic-quality-gate \
                --synthetic-output {out_prefix}_synth.csv \
                --augmented-train-output {out_prefix}_aug.csv \
                --metrics-output {out_prefix}_metrics.json \
                {exp.extra_args}"""

            log_file = f"{out_prefix}.log"

            try:
                with open(log_file, "w") as f:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=str(self.base_dir),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=7200  # 2 hours max
                    )

                if result.returncode != 0:
                    self.log(f"[{exp.name} s{exp.seed}] Generation failed (see {log_file})")
                    return False

                # Parse results
                metrics_file = f"{out_prefix}_metrics.json"
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        exp.results = json.load(f)

                    baseline = exp.results.get("baseline", {}).get("macro_f1", 0)
                    augmented = exp.results.get("augmented", {}).get("macro_f1", 0)
                    delta_pct = ((augmented - baseline) / baseline * 100) if baseline > 0 else 0
                    synth = exp.results.get("synthetic_data", {}).get("accepted_count", 0)

                    exp.results["delta_pct"] = delta_pct
                    self.log(f"[{exp.name} s{exp.seed}] COMPLETE: B={baseline:.4f} A={augmented:.4f} Δ={delta_pct:+.2f}% ({synth} synth)")
                    return True
                else:
                    self.log(f"[{exp.name} s{exp.seed}] No metrics file generated")
                    return False

            except subprocess.TimeoutExpired:
                self.log(f"[{exp.name} s{exp.seed}] Timeout in generation")
                return False
            except Exception as e:
                self.log(f"[{exp.name} s{exp.seed}] Generation error: {e}")
                return False

    def _embedding_worker(self):
        """Worker thread for embedding stage."""
        while self.running:
            try:
                exp = self.embedding_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self._run_embedding_stage(exp):
                self.generation_queue.put(exp)
            else:
                exp.status = "failed"
                exp.end_time = time.time()
                self.failed += 1

            self.embedding_queue.task_done()

    def _generation_worker(self):
        """Worker thread for generation stage."""
        while self.running:
            try:
                exp = self.generation_queue.get(timeout=1)
            except queue.Empty:
                continue

            if self._run_generation_stage(exp):
                exp.status = "completed"
                self.completed += 1
            else:
                exp.status = "failed"
                self.failed += 1

            exp.end_time = time.time()
            self.generation_queue.task_done()

    def _status_reporter(self):
        """Periodically report status."""
        while self.running:
            time.sleep(30)

            pending = sum(1 for e in self.experiments if e.status == "pending")
            embedding = sum(1 for e in self.experiments if e.status == "embedding")
            generation = sum(1 for e in self.experiments if e.status == "generation")

            if pending + embedding + generation > 0:
                self.log(f"STATUS: {self.completed} done, {self.failed} failed, "
                        f"{embedding} embedding, {generation} generating, {pending} pending")

    def run(self):
        """Run all experiments with pipelining."""
        self.log("=" * 60)
        self.log("  PIPELINED EXPERIMENT RUNNER")
        self.log(f"  Total experiments: {len(self.experiments)}")
        self.log(f"  Max concurrent LLM: {self.max_concurrent_llm}")
        self.log("=" * 60)

        start_time = time.time()

        # Start worker threads
        # 1 embedding worker (sequential to maximize batch efficiency)
        embedding_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        embedding_thread.start()

        # Multiple generation workers (can run in parallel)
        generation_threads = []
        for i in range(self.max_concurrent_llm):
            t = threading.Thread(target=self._generation_worker, daemon=True)
            t.start()
            generation_threads.append(t)

        # Status reporter
        status_thread = threading.Thread(target=self._status_reporter, daemon=True)
        status_thread.start()

        # Wait for all work to complete
        self.embedding_queue.join()
        self.generation_queue.join()

        self.running = False

        # Summary
        total_time = time.time() - start_time
        self.log("=" * 60)
        self.log("  PIPELINE COMPLETE")
        self.log(f"  Completed: {self.completed}/{len(self.experiments)}")
        self.log(f"  Failed: {self.failed}")
        self.log(f"  Total time: {total_time/60:.1f} minutes")
        self.log("=" * 60)

        # Write summary
        self._write_summary()

        return self.completed, self.failed

    def _write_summary(self):
        """Write experiment summary to CSV and JSON."""
        summary_csv = self.results_dir / "summary.csv"
        summary_json = self.results_dir / "summary.json"

        with open(summary_csv, "w") as f:
            f.write("experiment,seed,config_type,baseline_f1,augmented_f1,delta_pct,synthetics,duration_s,status\n")
            for exp in self.experiments:
                baseline = exp.results.get("baseline", {}).get("macro_f1", "N/A")
                augmented = exp.results.get("augmented", {}).get("macro_f1", "N/A")
                delta = exp.results.get("delta_pct", "N/A")
                synth = exp.results.get("synthetic_data", {}).get("accepted_count", "N/A")

                if isinstance(baseline, float):
                    baseline = f"{baseline:.4f}"
                if isinstance(augmented, float):
                    augmented = f"{augmented:.4f}"
                if isinstance(delta, float):
                    delta = f"{delta:.2f}"

                f.write(f"{exp.name},{exp.seed},{exp.config_type},{baseline},{augmented},{delta},{synth},{exp.duration:.0f},{exp.status}\n")

        self.log(f"Summary saved to: {summary_csv}")

        # JSON summary
        results = {
            "total": len(self.experiments),
            "completed": self.completed,
            "failed": self.failed,
            "experiments": [
                {
                    "name": exp.name,
                    "seed": exp.seed,
                    "status": exp.status,
                    "duration": exp.duration,
                    "results": exp.results
                }
                for exp in self.experiments
            ]
        }

        with open(summary_json, "w") as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Pipelined Experiment Runner")
    parser.add_argument("--experiments", type=str, help="YAML file with experiment definitions")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with 2 experiments")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--max-concurrent-llm", type=int, default=2, help="Max concurrent LLM generations")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Setup
    base_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or f"results/pipelined_{timestamp}"

    runner = PipelinedRunner(
        base_dir=str(base_dir),
        results_dir=results_dir,
        max_concurrent_llm=args.max_concurrent_llm
    )

    if args.quick_test:
        # Quick test with 2 configs
        print("Running quick test with 2 experiments...")
        runner.add_experiment("test_baseline", 42, "", "test")
        runner.add_experiment("test_sim095", 42, "--similarity-threshold 0.95", "test")
    elif args.experiments:
        # Load from YAML
        import yaml
        with open(args.experiments) as f:
            config = yaml.safe_load(f)

        for exp in config.get("experiments", []):
            runner.add_experiment(
                exp["name"],
                exp["seed"],
                exp.get("extra_args", ""),
                exp.get("config_type", "default")
            )
    else:
        # Default: remaining Phase E experiments
        print("Loading default Phase E experiments...")

        # Standard configs for both seeds
        configs = [
            ("cfg07_anchor_035", "--anchor-quality-threshold 0.35 --similarity-threshold 0.90", "anchor_sweep"),
            ("cfg08_anchor_040", "--anchor-quality-threshold 0.40 --similarity-threshold 0.90", "anchor_sweep"),
            ("cfg09_sim_085", "--anchor-quality-threshold 0.30 --similarity-threshold 0.85", "sim_sweep"),
            ("cfg10_sim_088", "--anchor-quality-threshold 0.30 --similarity-threshold 0.88", "sim_sweep"),
            ("cfg11_sim_092", "--anchor-quality-threshold 0.30 --similarity-threshold 0.92", "sim_sweep"),
            ("cfg12_sim_095", "--anchor-quality-threshold 0.30 --similarity-threshold 0.95", "sim_sweep"),
            ("cfg13_combined", "--anchor-quality-threshold 0.25 --similarity-threshold 0.85", "combined"),
        ]

        # Improvement experiments
        improvements = [
            ("imp_more_synth", "--anchor-quality-threshold 0.30 --similarity-threshold 0.90 --max-clusters 5 --prompts-per-cluster 9", "improvement"),
            ("imp_relaxed_gate", "--anchor-quality-threshold 0.15 --similarity-threshold 0.80", "improvement"),
        ]

        for seed in [42, 100]:
            for name, args_str, ctype in configs + improvements:
                runner.add_experiment(name, seed, args_str, ctype)

    # Run
    completed, failed = runner.run()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
