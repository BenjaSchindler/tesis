# Production evaluation feasibility harness

This module executes the bounded public-data feasibility stage of
`output/research/production_eval/experiment_plan.json`. The design document is
preserved as originally frozen. Execution authorization and measured results
are recorded separately under `output/research/production_eval/execution/`.

The harness does not yet implement the complete longitudinal experiment. It
creates no verified human gold labels and does not treat source assistant
responses as correct answers.

## Components

- `resource_probe.py`: reports local CUDA availability and configured Gemini
  model access without printing credentials.
- `data_access_probe.py`: the same 300 predetermined stratified source rows,
  downloaded through revision-checked row requests or pinned Parquet ranges.
  The working acquisition route uses a bounded projected transfer, persistent
  footer cache and row checkpoints; row indices are unchanged across retries.
- `metadata_scan.py`: source-row temporal census using only role, timestamp,
  language and model columns. It preserves partial status on errors and resumes
  with cumulative transfer accounting. Signed CDN redirects are used only in
  memory to avoid repeatedly calling the Hub resolver.
- `normalize_sample.py`: fixed first-user request, source provenance, exact
  hashes, timestamp proxy, duplicate linkage and explicit technical exclusions.
- `api_audit.py`: Gemini input-only feasibility predictions with a frozen rubric,
  durable spending reservations, raw response hashes and resumable caching.
- `audit_records.py`: rejects mixed or stale annotation configurations before
  analysis. Automated annotations remain unvalidated machine predictions.
- `selection_methods.py`: six selectors, matched counts, actual sampling
  probabilities for randomized policies and separate estimator weights.
- `selector_rehearsal.py`: real-input GPU embeddings and selector execution with
  MPNet and MiniLM. Its coverage/length metrics are engineering diagnostics.
- `summarize_execution.py`: verifies completed source provenance, summarizes
  observed results and prepares blank, blind double-review sheets. Existing
  human review files are preserved when regenerating the summary.

## Reproduce local validation

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s experiments/production_eval -p 'test_*.py'
```

## Resume the current execution

Network and GPU access require the execution environment to expose the host
network and devices. The default Codex sandbox in this session did neither.
The commands below assume that access is already available.

Acquisition uses the system Conda Python because it has PyArrow. The exact
projection was preflighted before content transfer; 1.4 GB covers the selected
row-group chunks for the unchanged 300-row sample, not full dataset files.
Existing completed acquisition files are protected from accidental replacement.

```bash
python3 experiments/production_eval/data_access_probe.py --method parquet --max-transfer-bytes 1400000000
python3 experiments/production_eval/metadata_scan.py --resume
.venv/bin/python experiments/production_eval/normalize_sample.py
.venv/bin/python experiments/production_eval/api_audit.py --input output/research/production_eval/execution/audit_requests.jsonl --output output/research/production_eval/execution/audit.jsonl --limit 300 --workers 4 --max-cost-usd 2 --dry-run
```

The real paid audit uses the same command without `--dry-run`. Its US$2 cap is
cumulative for the output ledger, including earlier preflight calls. Successful
full-request hashes are reused. Failed/uncertain calls retain conservative
charges; a provider usage bound breach stops later calls and resumes.

After the audit has reached a terminal state:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python experiments/production_eval/selector_rehearsal.py --device cuda:1 --cohort machine_eligible
.venv/bin/python experiments/production_eval/summarize_execution.py
```

The rehearsal uses the first tokens up to each encoder's actual sequence limit,
reports truncation, and saves exact selected IDs and weights. Its directed
sampling signal is the input-length 80th percentile, a disclosed implementation
exercise; it does not claim to measure model difficulty.

The GPU command removes `LD_LIBRARY_PATH` only in that child process. The host's
inherited path mixed the virtual environment's cuBLAS 12.8 with a system
cuBLASLt 12.9, causing even tiny batched matrix multiplications to fail. The
process-local wrapper was verified with full-size float32 MPNet inference; no
global environment settings or installed packages were changed. CPU results
are retained alongside a separate GPU verification run.

## Scientific continuation

The blank review sheets assess task validity. A separate set of fresh paired
model responses and independent output grades is needed to calibrate a quality
judge. Then construct the disjoint time windows, freeze systems/selectors and
measure paired quality deltas against later references. All 300 inspected
feasibility examples and their duplicate families remain development material.
No selector-quality ranking or 30/60-day predictive result can be inferred from
the rehearsal alone.
