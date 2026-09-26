# Reproducibility Guide

Every reported number in this revision comes from a committed machine-readable
artifact produced by one of the commands below. Run them from the repository
root in the order given; each stage consumes the previous stage's outputs.

Software and hardware versions are recorded in `environment/RUNTIME.md`.

## 0. Environment

```bash
python -m venv .venv && . .venv/Scripts/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r environment/requirements-model4.txt
```

`torch` is installed first from the CPU index so that the generic wheel is not
pulled in. `environment/requirements-audit.txt` remains sufficient on its own
for the dataset and protocol stages (1–2) if model training is not needed.

## 1. Dataset audit

```bash
python scripts/audit_dataset.py --input cleanData.xlsx --sheet Sheet1 --output results/dataset_audit/dataset_audit.json --duplicates results/dataset_audit/duplicate_groups.csv
```

Produces the verified row, class, validity, and duplicate counts.

## 2. Split protocol and leakage audit

```bash
python scripts/extract_split_protocol.py --repo . --output results/protocol/historical_protocol.json

python scripts/audit_split_protocol.py \
  --input cleanData.xlsx --sheet Sheet1 \
  --expected-sha256 168ab208ca7625994e1f92ef793eb8fd78f8fdd1ee12947478b3c6bf9a97cf28 \
  --random-manifest results/protocol/random_split_manifest.csv \
  --scaffold-manifest results/protocol/scaffold_split_manifest.csv \
  --summary results/leakage/leakage_summary.json \
  --groups results/leakage/leakage_groups.csv
```

Produces the two immutable split manifests and the four-level molecular-overlap
audit. These manifests are inputs to every later stage and are never recomputed
during training.

## 3. Model 4 specification recovery

```bash
python scripts/recover_model4_config.py --repo . --output results/model4/model4_spec_evidence.json
```

Parses the candidate notebooks with `ast` without executing them and reports
where they agree and where they differ. The recovered specification is written
by hand into `configs/model4_pyg.yaml`, with each deliberate correction flagged.

## 4. Training and held-out evaluation

```bash
python scripts/train_model4.py --protocol reproduction --epochs 700
python scripts/train_model4.py --protocol generalization --epochs 700
```

Each run writes `checkpoint.pt`, `training_history.csv`, `predictions.csv`, and
`metrics.json` under `results/model4/<protocol>/`. Metrics come from the single
checkpoint selected by maximum validation F1 and are reported for both the
validation and the held-out test partition.

To recompute the derived files from an existing checkpoint without retraining:

```bash
python scripts/train_model4.py --protocol reproduction --predict-only
```

## 5. Variability

```bash
python scripts/cross_validate_model4.py --folds 5 --epochs 700
python scripts/seed_sensitivity.py --seeds 42 43 44 --epochs 700
```

Cross-validation varies the data partition and excludes the held-out test
partition entirely. Seed sensitivity varies only initialisation on a fixed
split. The two are reported separately and must not be pooled.

## 6. Statistical comparison

```bash
python scripts/bootstrap_intervals.py
python scripts/statistical_tests.py
```

`bootstrap_intervals.py` is the primary analysis. It produces confidence
intervals with numerical bounds for every metric on each held-out test partition
(10,000 resamples, percentile and BCa), and a paired cluster bootstrap for the
graph model against the ECFP4 support vector classifier that resamples molecules
within fold so both models always see identical rows. Outputs:
`results/model4/bootstrap_intervals.json`,
`results/tables/table_bootstrap_intervals.{csv,md}`,
`results/tables/table_paired_bootstrap.{csv,md}`. It reads only committed
prediction files, so it needs neither RDKit nor PyTorch and runs in about two and
a half minutes on a CPU.

No hypothesis test is applied to performance figures quoted from other
publications. Such a value is a point estimate carrying its own unreported
sampling uncertainty, obtained on a different split and preprocessing pipeline;
a one-sample test would treat it as a known population mean and attribute all
uncertainty to this study's folds (D-017). External figures are read from
`configs/literature_baselines.json`, where they are flagged as unverified, and
are only *located* relative to this study's intervals.

`statistical_tests.py` is retained as a sensitivity check. It compares
within-study models on identical folds only, and reports the power limitation
that five paired folds cannot yield a two-sided Wilcoxon p below 0.0625. Where it
and the bootstrap differ in what they resolve, the bootstrap is the finding.

## 7. Error analysis

```bash
python scripts/error_analysis.py --protocol reproduction
python scripts/error_analysis.py --protocol generalization
```

## 8. Explanations

```bash
python scripts/run_gnnexplainer.py --protocol reproduction
```

Explains the exact frozen checkpoint whose performance is reported, selects
representative molecules by a stated rule, and writes fidelity, stability, and
sparsity measurements alongside the atom tables.

## 9. Tables and figures

```bash
python scripts/generate_publication_figures.py
```

Regenerates every manuscript table and figure from the committed result files as
vector SVG plus 300 DPI PNG. Anything whose source file is missing is reported as
skipped rather than invented.

Output is byte-stable: the SVG hash salt is fixed and the wall-clock date is
omitted from SVG metadata, so re-running this command on unchanged inputs leaves
`git diff` empty. A non-empty diff after regeneration therefore means a result
actually changed, which makes it usable as a verification step.

## 10. Artifact inventory

```bash
python scripts/build_artifact_inventory.py
```

Records every committed result file with its size, SHA-256, and the command
that regenerates it, and refuses to leave anything unattributed. Together these
let a reader verify two things without rerunning any experiment: that each
reported number has a file behind it, and that the file is the one that was
recorded.

The inventory reads the file list from `git ls-files`, so **run it after
staging**. On an unstaged tree it reports zero artifacts rather than failing.

Re-run it whenever a result file changes. `tests/qa/test_submission_qa.py`
asserts that every recorded hash and size still matches its file, so a stale
inventory fails the suite rather than shipping.

## 11. Tests

```bash
python -m pytest -q
```

If the default temporary directory is not writable, pass an explicit base:

```bash
python -m pytest -q --basetemp=/path/to/writable/dir
```

## Determinism notes

- The workbook SHA-256 is verified before any split is loaded; a mismatch aborts
  the run.
- Split assignments come from committed manifests, never from a recomputed split.
- Seeds are set for Python, NumPy, and PyTorch, and deterministic algorithms are
  requested where available.
- Artifacts avoid wall-clock timestamps so regeneration is byte-comparable.
  Training runtime in minutes is the one recorded value that varies by machine.
- Training on CPU with these settings reproduces the same selected epoch and
  metrics across repeated runs on the same machine; this is asserted by a test
  for the training loop and verified for the full pipeline by rerunning with
  `--predict-only`.
