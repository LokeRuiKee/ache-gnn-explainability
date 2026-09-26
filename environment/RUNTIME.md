# Verified runtime environments

Reproducibility requirement P0.36 asks for the exact software environment behind
every reported result. This file records the environments that have actually
executed this repository's code, and the cross-version checks between them.

## Environment A — dataset and protocol audit (original run)

Recorded in `results/leakage/leakage_summary.json` and
`results/dataset_audit/dataset_audit.json`.

| Component | Version |
|---|---|
| Python | 3.12.13 |
| pandas | 2.3.3 |
| RDKit | 2026.03.5 |
| scikit-learn | 1.9.0 |
| openpyxl | 3.1.5 |
| pytest | 8.4.2 |

## Environment B — Model 4 reconstruction (current)

| Component | Version |
|---|---|
| Python | 3.10.8 |
| PyTorch | 2.1.2+cpu |
| PyTorch Geometric | 2.8.0.post1 |
| pandas | 2.3.3 |
| RDKit | 2026.03.5 |
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| matplotlib | 3.10.9 |
| openpyxl | 3.1.5 |
| pytest | 9.1.1 |
| Device | CPU |

Environment A was not available on the machine used for Environment B, so the
audit artifacts were regenerated to check whether the difference matters.

## Cross-version reproduction check

Both protocol CLIs were re-run under Environment B with the committed arguments:

```
python scripts/extract_split_protocol.py --repo . --output results/protocol/historical_protocol.json
python scripts/audit_split_protocol.py --input cleanData.xlsx --sheet Sheet1 \
  --expected-sha256 168ab208ca7625994e1f92ef793eb8fd78f8fdd1ee12947478b3c6bf9a97cf28 \
  --random-manifest results/protocol/random_split_manifest.csv \
  --scaffold-manifest results/protocol/scaffold_split_manifest.csv \
  --summary results/leakage/leakage_summary.json \
  --groups results/leakage/leakage_groups.csv
```

**Result:** `results/protocol/historical_protocol.json`,
`results/protocol/random_split_manifest.csv`,
`results/protocol/scaffold_split_manifest.csv`, and
`results/leakage/leakage_groups.csv` regenerated **byte-identically**. The only
difference anywhere was the recorded `software.scikit_learn` string inside
`results/leakage/leakage_summary.json` (`1.9.0` under A, `1.7.2` under B); every
partition assignment, group count, and leakage figure was unchanged.

The committed artifact retains `1.9.0` because that string correctly describes
the environment that produced the committed file. This section is the record
that its content is not sensitive to the Python 3.12→3.10 or
scikit-learn 1.9.0→1.7.2 difference.

## Note on running the tests

`pytest` needs a writable temporary directory. On the current machine the
default `%TEMP%\pytest-of-*` path is not writable, so the suite is run with an
explicit base:

```
python -m pytest -q --basetemp=<writable-directory>
```

Plain `python -m pytest -q` is sufficient wherever the default temp path works.
