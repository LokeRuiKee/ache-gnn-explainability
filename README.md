# Graph neural network prediction and explanation for acetylcholinesterase inhibitors

Code, split indices, trained checkpoints, predictions, and results for Graph Convolutional Classification of Acetylcholinesterase Inhibitors with GNNExplainer  (submitted to Baghdad Science Journal, 2026).

Every number reported in the paper is produced by the code here from the source
dataset (obtained separately; see [Dataset](#dataset-not-included--obtain-from-the-source)),
and every result file records the command that regenerates it.

## What the study found

A graph convolutional network classifies acetylcholinesterase inhibitors from
molecular structure. GNNExplainer is applied to the same frozen checkpoint whose
performance is reported, and its attributions are then tested rather than
displayed.

**Predictive performance** — one checkpoint per run, selected at the epoch of
maximum validation F1; all metrics computed from that checkpoint's saved
predictions. Bootstrap 95% confidence intervals over the evaluated molecules
(10,000 resamples, seed 42):

| Protocol | Held-out test | Accuracy | F1 | ROC-AUC |
|---|---:|---|---|---|
| Random split | n = 407 | 0.8821 [0.8501, 0.9140] | 0.8710 [0.8333, 0.9059] | 0.9448 [0.9217, 0.9654] |
| Scaffold-disjoint split | n = 407 | 0.7936 [0.7543, 0.8329] | 0.8166 [0.7759, 0.8531] | 0.8727 [0.8372, 0.9056] |

The two protocols answer different questions and are never averaged. The random
split shares Bemis–Murcko scaffolds between partitions, so it measures
interpolation within known chemical series. The scaffold-disjoint split is the
generalisation estimate, and it is the relevant figure for prospective
screening.

**Against a classical baseline.** An ECFP4 support vector classifier was trained
and scored on identical folds, making the comparison paired at the level of the
individual molecule. Resampling molecules within fold across all 3,668 holdout
molecules gives, for the graph model minus the baseline: F1 −0.0007 (95% CI
−0.0131 to 0.0117), ROC-AUC −0.0010 (−0.0087 to 0.0065). **The graph
representation conferred no measurable accuracy advantage on this dataset**, and
the intervals are narrow enough to exclude any F1 advantage above roughly one
percentage point. The differences that the comparison does resolve are a
precision/recall trade-off: the graph model over-predicts activity.

**Explanation faithfulness is undetermined.** Masking the highest-attributed
atoms degraded the prediction *less* than random masking for three of five
representative molecules; deleting the highest-ranked bonds degraded it *more*
than random deletion for three of five. The two probes agree for only two of the
five, and the top-attributed atom set varies across explainer seeds (Jaccard
0.230–0.506). We therefore report fidelity, stability, and sparsity as results
in their own right and make no claim that the highlighted regions are
chemically meaningful determinants of activity.

## Dataset (not included — obtain from the source)

The models were trained on the human acetylcholinesterase (AChE) classification
dataset curated by:

> Vignaux PA, Lane TR, Urbina F, Gerlach J, Puhl AC, Snyder SH, Ekins S.
> Validation of acetylcholinesterase inhibition machine learning models for
> multiple species. *Chem Res Toxicol.* 2023;36(2):188–201.
> [doi:10.1021/acs.chemrestox.2c00283](https://doi.org/10.1021/acs.chemrestox.2c00283)
> · [PMC9945174](https://pmc.ncbi.nlm.nih.gov/articles/PMC9945174/)

That dataset is distributed with the article under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), which
does not permit redistribution of modified versions, so the two-column table
used here is **not included**. Curation is described in that paper, which
binarised activity at 1 µM (pIC50 = 6); that is the threshold used here. Its
Table 2 reports 1,813 active of 4,075 compounds for the human training set,
matching the table used here.

**To rebuild `cleanData.xlsx`:**

1. Download the Supplementary Material file tx2c00283_si_002.zip  from the
   article and open sheet Human_dataset_1micromolar.xlsx.
2. Keep only SMILES column and single-class-label`.
   **Do not sort or filter rows**: the split manifests identify molecules by
   row position.
3. Save as `cleanData.xlsx`, sheet `Sheet1`, in the repository root.
4. Check: 4,075 rows; 1,813 labelled 1 and 2,262 labelled 0.
5. The pipeline refuses a file whose SHA-256 differs from `expected_sha256` in
   `configs/model4_pyg.yaml`. The authors' copy hashes to
   `168ab208ca7625994e1f92ef793eb8fd78f8fdd1ee12947478b3c6bf9a97cf28`; a rebuilt
   file has identical rows but a different hash, because Excel files embed
   metadata. After step 4, set `expected_sha256` to your file's hash
   (`sha256sum cleanData.xlsx`, or `Get-FileHash cleanData.xlsx` in Windows
   PowerShell).

The split manifests in `results/protocol/` contain row indices, partitions and
labels only. Per-compound potency values are not part of the two-column table,
so threshold-borderline analysis was not performed; those values are available
from the source publication.

## Layout

```
cleanData.xlsx        not included — rebuild from the source (see Dataset)
configs/              model configuration; quoted literature values, flagged unverified
src/                  featurisation, graph construction, model, training, metrics,
                      baselines, explanations, splits, leakage audit, bootstrap
scripts/              one command per reported artifact
results/              split indices, checkpoints, predictions, metrics, tables, figures
tests/                unit tests for everything above
docs/                 reproduction steps and model specification
environment/          pinned dependency sets and the environments actually used
FYP2/                 historical experiment notebooks kept as provenance evidence
```

## Reproducing

First obtain the dataset and update `expected_sha256` as described under
[Dataset](#dataset-not-included--obtain-from-the-source). Then:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r environment/requirements-model4.txt
```

Verified on Python 3.10.8, PyTorch 2.1.2+cpu, PyTorch Geometric 2.8.0.post1,
RDKit 2026.03.5, scikit-learn 1.7.2, CPU only. Full version table in
`environment/RUNTIME.md`.

```bash
python -m pytest -q
python scripts/train_model4.py --protocol reproduction
python scripts/train_model4.py --protocol generalization
python scripts/bootstrap_intervals.py
python scripts/run_gnnexplainer.py
python scripts/generate_publication_figures.py
```

`docs/reproducibility.md` gives the complete sequence, step by step, with the
determinism guarantees for each. Split assignments are read from committed
manifests rather than recomputed, so a rerun cannot silently repartition the
data.

## Verifying without rerunning anything

`results/artifact_inventory.json` records every result file with its size,
SHA-256, and the command that produces it, and refuses to leave anything
unattributed:

```bash
python scripts/build_artifact_inventory.py
```

It reads the file list from `git ls-files`, so run it on a staged or committed
tree. The test suite asserts that every recorded hash still matches its file.

## Corrections to earlier work

Making this work reproducible exposed three defects in the earlier analysis,
which are corrected here rather than silently replaced: the previously reported
figures were validation-set values at the selected epoch, and the test partition
had never been scored; the featuriser reordered atoms by degree, so earlier
explanation indices were not RDKit atom indices; and the explainer was
configured for binary classification against a two-logit head. The reason a
number changed is part of the result.

## Historical notebooks

`FYP2/` contains the seven Model 4 candidate notebooks and, at the repository
root, the cross-validation demo notebook. They are provenance evidence for the
recovered specification, not part of the reproduction path.

`results/provenance/model_provenance.json` was generated from the complete set
of 25 historical notebooks, of which only those relevant to Model 4 are included
here; re-running `scripts/recover_model4_spec.py` on this repository therefore
reports only the included subset.

## Citing

Please cite the paper (full reference to follow on publication) and this
repository.

## Licence

Code: MIT Licence (`LICENSE`). Data: not included. The source dataset belongs to
its authors and is distributed by them under CC BY-NC-ND 4.0; obtain it from the
article above and follow its terms. Result files are provided to verify the
paper; the labels they contain derive from the source dataset and remain under
its terms.
