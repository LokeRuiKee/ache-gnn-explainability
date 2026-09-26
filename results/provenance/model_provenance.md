# Historical Model Provenance Audit

Notebooks inspected: **25**.

## Ranked Model 4 Candidates

| Rank | Notebook | Cell | Accuracy | F1 | ROC-AUC | Selection | Score |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | `FYP2/FYP2exp13.5_best hyperparameter700epoch.ipynb` | 16 | 0.8676 | 0.8533 | 0.9281 | f1 | 0.007200 |
| 2 | `FYP2/FYP2exp13.1_best hyperparameter1000epoch.ipynb` | 16 | 0.8725 | 0.8556 | 0.9308 | f1 | 0.007700 |
| 3 | `FYP2/FYP2exp13.2_best hyperparameter200epoch.ipynb` | 16 | 0.8627 | 0.8503 | 0.9239 | f1 | 0.013100 |
| 4 | `FYP2/FYP2exp13.6.1_best hyperparameter600epoch.ipynb` | 16 | 0.8799 | 0.8635 | 0.9308 | f1 | 0.014200 |
| 5 | `FYP2/FYP2exp13.4_best hyperparameter400epoch.ipynb` | 16 | 0.8775 | 0.8641 | 0.9400 | f1 | 0.021600 |
| 6 | `FYP2/FYP2exp13.3_best hyperparameter500epoch.ipynb` | 16 | 0.8799 | 0.8679 | 0.9342 | f1 | 0.022000 |
| 7 | `FYP2/FYP2exp13.6.0_best hyperparameter700epoch.ipynb` | 16 | 0.8848 | 0.8683 | 0.9304 | f1 | 0.023500 |
| 8 | `FYP2/Copy of FYP2exp12_gridSearchGCN_wip.ipynb` | 18 | 0.8600 | 0.8480 | 0.9342 | f1 | 0.026240 |
| 9 | `FYP2/FYP2exp12_gridSearchGCN_wip.ipynb` | 18 | 0.8600 | 0.8480 | 0.9342 | f1 | 0.026240 |
| 10 | `FYP2 demo with best hyper & cross val.ipynb` | 18 | 0.8344 | 0.8193 | 0.9006 | f1, auc | 0.095700 |

The ranking is an evidence-discovery aid, not proof that the first row is the original Model 4 artifact.

## Cross-Validation Average Blocks

| Notebook | Cell | Accuracy | F1 | ROC-AUC | Minimum validation loss |
|---|---:|---:|---:|---:|---:|
| `FYP2 demo with best hyper & cross val.ipynb` | 18 | 0.822331 | 0.806676 | 0.898190 | 0.404623 |
| `FYP2/FYP2exp14_new hyperparameter.ipynb` | 18 | 0.823067 | 0.805421 | 0.895520 | 0.409745 |

These historical summaries independently maximize each metric across epochs and therefore require correction in the final evaluation pipeline.

## Framework and Error Evidence

### `FYP2 demo with best hyper & cross val.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 3: NameError: name 'tqdm' is not defined

### `FYP2/Copy of FYP2exp12_gridSearchGCN_wip.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 19: AttributeError: 'str' object has no attribute 'items'
- Saved error, cell 22: NameError: name 'model' is not defined

### `FYP2/FYP2 Molecular Validation.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 6: SyntaxError: invalid syntax (ipython-input-11-2438174211.py, line 2)

### `FYP2/FYP2 demo epoch with best hyper & cross val.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp10_1000epoch_wip.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp11_CNN_wip.ipynb`

- Framework evidence: not detected
- Saved error, cell 12: KeyboardInterrupt: 

### `FYP2/FYP2exp12_gridSearchGCN_wip.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 19: AttributeError: 'str' object has no attribute 'items'
- Saved error, cell 22: NameError: name 'model' is not defined

### `FYP2/FYP2exp13.0 EDA.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.1_best hyperparameter1000epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.2_best hyperparameter200epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.3_best hyperparameter500epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.4_best hyperparameter400epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.5_best hyperparameter700epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.6.0_best hyperparameter700epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp13.6.1_best hyperparameter600epoch.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp14_new hyperparameter.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp15_Sutthibutpong + explainable.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 7: AttributeError: 'GraphConvModel' object has no attribute '_input_dtypes'

### `FYP2/FYP2exp1_GNNExplainer.ipynb`

- Framework evidence: PyTorch Geometric
- Saved error, cell 19: NameError: name 'your_input_dimension' is not defined

### `FYP2/FYP2exp2, 3_XGNN.ipynb`

- Framework evidence: PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp4_GNNExplainer.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp5_GNNExplainer.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp6_GNNExplainer.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 45: NameError: name 'graphs' is not defined
- Saved error, cell 49: NameError: name 'test' is not defined

### `FYP2/FYP2exp7_GNNExplainer.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved error, cell 50: NameError: name 'graphs' is not defined
- Saved error, cell 54: NameError: name 'test' is not defined

### `FYP2/FYP2exp8_GNNExplainer_fixed explainer.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

### `FYP2/FYP2exp9_GNNExplainer_wip.ipynb`

- Framework evidence: DeepChem, PyTorch Geometric
- Saved errors: none detected

## Limitations

- No original trained Model 4 checkpoint is present.
- No immutable historical train/validation/test indices are present.
- Notebook outputs are historical evidence and may reflect non-linear execution order.
- DeepChemStyleGraphConv is PyTorch Geometric code, not the original DeepChem GraphConvModel implementation.
