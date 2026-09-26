"""Within-study baselines evaluated on identical folds.

The manuscript currently compares against SVC, AttentiveFP, and CNN numbers
taken from other publications. Those are indirect literature comparisons and
cannot support a statistical test (see ``RESEARCH_CONTEXT.md`` section 6).

This module provides a baseline that is rerun *inside* this study on the exact
same folds as the graph model, which is what makes a paired test legitimate.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048


def morgan_fingerprints(smiles_list: list[str]) -> np.ndarray:
    """ECFP4-equivalent Morgan fingerprints as a dense binary matrix."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")
        rows.append(np.asarray(generator.GetFingerprint(mol), dtype=np.float32))
    return np.stack(rows)


def build_svc(seed: int = 42) -> Pipeline:
    """RBF SVC with probability estimates, matching the literature baseline type."""
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            ("svc", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]
    )


def fit_predict_svc(
    train_smiles: list[str],
    train_labels: np.ndarray,
    eval_smiles: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the fold's training molecules and score its held-out molecules."""
    model = build_svc(seed)
    model.fit(morgan_fingerprints(train_smiles), train_labels)
    features = morgan_fingerprints(eval_smiles)
    return model.predict(features), model.predict_proba(features)[:, 1]
