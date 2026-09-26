"""Deterministic molecular dataset auditing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger


REQUIRED_COLUMNS = {"SMILES", "single-class-label"}


def _canonicalize(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def audit_dataframe(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit schema, molecular validity, labels, and canonical duplicates."""

    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

    labels = frame["single-class-label"]
    nonmissing_labels = labels.dropna()
    normalized_labels = set(nonmissing_labels.astype(int).tolist())
    if not normalized_labels.issubset({0, 1}) or any(
        float(value) != int(value) for value in nonmissing_labels
    ):
        raise ValueError("single-class-label must contain binary labels 0/1")

    class_counts = {
        str(label): int((nonmissing_labels.astype(int) == label).sum())
        for label in (0, 1)
    }
    label_total = sum(class_counts.values())
    class_percentages = {
        str(label): round(class_counts[str(label)] * 100.0 / label_total, 6)
        if label_total
        else 0.0
        for label in (0, 1)
    }

    records: list[dict[str, Any]] = []
    invalid_smiles = 0
    RDLogger.DisableLog("rdApp.error")
    try:
        for row_index, row in frame.iterrows():
            raw_smiles = row["SMILES"]
            if pd.isna(raw_smiles) or not str(raw_smiles).strip():
                continue
            smiles = str(raw_smiles).strip()
            canonical = _canonicalize(smiles)
            if canonical is None:
                invalid_smiles += 1
                continue
            label = row["single-class-label"]
            records.append(
                {
                    "row_index": int(row_index),
                    "smiles": smiles,
                    "canonical_smiles": canonical,
                    "label": None if pd.isna(label) else int(label),
                }
            )
    finally:
        RDLogger.EnableLog("rdApp.error")

    valid_frame = pd.DataFrame.from_records(
        records,
        columns=["row_index", "smiles", "canonical_smiles", "label"],
    )
    duplicate_frame = pd.DataFrame(
        columns=["duplicate_group", "row_index", "smiles", "canonical_smiles", "label", "conflicting_labels"]
    )
    duplicate_groups = 0
    conflicting_groups = 0
    if not valid_frame.empty:
        duplicate_mask = valid_frame.duplicated("canonical_smiles", keep=False)
        duplicates = valid_frame.loc[duplicate_mask].copy()
        if not duplicates.empty:
            canonical_values = sorted(duplicates["canonical_smiles"].unique())
            group_map = {
                canonical: group_number
                for group_number, canonical in enumerate(canonical_values, start=1)
            }
            duplicates["duplicate_group"] = duplicates["canonical_smiles"].map(group_map)
            label_diversity = duplicates.groupby("canonical_smiles")["label"].nunique(dropna=True)
            conflicting = set(label_diversity[label_diversity > 1].index)
            duplicates["conflicting_labels"] = duplicates["canonical_smiles"].isin(conflicting)
            duplicate_groups = len(canonical_values)
            conflicting_groups = len(conflicting)
            duplicate_frame = duplicates[
                ["duplicate_group", "row_index", "smiles", "canonical_smiles", "label", "conflicting_labels"]
            ].sort_values(["duplicate_group", "row_index"], ignore_index=True)

    valid_count = len(valid_frame)
    unique_count = int(valid_frame["canonical_smiles"].nunique()) if valid_count else 0
    duplicate_rows = len(duplicate_frame)
    summary = {
        "raw_rows": int(len(frame)),
        "missing_smiles": int(
            (
                frame["SMILES"].isna()
                | frame["SMILES"].fillna("").astype(str).str.strip().eq("")
            ).sum()
        ),
        "missing_labels": int(labels.isna().sum()),
        "invalid_smiles": int(invalid_smiles),
        "valid_molecules": int(valid_count),
        "analysis_ready_rows": int(sum(record["label"] is not None for record in records)),
        "class_counts_raw": class_counts,
        "class_percentages_raw": class_percentages,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_excess_rows": int(valid_count - unique_count),
        "duplicate_groups": int(duplicate_groups),
        "conflicting_label_groups": int(conflicting_groups),
        "unique_valid_canonical_smiles": unique_count,
    }
    return summary, duplicate_frame
