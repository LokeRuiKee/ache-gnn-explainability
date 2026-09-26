"""Deterministic historical-random and prospective-scaffold partitions."""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.chemistry.identities import derive_identities


REQUIRED_COLUMNS = {"SMILES", "single-class-label"}
PARTITIONS = ("train", "validation", "test")


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    labels = set(frame["single-class-label"].dropna().astype(int))
    if labels != {0, 1} or frame["single-class-label"].isna().any():
        raise ValueError("expected complete binary labels 0/1")


def _base_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        identities = derive_identities(str(row["SMILES"]))
        rows.append(
            {
                "row_index": int(row_index),
                "smiles": str(row["SMILES"]),
                "canonical_smiles": identities["canonical_isomeric"],
                "label": int(row["single-class-label"]),
                "scaffold_id": identities["murcko_scaffold"],
            }
        )
    return pd.DataFrame(rows)


def validate_manifest(
    manifest: pd.DataFrame,
    expected_rows: int,
    group_column: str | None = None,
) -> dict:
    """Reject incomplete, repeated, invalid, or group-leaking manifests."""
    required = {"row_index", "partition"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    observed = manifest["row_index"].astype(int).tolist()
    if len(observed) != expected_rows or sorted(observed) != list(range(expected_rows)):
        raise ValueError("every source row must occur exactly once")
    invalid_partitions = set(manifest["partition"]).difference(PARTITIONS)
    if invalid_partitions:
        raise ValueError(f"invalid partitions: {sorted(invalid_partitions)}")
    if group_column:
        if group_column not in manifest:
            raise ValueError(f"manifest missing group column: {group_column}")
        crossing = manifest.groupby(group_column)["partition"].nunique()
        if (crossing > 1).any():
            raise ValueError(f"{group_column} crosses partitions")
    return {
        "rows": expected_rows,
        "partition_counts": {
            name: int((manifest["partition"] == name).sum()) for name in PARTITIONS
        },
    }


def historical_random_split(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Reproduce the notebook's two-stage stratified 80/10/10 split."""
    _validate_frame(frame)
    base = _base_manifest(frame)
    all_indices = base["row_index"].to_numpy()
    labels = base["label"].to_numpy()
    train_indices, temporary_indices = train_test_split(
        all_indices,
        train_size=0.8,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    temporary_labels = base.set_index("row_index").loc[temporary_indices, "label"].to_numpy()
    test_indices, validation_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=seed,
        stratify=temporary_labels,
    )
    assignments = {
        **{int(index): "train" for index in train_indices},
        **{int(index): "test" for index in test_indices},
        **{int(index): "validation" for index in validation_indices},
    }
    base["partition"] = base["row_index"].map(assignments)
    base = base.sort_values("row_index", kind="stable").reset_index(drop=True)
    validate_manifest(base, len(frame))
    return base[
        ["row_index", "smiles", "canonical_smiles", "label", "partition", "scaffold_id"]
    ]


def _tie_break(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()


def scaffold_split(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign whole Bemis-Murcko groups toward deterministic 80/10/10 targets."""
    _validate_frame(frame)
    base = _base_manifest(frame)
    base["assignment_group"] = base.apply(
        lambda row: (
            f"scaffold:{row['scaffold_id']}"
            if row["scaffold_id"]
            else f"acyclic:{row['canonical_smiles']}:{int(row['row_index'])}"
        ),
        axis=1,
    )
    groups = [
        (str(group), list(rows["row_index"].astype(int)))
        for group, rows in base.groupby("assignment_group", sort=False)
    ]
    groups.sort(key=lambda item: (-len(item[1]), _tie_break(seed, item[0]), item[0]))

    total = len(base)
    targets = {"train": total * 0.8, "validation": total * 0.1, "test": total * 0.1}
    counts = {name: 0 for name in PARTITIONS}
    assignments: dict[int, str] = {}
    priority = {name: index for index, name in enumerate(PARTITIONS)}
    for group, row_indices in groups:
        partition = max(
            PARTITIONS,
            key=lambda name: (targets[name] - counts[name], -priority[name]),
        )
        for row_index in row_indices:
            assignments[row_index] = partition
        counts[partition] += len(row_indices)

    base["partition"] = base["row_index"].map(assignments)
    base = base.sort_values("row_index", kind="stable").reset_index(drop=True)
    validate_manifest(base, len(frame), group_column="assignment_group")
    return base[
        [
            "row_index",
            "smiles",
            "canonical_smiles",
            "label",
            "partition",
            "scaffold_id",
            "assignment_group",
        ]
    ]
