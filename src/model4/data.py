"""Load the workbook and an immutable split manifest into PyG graph partitions.

Partitions are always read from a committed manifest CSV, never recomputed at
training time. That is what keeps the reproduction split and the scaffold split
immutable and comparable across runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.model4.graphs import GraphBuildReport, build_graphs

PARTITIONS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_workbook(path: Path, sheet: str, expected_sha256: str | None = None) -> pd.DataFrame:
    """Read the modelling table, refusing to proceed on a hash mismatch."""
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"{path} SHA-256 is {actual}, expected {expected_sha256}. "
            "Refusing to train against an unverified dataset."
        )
    frame = pd.read_excel(path, sheet_name=sheet)
    frame = frame.reset_index(drop=True)
    frame.index.name = "row_index"
    return frame


def load_manifest(path: Path, expected_rows: int) -> pd.DataFrame:
    """Read a split manifest and verify it covers every row exactly once."""
    manifest = pd.read_csv(path)
    missing = {"row_index", "partition", "label"} - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows = sorted(manifest["row_index"].tolist())
    if rows != list(range(expected_rows)):
        raise ValueError(
            f"{path} must cover rows 0..{expected_rows - 1} exactly once; "
            f"found {len(rows)} entries with {len(set(rows))} unique values."
        )

    unexpected = set(manifest["partition"]) - set(PARTITIONS)
    if unexpected:
        raise ValueError(f"{path} contains unknown partitions: {sorted(unexpected)}")
    return manifest


def build_partitions(
    workbook: pd.DataFrame,
    manifest: pd.DataFrame,
    smiles_column: str = "SMILES",
) -> tuple[dict[str, list], dict[str, GraphBuildReport]]:
    """Return ``{partition: [Data, ...]}`` plus a build report per partition."""
    graphs: dict[str, list] = {}
    reports: dict[str, GraphBuildReport] = {}

    for partition in PARTITIONS:
        rows = manifest[manifest["partition"] == partition].sort_values("row_index")
        row_indices = rows["row_index"].tolist()
        smiles = workbook.loc[row_indices, smiles_column].tolist()
        labels = rows["label"].tolist()
        partition_graphs, report = build_graphs(smiles, labels, row_indices)
        graphs[partition] = partition_graphs
        reports[partition] = report

    return graphs, reports
