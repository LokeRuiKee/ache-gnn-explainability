"""Quantify molecular-identity overlap within and across data partitions."""

from __future__ import annotations

import pandas as pd

from src.chemistry.identities import derive_identities


IDENTITY_LEVELS = (
    "canonical_isomeric",
    "stereo_insensitive",
    "largest_fragment",
    "murcko_scaffold",
)
GROUP_COLUMNS = [
    "identity_level",
    "identity",
    "group_scope",
    "conflicting_labels",
    "row_index",
    "partition",
    "label",
    "smiles",
]


def audit_leakage(manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return stable summary and row evidence for repeated molecular identities."""
    required = {"row_index", "smiles", "label", "partition"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    records = []
    for row in manifest.itertuples(index=False):
        identities = derive_identities(str(row.smiles))
        record = {
            "row_index": int(row.row_index),
            "smiles": str(row.smiles),
            "label": int(row.label),
            "partition": str(row.partition),
        }
        record.update({level: identities[level] for level in IDENTITY_LEVELS})
        records.append(record)
    identity_frame = pd.DataFrame(records)

    summary_levels: dict[str, dict] = {}
    group_rows: list[dict] = []
    for level in IDENTITY_LEVELS:
        empty_rows = int(identity_frame[level].eq("").sum())
        eligible = identity_frame[identity_frame[level].ne("")]
        duplicate_groups = []
        for identity, group in eligible.groupby(level, sort=True):
            if len(group) < 2:
                continue
            partition_count = int(group["partition"].nunique())
            conflicting = bool(group["label"].nunique() > 1)
            scope = "cross_partition" if partition_count > 1 else "within_partition"
            duplicate_groups.append((identity, group, scope, conflicting))
            for row in group.sort_values("row_index", kind="stable").itertuples(index=False):
                group_rows.append(
                    {
                        "identity_level": level,
                        "identity": identity,
                        "group_scope": scope,
                        "conflicting_labels": conflicting,
                        "row_index": int(row.row_index),
                        "partition": str(row.partition),
                        "label": int(row.label),
                        "smiles": str(row.smiles),
                    }
                )
        summary_levels[level] = {
            "empty_identity_rows": empty_rows,
            "unique_nonempty_identities": int(eligible[level].nunique()),
            "duplicate_groups": len(duplicate_groups),
            "within_partition_groups": sum(
                scope == "within_partition" for _, _, scope, _ in duplicate_groups
            ),
            "cross_partition_groups": sum(
                scope == "cross_partition" for _, _, scope, _ in duplicate_groups
            ),
            "affected_rows": sum(len(group) for _, group, _, _ in duplicate_groups),
            "conflicting_label_groups": sum(
                conflicting for _, _, _, conflicting in duplicate_groups
            ),
        }

    groups = pd.DataFrame(group_rows, columns=GROUP_COLUMNS)
    if not groups.empty:
        groups = groups.sort_values(
            ["identity_level", "identity", "row_index"], kind="stable"
        ).reset_index(drop=True)
    return {"schema_version": 1, "identity_levels": summary_levels}, groups
