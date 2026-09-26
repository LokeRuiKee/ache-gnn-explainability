import pandas as pd

from src.protocol.leakage import audit_leakage


def leaking_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "smiles": ["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O", "c1ccccc1"],
            "label": [0, 1, 1],
            "partition": ["train", "test", "validation"],
        }
    )


def test_audit_reports_cross_partition_and_conflicting_label_groups():
    summary, groups = audit_leakage(leaking_manifest())

    stereo = summary["identity_levels"]["stereo_insensitive"]
    assert stereo["duplicate_groups"] == 1
    assert stereo["cross_partition_groups"] == 1
    assert stereo["within_partition_groups"] == 0
    assert stereo["affected_rows"] == 2
    assert stereo["conflicting_label_groups"] == 1
    stereo_rows = groups[groups["identity_level"] == "stereo_insensitive"]
    assert set(stereo_rows["row_index"]) == {0, 1}


def test_empty_scaffolds_are_counted_but_not_collapsed_into_leakage_group():
    manifest = pd.DataFrame(
        {
            "row_index": [0, 1],
            "smiles": ["CC", "CCC"],
            "label": [0, 1],
            "partition": ["train", "test"],
        }
    )

    summary, groups = audit_leakage(manifest)

    scaffold = summary["identity_levels"]["murcko_scaffold"]
    assert scaffold["empty_identity_rows"] == 2
    assert scaffold["duplicate_groups"] == 0
    assert groups[groups["identity_level"] == "murcko_scaffold"].empty


def test_output_order_is_stable_by_identity_and_row():
    _, groups = audit_leakage(leaking_manifest().iloc[::-1].reset_index(drop=True))

    expected = groups.sort_values(
        ["identity_level", "identity", "row_index"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(groups, expected)
