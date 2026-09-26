import pandas as pd
import pytest

from src.protocol.splits import historical_random_split, scaffold_split, validate_manifest


def binary_frame(size: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES": ["C" if index % 2 == 0 else "O" for index in range(size)],
            "single-class-label": [index % 2 for index in range(size)],
        }
    )


def scaffold_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES": [
                "Oc1ccccc1",
                "Cc1ccccc1",
                "C1CCCCC1",
                "OC1CCCCC1",
                "CC",
                "CCC",
            ],
            "single-class-label": [1, 0, 0, 1, 0, 1],
        }
    )


def test_historical_split_is_complete_disjoint_and_reproducible():
    first = historical_random_split(binary_frame(), seed=42)
    second = historical_random_split(binary_frame(), seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["partition"].value_counts().to_dict() == {
        "train": 80,
        "test": 10,
        "validation": 10,
    }
    assert sorted(first["row_index"]) == list(range(100))


def test_historical_split_preserves_binary_class_balance():
    manifest = historical_random_split(binary_frame(), seed=42)

    counts = manifest.groupby(["partition", "label"]).size().to_dict()
    assert counts == {
        ("test", 0): 5,
        ("test", 1): 5,
        ("train", 0): 40,
        ("train", 1): 40,
        ("validation", 0): 5,
        ("validation", 1): 5,
    }


def test_manifest_validator_rejects_repeated_rows():
    manifest = pd.DataFrame(
        {"row_index": [0, 0], "partition": ["train", "test"]}
    )

    with pytest.raises(ValueError, match="exactly once"):
        validate_manifest(manifest, expected_rows=2)


def test_manifest_validator_rejects_missing_rows():
    manifest = pd.DataFrame({"row_index": [0], "partition": ["train"]})

    with pytest.raises(ValueError, match="exactly once"):
        validate_manifest(manifest, expected_rows=2)


def test_scaffold_split_keeps_nonempty_groups_disjoint():
    manifest = scaffold_split(scaffold_test_frame(), seed=42)
    nonempty = manifest[manifest["scaffold_id"] != ""]

    assert nonempty.groupby("scaffold_id")["partition"].nunique().max() == 1
    assert sorted(manifest["row_index"]) == list(range(len(manifest)))


def test_empty_scaffolds_are_distinct_deterministic_groups():
    frame = pd.DataFrame(
        {"SMILES": ["CC", "CCC"], "single-class-label": [0, 1]}
    )

    first = scaffold_split(frame, seed=42)
    second = scaffold_split(frame, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["assignment_group"].nunique() == 2


def test_validator_rejects_group_crossing_partitions():
    manifest = pd.DataFrame(
        {
            "row_index": [0, 1],
            "partition": ["train", "test"],
            "assignment_group": ["scaffold:ring", "scaffold:ring"],
        }
    )

    with pytest.raises(ValueError, match="crosses partitions"):
        validate_manifest(manifest, expected_rows=2, group_column="assignment_group")
