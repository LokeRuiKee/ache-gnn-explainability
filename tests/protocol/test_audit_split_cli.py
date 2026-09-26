import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_split_protocol.py"


def write_workbook(path: Path) -> str:
    frame = pd.DataFrame(
        {
            "SMILES": ["C" if index % 2 == 0 else "O" for index in range(40)],
            "single-class-label": [index % 2 for index in range(40)],
        }
    )
    frame.to_excel(path, index=False, sheet_name="Sheet1")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(workbook: Path, digest: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(workbook),
        "--sheet",
        "Sheet1",
        "--expected-sha256",
        digest,
        "--random-manifest",
        str(output_dir / "random.csv"),
        "--scaffold-manifest",
        str(output_dir / "scaffold.csv"),
        "--summary",
        str(output_dir / "summary.json"),
        "--groups",
        str(output_dir / "groups.csv"),
    ]


def test_audit_script_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/audit_split_protocol.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_wrong_source_hash_writes_no_artifacts(tmp_path):
    workbook = tmp_path / "data.xlsx"
    write_workbook(workbook)
    output_dir = tmp_path / "outputs"

    result = subprocess.run(
        command(workbook, "0" * 64, output_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert not output_dir.exists()


def test_successful_regeneration_is_byte_stable(tmp_path):
    workbook = tmp_path / "data.xlsx"
    digest = write_workbook(workbook)
    output_dir = tmp_path / "outputs"

    first = subprocess.run(
        command(workbook, digest, output_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    paths = sorted(output_dir.iterdir())
    first_bytes = {path.name: path.read_bytes() for path in paths}

    second = subprocess.run(
        command(workbook, digest, output_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert {path.name: path.read_bytes() for path in paths} == first_bytes

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_sha256"] == digest
    assert summary["splits"]["historical_random"]["rows"] == 40
