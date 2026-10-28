from typer.testing import CliRunner

import torch

from emap2lig import pipeline
from emap2lig.cli import common
from emap2lig.cli.main import app
from emap2lig.errors import NoBlobsDetectedError


def test_failed_stage_exits_non_zero(monkeypatch, tmp_path) -> None:
    def no_blobs(*args, **kwargs):
        raise NoBlobsDetectedError("No ligand blobs detected")

    def unreachable(*args, **kwargs):
        raise AssertionError("build stage must not run")

    monkeypatch.setattr(common, "resolve_device", lambda gpu: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "detect_ligand_objects", no_blobs)
    monkeypatch.setattr(pipeline, "run_structure_modeling", unreachable)
    ligands = tmp_path / "ligands.yaml"
    ligands.write_text("- CCD: NAG\n")

    result = CliRunner().invoke(
        app,
        [
            "--input-map",
            str(tmp_path / "map.mrc"),
            "--ligand-list",
            str(ligands),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1


def test_ligand_list_is_required(tmp_path) -> None:
    result = CliRunner().invoke(app, ["--input-map", str(tmp_path / "map.mrc")])
    assert result.exit_code == 2
