from types import SimpleNamespace

import pytest
import torch

from emap2lig.config import InferenceConfig
from emap2lig.errors import Emap2ligError
from emap2lig.pipeline import build as build_module


@pytest.fixture
def fake_build(monkeypatch):
    """Replace the model, dataset and trainer of the build stage with stubs."""

    class FakeDataset:
        def __init__(self, density_object_list, ref_mol_dir, multiplicity):
            self.density_object_list = density_object_list
            self.ref_mol_dir = ref_mol_dir
            self.multiplicity = multiplicity

        def __len__(self):
            return len(self.density_object_list)

    calls: list[dict] = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def predict(self, model, dataloaders, return_predictions=None):
            dataset = dataloaders.dataset
            calls.append(
                {
                    "blobs": [p.name for p in dataset.density_object_list],
                    "ligands": sorted(p.name for p in dataset.ref_mol_dir.glob("*")),
                    "multiplicity": dataset.multiplicity,
                    "return_predictions": return_predictions,
                }
            )

    def fake_prepare(records, ligands_dir):
        ligands_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            (ligands_dir / f"{record}.npz").write_text("stub")

    model = SimpleNamespace(predict_args=SimpleNamespace())
    monkeypatch.setattr(build_module, "prepare_ligands", fake_prepare)
    monkeypatch.setattr(build_module, "LigandModelingDataset", FakeDataset)
    monkeypatch.setattr(
        build_module,
        "DataLoader",
        lambda dataset, **kwargs: SimpleNamespace(dataset=dataset),
    )
    monkeypatch.setattr(build_module, "build_structure_model", lambda **kw: model)
    monkeypatch.setattr(build_module, "LigandWriter", lambda **kwargs: object())
    monkeypatch.setattr(build_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(build_module, "resolve_device", lambda gpu: torch.device("mps"))
    monkeypatch.setattr(build_module, "rank_conformers", lambda *args: None)
    return SimpleNamespace(calls=calls, model=model)


def _write_blobs(root, ids):
    blobs_dir = root / "find_blobs"
    blobs_dir.mkdir()
    for i in ids:
        (blobs_dir / f"blob_{i}.npz").write_text("stub")


def test_one_predict_pass_with_capped_parallelism(fake_build, tmp_path) -> None:
    _write_blobs(tmp_path, [1])
    cfg = InferenceConfig(multiplicity=64, max_parallel_multiplicity=8)

    out = build_module.run_structure_modeling(tmp_path, ["NAG"], cfg)

    assert out == tmp_path / "build_struct"
    assert fake_build.calls == [
        {
            "blobs": ["blob_1.npz"],
            "ligands": ["NAG.npz"],
            "multiplicity": 64,
            "return_predictions": False,
        }
    ]
    assert fake_build.model.predict_args.multiplicity == 64
    assert fake_build.model.predict_args.max_parallel_multiplicity == 8


def test_incremental_run_isolates_ligands(fake_build, tmp_path) -> None:
    _write_blobs(tmp_path, [1, 2, 3])
    ligands = tmp_path / "preprocess" / "ligands"
    ligands.mkdir(parents=True)
    (ligands / "OLD.npz").write_text("stub")

    build_module.run_structure_modeling(
        tmp_path, ["HEM"], InferenceConfig(), blob_ids=[2]
    )

    assert fake_build.calls[0]["blobs"] == ["blob_2.npz"]
    assert fake_build.calls[0]["ligands"] == ["HEM.npz"]
    # The temporary ligands directory is removed afterwards.
    assert sorted(p.name for p in (tmp_path / "preprocess").iterdir()) == ["ligands"]


def test_missing_blobs_or_ligands_raise(fake_build, tmp_path) -> None:
    with pytest.raises(Emap2ligError, match="No blob files"):
        build_module.run_structure_modeling(tmp_path, ["NAG"], InferenceConfig())
    _write_blobs(tmp_path, [1])
    with pytest.raises(Emap2ligError, match="No ligand"):
        build_module.run_structure_modeling(tmp_path, [], InferenceConfig())
