import pytest
import torch
from types import SimpleNamespace

from emap2lig import main as main_module
from emap2lig.main import _multiplicity_chunks


@pytest.mark.parametrize(
    ("multiplicity", "max_parallel_multiplicity", "expected"),
    [
        (1, 8, [1]),
        (8, 8, [8]),
        (9, 8, [8, 1]),
        (20, 8, [8, 8, 4]),
    ],
)
def test_multiplicity_chunks(multiplicity, max_parallel_multiplicity, expected) -> None:
    assert _multiplicity_chunks(multiplicity, max_parallel_multiplicity) == expected


@pytest.mark.parametrize(
    ("multiplicity", "max_parallel_multiplicity"),
    [(0, 8), (1, 0)],
)
def test_multiplicity_chunks_rejects_non_positive_values(
    multiplicity, max_parallel_multiplicity
) -> None:
    with pytest.raises(ValueError):
        _multiplicity_chunks(multiplicity, max_parallel_multiplicity)


def test_run_structure_modeling_uses_one_predict_with_capped_model_parallelism(
    monkeypatch, tmp_path
) -> None:
    blobs_dir = tmp_path / "find_blobs"
    blobs_dir.mkdir()
    (blobs_dir / "blob_1.npz").write_text("stub")

    ligands_dir = tmp_path / "preprocess" / "ligands"
    ligands_dir.mkdir(parents=True)
    (ligands_dir / "LIG.npz").write_text("stub")

    class FakeDataset:
        def __init__(self, density_object_list, ref_mol_dir, multiplicity):
            self.density_object_list = density_object_list
            self.ref_mol_dir = ref_mol_dir
            self.multiplicity = multiplicity

        def __len__(self):
            return 1

    trainer_calls: list[int] = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, model, dataloaders):
            trainer_calls.append(dataloaders.dataset.multiplicity)

    fake_model = SimpleNamespace(predict_args=SimpleNamespace(multiplicity=None))

    def fake_prepare_ligand_dataset(ligand_records, output_dir, ligands_dir=None):
        return 0, ligands_dir or tmp_path / "preprocess" / "ligands"

    monkeypatch.setattr(
        main_module,
        "prepare_ligand_dataset",
        fake_prepare_ligand_dataset,
    )
    monkeypatch.setattr(main_module, "LigandModelingDataset", FakeDataset)
    monkeypatch.setattr(
        main_module,
        "DataLoader",
        lambda dataset, **kwargs: SimpleNamespace(dataset=dataset, kwargs=kwargs),
    )
    monkeypatch.setattr(main_module, "instantiate", lambda cfg: fake_model)
    monkeypatch.setattr(main_module, "LigandWriter", lambda **kwargs: object())
    monkeypatch.setattr(main_module, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        main_module, "resolve_inference_device", lambda gpu: torch.device("mps")
    )
    monkeypatch.setattr(main_module, "create_blob_csv_tables", lambda output_dir: None)

    cfg = SimpleNamespace(num_workers=0, model=SimpleNamespace(_target_="FakeModel"))
    status = main_module.run_structure_modeling(
        blobs_dir=blobs_dir,
        output_dir=tmp_path,
        ligand_records=[],
        cfg=cfg,
        gpu=0,
        multiplicity=64,
        max_parallel_multiplicity=8,
    )

    assert status == 0
    assert trainer_calls == [64]
    assert fake_model.predict_args.multiplicity == 64
    assert fake_model.predict_args.max_parallel_multiplicity == 8
