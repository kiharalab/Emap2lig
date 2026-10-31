from types import SimpleNamespace

import pytest
import torch

from emap2lig.config import InferenceConfig
from emap2lig.errors import Emap2ligError, JobCancelledError
from emap2lig.pipeline import build as build_module


@pytest.fixture
def fake_build(monkeypatch):
    """Replace the model, dataset and predict loop of the build stage with stubs."""

    class FakeDataset:
        def __init__(self, density_object_list, ref_mol_dir, multiplicity):
            self.density_object_list = density_object_list
            self.ref_mol_dir = ref_mol_dir
            self.multiplicity = multiplicity

        def __len__(self):
            return len(self.density_object_list)

    calls: list[dict] = []

    def fake_predict(model, dataloader, writer, device, on_progress=None):
        dataset = dataloader.dataset
        calls.append(
            {
                "blobs": [p.name for p in dataset.density_object_list],
                "ligands": sorted(p.name for p in dataset.ref_mol_dir.glob("*")),
                "multiplicity": dataset.multiplicity,
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
    monkeypatch.setattr(build_module, "_predict", fake_predict)
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


def test_predict_moves_batches_and_writes_each() -> None:
    class FakeModel(torch.nn.Module):
        def predict_step(self, batch, batch_idx, num_batches, on_step=None):
            assert not torch.is_grad_enabled()
            assert not self.training
            return {"total": num_batches, "x": batch["x"] * 2}

    class FakeWriter:
        def __init__(self):
            self.written = []
            self.finished = False
            self.failed = 0

        def write(self, prediction, batch, batch_idx):
            self.written.append((batch_idx, prediction, batch))

        def finish(self):
            self.finished = True

    batches = [{"x": torch.tensor([i]), "object_id": [str(i)]} for i in range(3)]
    writer = FakeWriter()

    build_module._predict(FakeModel(), batches, writer, torch.device("cpu"))

    assert writer.finished
    assert [idx for idx, _, _ in writer.written] == [0, 1, 2]
    assert [p["total"] for _, p, _ in writer.written] == [3, 3, 3]
    assert writer.written[2][1]["x"].item() == 4
    assert writer.written[2][2]["object_id"] == ["2"]


def test_predict_reports_failed_counts() -> None:
    class FlakyModel(torch.nn.Module):
        def predict_step(self, batch, batch_idx, num_batches, on_step=None):
            if batch_idx == 1:
                return None
            return {"ok": True}

    class FakeWriter:
        def __init__(self):
            self.written = []
            self.failed = 0

        def write(self, prediction, batch, batch_idx):
            self.written.append(batch_idx)
            if prediction is None:
                self.failed += 1

        def finish(self):
            pass

    events = []
    writer = FakeWriter()
    batches = [{"x": torch.tensor([i]), "identifier": [f"lig{i}"]} for i in range(3)]
    build_module._predict(
        FlakyModel(),
        batches,
        writer,
        torch.device("cpu"),
        on_progress=events.append,
    )

    assert writer.failed == 1
    assert writer.written == [0, 1, 2]
    assert [e.failed for e in events] == [0, 0, 1, 1]
    assert events[-1].current == 3
    assert events[-1].total == 3
    assert events[-1].stage == "predict"


def test_predict_reports_diffusion_steps() -> None:
    class SteppedModel(torch.nn.Module):
        predict_args = SimpleNamespace(
            multiplicity=1,
            max_parallel_multiplicity=8,
            num_sampling_steps=4,
        )

        def predict_step(self, batch, batch_idx, num_batches, on_step=None):
            if on_step is not None:
                for step in range(1, 5):
                    on_step(step)
            return {"ok": True}

    class FakeWriter:
        failed = 0

        def write(self, prediction, batch, batch_idx):
            pass

        def finish(self):
            pass

    events = []
    batches = [{"identifier": ["ATP"]}, {"identifier": ["NAD"]}]
    build_module._predict(
        SteppedModel(),
        batches,
        FakeWriter(),
        torch.device("cpu"),
        on_progress=events.append,
    )
    currents = [e.current for e in events if e.stage == "predict"]
    assert currents[0] == 0
    assert currents[-1] == 8
    assert currents == sorted(currents)
    assert all(e.total == 8 for e in events)


def test_predict_cancel_does_not_count_as_failed() -> None:
    class CancellingModel(torch.nn.Module):
        def predict_step(self, batch, batch_idx, num_batches, on_step=None):
            if on_step is not None:
                on_step(1)
            raise JobCancelledError("Job cancelled")

    class FakeWriter:
        def __init__(self):
            self.failed = 0
            self.written = []

        def write(self, prediction, batch, batch_idx):
            self.written.append(batch_idx)
            if prediction is None:
                self.failed += 1

        def finish(self):
            pass

    writer = FakeWriter()
    with pytest.raises(JobCancelledError):
        build_module._predict(
            CancellingModel(),
            [{"identifier": ["ATP"]}],
            writer,
            torch.device("cpu"),
        )
    assert writer.failed == 0
    assert writer.written == []
