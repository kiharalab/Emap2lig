"""Regression tests for host-memory-safe predict outputs."""

from types import SimpleNamespace

import pytest
import torch

from emap2lig.model.model import Emap2lig


def test_predict_step_returns_only_writer_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop bulky forward tensors so Lightning cannot retain voxel_features."""
    model = Emap2lig.__new__(Emap2lig)
    model.predict_args = SimpleNamespace(
        num_sampling_steps=20,
        multiplicity=1,
        max_parallel_multiplicity=8,
    )

    forward_output = {
        "sampled_atom_coords": torch.zeros(1, 4, 3),
        "instance_mask_output": torch.zeros(1, 1, 8, 8, 8),
        "voxel_features": torch.zeros(1, 64, 48, 48, 48),
        "augment_output": torch.zeros(1, 15, 48, 48, 48),
        "global_features": torch.zeros(1, 1, 256),
        "pair_dist_logits": torch.zeros(1, 4, 4, 20),
    }

    def fake_forward(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return forward_output

    monkeypatch.setattr(model, "forward", fake_forward)

    prediction = model.predict_step({"identifier": ["ATP"]}, batch_idx=0)

    assert prediction is not None
    assert set(prediction) == {"sampled_atom_coords", "instance_mask_output"}
    assert prediction["sampled_atom_coords"] is forward_output["sampled_atom_coords"]
    assert prediction["instance_mask_output"] is forward_output["instance_mask_output"]
