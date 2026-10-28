import pytest

from emap2lig.config import InferenceConfig
from emap2lig.layout import OutputLayout, blob_id_of


@pytest.mark.parametrize(
    "field",
    [
        "detection_batch_size",
        "multiplicity",
        "max_parallel_multiplicity",
        "num_sampling_steps",
    ],
)
def test_config_rejects_non_positive_counts(field) -> None:
    with pytest.raises(ValueError, match=field):
        InferenceConfig(**{field: 0})


def test_layout_paths(tmp_path) -> None:
    layout = OutputLayout(tmp_path)
    assert layout.unified_map == tmp_path / "preprocess" / "unified.mrc"
    assert layout.label_mask("ligand") == tmp_path / "find_maps" / "ligand_mask.mrc"
    assert layout.blob(3) == tmp_path / "find_blobs" / "blob_3.npz"
    assert layout.blob_mask(3) == tmp_path / "find_blobs" / "mask_3.mrc"
    assert layout.best == tmp_path / "build_struct" / "best"
    assert blob_id_of(layout.blob(12)) == 12


def test_layout_blob_paths_filters_by_id(tmp_path) -> None:
    layout = OutputLayout(tmp_path)
    layout.find_blobs.mkdir()
    for i in (1, 2, 10):
        layout.blob(i).write_text("stub")
    assert [p.name for p in layout.blob_paths()] == [
        "blob_1.npz",
        "blob_10.npz",
        "blob_2.npz",
    ]
    assert layout.blob_paths([2, 7]) == [layout.blob(2)]
