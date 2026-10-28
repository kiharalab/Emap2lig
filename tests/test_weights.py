"""Tests for the storage root and weight-file helpers."""

from pathlib import Path

import pytest

from emap2lig.utils import storage, weights


@pytest.fixture(autouse=True)
def _restore_home():
    original = storage.get_home()
    yield
    storage.set_home(original)


def test_set_home_moves_model_paths(tmp_path: Path) -> None:
    storage.set_home(tmp_path)

    assert storage.get_model_dir() == tmp_path / "models"
    assert (
        weights.get_model_path("w.safetensors") == tmp_path / "models" / "w.safetensors"
    )


def test_fetch_weights_downloads_into_current_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_download(repo_id: str, filename: str, local_dir: str) -> str:
        calls.append(local_dir)
        target = Path(local_dir) / filename
        target.write_bytes(b"")
        return str(target)

    monkeypatch.setattr(weights, "hf_hub_download", fake_download)
    storage.set_home(tmp_path)

    path = weights.fetch_weights("w.safetensors", repo_id="repo")
    assert path == tmp_path / "models" / "w.safetensors"
    assert calls == [str(tmp_path / "models")]

    # A second call finds the cached file and does not download again.
    assert weights.fetch_weights("w.safetensors", repo_id="repo") == path
    assert len(calls) == 1


def test_web_set_cache_dir_reaches_model_loading(tmp_path: Path) -> None:
    from emap2lig.web.services import get_cache_info, set_cache_dir

    info = set_cache_dir(str(tmp_path))

    assert info["cache_dir"] == str(tmp_path.resolve())
    assert storage.get_model_dir() == tmp_path.resolve() / "models"
    assert get_cache_info()["models"][0]["path"].startswith(str(tmp_path.resolve()))
