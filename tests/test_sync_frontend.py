"""Tests for ``scripts/sync_frontend.py`` (local vs npm frontend sync)."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_frontend.py"


def _load_script() -> ModuleType:
    """Import the standalone sync script as a module."""
    spec = importlib.util.spec_from_file_location("sync_frontend", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sync() -> ModuleType:
    """Loaded ``scripts/sync_frontend.py`` module."""
    return _load_script()


def _write_frontend(
    root: Path, *, name: str = "emap2lig-webui", version: str = "0.7.0"
) -> Path:
    """Create a fake Emap2lig-web checkout with a built ``dist/``."""
    root.mkdir(parents=True)
    (root / "package.json").write_text(json.dumps({"name": name, "version": version}))
    dist = root / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>local</html>")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ok')")
    return root


def test_local_sync_copies_dist(sync: ModuleType, tmp_path: Path) -> None:
    webui = _write_frontend(tmp_path / "Emap2lig-web")
    dest = tmp_path / "frontend" / "dist"
    pin = tmp_path / ".webui-version"
    pin.write_text("0.6.3\n")

    args = sync.parse_args(["--local", str(webui)])
    sync.run(args, dest=dest, version_file=pin, root=tmp_path)

    assert (dest / "index.html").read_text() == "<html>local</html>"
    assert (dest / "assets" / "app.js").is_file()
    assert pin.read_text() == "0.6.3\n"


def test_local_sync_default_sibling(sync: ModuleType, tmp_path: Path) -> None:
    repo = tmp_path / "Emap2lig"
    repo.mkdir()
    webui = _write_frontend(tmp_path / "Emap2lig-web")
    dest = repo / "frontend" / "dist"

    args = sync.parse_args(["--local"])
    sync.run(args, dest=dest, version_file=repo / ".webui-version", root=repo)

    assert (dest / "index.html").is_file()
    assert webui.is_dir()


def test_local_sync_env_override(
    sync: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    webui = _write_frontend(tmp_path / "custom-web")
    dest = tmp_path / "dist"
    monkeypatch.setenv("EMAP2LIG_WEBUI", str(webui))

    args = sync.parse_args(["--local"])
    sync.run(args, dest=dest, version_file=tmp_path / ".webui-version", root=tmp_path)

    assert (dest / "index.html").is_file()


def test_local_rejects_wrong_package(sync: ModuleType, tmp_path: Path) -> None:
    webui = _write_frontend(tmp_path / "other", name="not-emap2lig")
    dest = tmp_path / "dist"
    args = sync.parse_args(["--local", str(webui)])
    with pytest.raises(SystemExit, match="expected 'emap2lig-webui'"):
        sync.run(
            args, dest=dest, version_file=tmp_path / ".webui-version", root=tmp_path
        )


def test_local_requires_dist(sync: ModuleType, tmp_path: Path) -> None:
    webui = tmp_path / "Emap2lig-web"
    webui.mkdir()
    (webui / "package.json").write_text(
        json.dumps({"name": "emap2lig-webui", "version": "0.7.0"})
    )
    args = sync.parse_args(["--local", str(webui)])
    with pytest.raises(SystemExit, match=r"No index\.html"):
        sync.run(
            args,
            dest=tmp_path / "dist",
            version_file=tmp_path / ".webui-version",
            root=tmp_path,
        )


def test_local_build_runs_bun(
    sync: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    webui = tmp_path / "Emap2lig-web"
    webui.mkdir()
    (webui / "package.json").write_text(
        json.dumps({"name": "emap2lig-webui", "version": "0.7.0"})
    )
    dest = tmp_path / "dist"
    bun = tmp_path / "bun"
    bun.write_text("#!/bin/sh\n")
    bun.chmod(0o755)

    def fake_which(name: str) -> str | None:
        return str(bun) if name == "bun" else None

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> object:
        assert cmd == [str(bun), "run", "build"]
        assert cwd == webui
        dist = webui / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>built</html>")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(sync.shutil, "which", fake_which)
    monkeypatch.setattr(sync.subprocess, "run", fake_run)

    args = sync.parse_args(["--local", str(webui), "--build"])
    sync.run(args, dest=dest, version_file=tmp_path / ".webui-version", root=tmp_path)
    assert (dest / "index.html").read_text() == "<html>built</html>"


def test_build_requires_local(sync: ModuleType) -> None:
    args = sync.parse_args(["--build"])
    with pytest.raises(SystemExit, match="--build requires --local"):
        sync.run(args)


def test_local_rejects_npm_version(sync: ModuleType, tmp_path: Path) -> None:
    args = sync.parse_args(["0.7.0", "--local", str(tmp_path)])
    with pytest.raises(SystemExit, match="not both"):
        sync.run(
            args, dest=tmp_path / "dist", version_file=tmp_path / "v", root=tmp_path
        )


def test_npm_sync_pins_version(
    sync: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "dist"
    pin = tmp_path / ".webui-version"
    pin.write_text("0.6.3\n")

    tarball_buf = io.BytesIO()
    with tarfile.open(fileobj=tarball_buf, mode="w:gz") as tar:
        content = b"<html>npm</html>"
        info = tarfile.TarInfo(name="package/dist/index.html")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    tarball = tarball_buf.getvalue()
    integrity = (
        "sha512-"
        + __import__("base64")
        .b64encode(__import__("hashlib").sha512(tarball).digest())
        .decode()
    )

    meta = json.dumps(
        {"dist": {"tarball": "https://example.test/pkg.tgz", "integrity": integrity}}
    ).encode()

    def fake_fetch(url: str) -> bytes:
        if url.endswith("/emap2lig-webui/0.7.0"):
            return meta
        return tarball

    monkeypatch.setattr(sync, "_fetch", fake_fetch)

    args = sync.parse_args(["0.7.0"])
    sync.run(args, dest=dest, version_file=pin, root=tmp_path)

    assert (dest / "index.html").read_text() == "<html>npm</html>"
    assert pin.read_text() == "0.7.0\n"


def test_extract_rejects_path_traversal(sync: ModuleType, tmp_path: Path) -> None:
    tarball_buf = io.BytesIO()
    with tarfile.open(fileobj=tarball_buf, mode="w:gz") as tar:
        content = b"nope"
        info = tarfile.TarInfo(name="package/dist/../../evil.txt")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    with pytest.raises(SystemExit, match="Unsafe path"):
        sync._extract_dist(tarball_buf.getvalue(), tmp_path)


def test_invalid_npm_version(sync: ModuleType, tmp_path: Path) -> None:
    args = sync.parse_args(["not-a-version"])
    with pytest.raises(SystemExit, match="Invalid version"):
        sync.run(
            args, dest=tmp_path / "dist", version_file=tmp_path / "v", root=tmp_path
        )
