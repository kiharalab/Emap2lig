"""Pytest fixtures for CCD caching tests."""

from pathlib import Path

import pytest

from emap2lig.data import ccd as ccd_module


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest command line options."""
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Run tests that require network access.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "network: marks tests that require network")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip network-marked tests unless explicitly enabled."""
    if "network" in item.keywords and not item.config.getoption("--network"):
        pytest.skip("need --network option to run")


@pytest.fixture
def tmp_ccd_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch CCD cache directory to an isolated temp location."""
    ccd_dir = tmp_path / "ccd"
    ccd_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ccd_module, "_CCD_DIR", ccd_dir)
    ccd_module._load_bulk_dict.cache_clear()
    return ccd_dir
