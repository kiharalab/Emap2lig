"""Local storage root for model weights and CCD data.

All paths derive from a single root, ``~/.emap2lig`` by default. Set the
``EMAP2LIG_HOME`` environment variable, or call :func:`set_home` at runtime,
to move it.
"""

import os
from pathlib import Path

_home = Path(os.environ.get("EMAP2LIG_HOME", Path.home() / ".emap2lig")).expanduser()


def get_home() -> Path:
    """Return the storage root (default ``~/.emap2lig``)."""
    return _home


def set_home(path: str | Path) -> Path:
    """Move the storage root for the rest of this process."""
    global _home
    _home = Path(path).expanduser().resolve()
    return _home


def get_model_dir() -> Path:
    """Return the directory holding model weight files."""
    return _home / "models"


def get_ccd_dir() -> Path:
    """Return the directory holding cached CCD dictionaries."""
    return _home / "ccd"
