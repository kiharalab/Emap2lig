"""Local storage locations for model weights and CCD data.

All paths derive from a single root, ``~/.emap2lig`` by default. Set the
``EMAP2LIG_HOME`` environment variable, or call :func:`set_home` at runtime,
to move it.
"""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

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


def fetch_weights(repo_id: str, filename: str) -> Path:
    """Return the local path of a weight file, downloading it if missing."""
    model_dir = get_model_dir()
    local_path = model_dir / filename
    if local_path.exists():
        return local_path
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {repo_id}/{filename} → {local_path}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(model_dir))
    return local_path
