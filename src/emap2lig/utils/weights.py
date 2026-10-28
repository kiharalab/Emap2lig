"""Pretrained weight files hosted on the Hugging Face Hub."""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from .storage import get_model_dir

logger = logging.getLogger(__name__)

WEIGHTS_REPO_ID = "KiharaLab/Emap2lig"
DETECTION_WEIGHTS = "emap2lig-find-v0.0.1.safetensors"
FRAGMENT_WEIGHTS = "emap2lig-frag.safetensors"
STRUCTURE_WEIGHTS = "emap2lig-build-v0.0.1.safetensors"

# Weights required by the main pipeline (fragment weights are fetched on demand).
MODEL_FILES: dict[str, str] = {
    "detection_model": DETECTION_WEIGHTS,
    "structure_model": STRUCTURE_WEIGHTS,
}


def get_model_path(filename: str) -> Path:
    """Return the local path of a weight file (``<home>/models/<filename>``)."""
    return get_model_dir() / filename


def fetch_weights(filename: str, repo_id: str = WEIGHTS_REPO_ID) -> Path:
    """Return the local path of a weight file, downloading it if missing."""
    local_path = get_model_path(filename)
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {repo_id}/{filename} → {local_path}")
    hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(local_path.parent)
    )
    return local_path


def check_weights() -> dict[str, bool]:
    """Map each :data:`MODEL_FILES` key to whether its file is present locally."""
    return {key: get_model_path(fn).exists() for key, fn in MODEL_FILES.items()}
