"""Shared map preprocessing for the segmentation stages."""

import logging
from pathlib import Path

from ..config import DETECTION_SPATIAL_SIZE, InferenceConfig
from ..data.io.map import parse_mrc, to_mrc
from ..data.map import get_unified_mrc
from ..data.types import MapObject

logger = logging.getLogger(__name__)


def map_stem(input_map: Path | str) -> str:
    """``emd_7783.map.gz`` → ``emd_7783``."""
    return Path(input_map).name.split(".")[0]


def load_unified_map(
    input_map: Path | str,
    cfg: InferenceConfig,
    emdb_id: str | None,
    save_to: Path,
) -> MapObject:
    """Resample and normalize *input_map* for segmentation, saving it to *save_to*."""
    logger.info(f"Loading cryo-EM map from {input_map}")
    raw = parse_mrc(Path(input_map), emdb_id=emdb_id)
    unified = get_unified_mrc(
        raw,
        emdb_id=emdb_id,
        min_spatial_size=DETECTION_SPATIAL_SIZE,
        extended_val=cfg.extended_val,
        contour_level=cfg.contour_level,
    )
    save_to.parent.mkdir(parents=True, exist_ok=True)
    to_mrc(unified, save_to, verbose=False)
    return unified
