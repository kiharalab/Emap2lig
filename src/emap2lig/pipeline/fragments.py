"""Fragment detection: per-voxel ring/fragment classes with FragmentRegSeg."""

import logging
from pathlib import Path

from ..config import DETECTION_SPATIAL_SIZE, InferenceConfig
from ..data.io.map import to_mrc
from ..model.pretrained import build_fragment_detection_model
from ..progress import ProgressCallback, emit
from ..utils.device import resolve_device
from ..utils.runtime import configure_runtime
from .preprocess import load_unified_map, map_stem

logger = logging.getLogger(__name__)


def run_fragment_detection(
    input_map: Path | str,
    output_dir: Path | str,
    cfg: InferenceConfig,
    emdb_id: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Save ``<stem>_frag_<label>.mrc`` and ``<stem>_frag_<label>_mask.mrc`` per class.

    Returns:
        *output_dir*.
    """
    output_dir = Path(output_dir)
    stem = map_stem(input_map)
    configure_runtime()
    device = resolve_device(cfg.gpu)

    emit(on_progress, "preprocess", 0, 1, f"Loading map from {input_map}")
    unified = load_unified_map(
        input_map, cfg, emdb_id, save_to=output_dir / f"{stem}_unified.mrc"
    )
    emit(on_progress, "preprocess", 1, 1, "Map resampled and normalized")
    logger.info("Initializing FragmentRegSeg model")
    emit(on_progress, "segment", 0, None, "Loading fragment detection model")
    model = build_fragment_detection_model()
    probabilities, binary, _ = model.segment(
        unified.grid_data,
        roi_size=DETECTION_SPATIAL_SIZE,
        batch_size=cfg.detection_batch_size,
        device=device,
        on_progress=on_progress,
    )
    for channel, label in enumerate(model.labels):
        prob_path = output_dir / f"{stem}_frag_{label}.mrc"
        mask_path = output_dir / f"{stem}_frag_{label}_mask.mrc"
        to_mrc(
            unified.with_grid(probabilities[channel].numpy()), prob_path, verbose=False
        )
        to_mrc(
            unified.with_grid(binary[channel].float().numpy()), mask_path, verbose=False
        )
        logger.info(f"Saved {label} prob to {prob_path} and mask to {mask_path}")

    logger.info("Fragment detection completed successfully")
    return output_dir
