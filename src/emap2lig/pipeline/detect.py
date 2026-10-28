"""Find stage: segment the map and split the ligand mask into density blobs."""

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import (
    DETECTION_LABELS,
    DETECTION_SPATIAL_SIZE,
    STRUCTURE_SPATIAL_SIZE,
    InferenceConfig,
)
from ..data.io.map import parse_mrc, to_mrc
from ..data.map import crop_mrcs
from ..data.types import DensityObject, MapObject
from ..errors import NoBlobsDetectedError, TooManyBlobsError
from ..layout import OutputLayout
from ..model.pretrained import build_detection_model
from ..utils.device import resolve_device
from ..utils.runtime import configure_runtime
from .preprocess import load_unified_map

logger = logging.getLogger(__name__)

MIN_BLOB_VOXELS = 32
MAX_BLOBS = 100


def detect_ligand_objects(
    input_map: Path | str,
    output_dir: Path | str,
    cfg: InferenceConfig,
    emdb_id: str | None = None,
) -> Path:
    """Detect ligand density blobs in a cryo-EM map (stage 1).

    Writes the unified map, per-label probability maps and masks, and one
    ``blob_<i>.npz`` / ``mask_<i>.mrc`` pair per blob.  Label maps already in
    *output_dir* are reused.

    Returns:
        The ``find_blobs/`` directory.

    Raises:
        NoBlobsDetectedError: The ligand mask has no blob of usable size.
        TooManyBlobsError: More than ``MAX_BLOBS`` blobs were found (they are
            still saved).
    """
    layout = OutputLayout(Path(output_dir))
    configure_runtime()
    device = resolve_device(cfg.gpu)

    unified = load_unified_map(input_map, cfg, emdb_id, save_to=layout.unified_map)
    if all(
        layout.label_map(label).exists() and layout.label_mask(label).exists()
        for label in DETECTION_LABELS
    ):
        logger.info(f"Reusing detection maps under {layout.find_maps}")
        ligand_mask = parse_mrc(layout.label_mask("ligand"))
    else:
        ligand_mask = _segment(unified, layout, cfg, device)

    num_blobs = _split_blobs(ligand_mask, unified, layout)
    if num_blobs == 0:
        raise NoBlobsDetectedError("No ligand blobs detected")
    if num_blobs > MAX_BLOBS:
        raise TooManyBlobsError(
            f"Detected {num_blobs} blobs (more than {MAX_BLOBS}); the map is "
            "likely too noisy or the contour level too low to model ligands"
        )
    return layout.find_blobs


def _segment(
    unified: MapObject, layout: OutputLayout, cfg: InferenceConfig, device
) -> MapObject:
    """Run the detection model; save non-empty label maps; return the ligand mask."""
    logger.info("Initializing Emap2lig detection model")
    model = build_detection_model()
    probabilities, binary, thresholds = model.segment(
        unified.grid_data,
        roi_size=DETECTION_SPATIAL_SIZE,
        batch_size=cfg.detection_batch_size,
        device=device,
    )
    logger.info(f"Ligand detection threshold: {thresholds['ligand']:.6f}")

    layout.find_maps.mkdir(parents=True, exist_ok=True)
    ligand_mask = None
    for channel, label in enumerate(DETECTION_LABELS):
        mask = unified.with_grid(binary[channel].numpy())
        if label == "ligand":
            ligand_mask = mask
        if np.count_nonzero(mask.grid_data) == 0:
            logger.info(f"Skipping {label} map/mask save: binarized map is all zero")
            continue
        logger.info(f"Saving {label} map and mask under {layout.find_maps}")
        to_mrc(
            unified.with_grid(probabilities[channel].numpy()),
            layout.label_map(label),
            verbose=False,
        )
        to_mrc(mask, layout.label_mask(label), verbose=False)

    logger.info("Ligand detection completed successfully")
    assert ligand_mask is not None
    return ligand_mask


def _split_blobs(
    ligand_mask: MapObject, unified: MapObject, layout: OutputLayout
) -> int:
    """Save each 26-connected component of *ligand_mask* as a cropped blob."""
    labeled, num_components = ndimage.label(
        ligand_mask.grid_data, structure=np.ones((3, 3, 3), dtype=int)
    )
    logger.info(f"Found {num_components} objects in the ligand mask")
    sizes = np.bincount(labeled.ravel())  # index 0 is background

    layout.find_blobs.mkdir(parents=True, exist_ok=True)
    blob_id = 0
    for component in range(1, num_components + 1):
        if sizes[component] < MIN_BLOB_VOXELS:  # likely noise
            continue
        blob_id += 1
        mask = ligand_mask.with_grid((labeled == component).astype(np.uint8))
        cropped, _ = crop_mrcs(
            mrc_objects={"mask": mask, "unified": unified},
            excluded_keys=["unified"],
            min_spatial_size=STRUCTURE_SPATIAL_SIZE,
        )
        DensityObject(
            object_id=blob_id,
            density_grid=cropped["unified"].grid_data,
            instance_grid=cropped["mask"].grid_data,
            global_origin=cropped["unified"].global_origin,
            voxel_size=cropped["unified"].voxel_size,
        ).dump(layout.blob(blob_id))
        to_mrc(cropped["mask"], layout.blob_mask(blob_id), verbose=False)
        logger.info(f"Saved blob {blob_id} to {layout.blob(blob_id)}")

    logger.info(f"Saved {blob_id} blobs to {layout.find_blobs}")
    return blob_id
