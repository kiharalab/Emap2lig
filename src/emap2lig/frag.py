#!/usr/bin/env python3
from pathlib import Path

import torch
import typer
from hydra import initialize, compose
from hydra.utils import instantiate

from emap2lig.data.io.map import parse_mrc, to_mrc
from emap2lig.data.map import get_unified_mrc
from emap2lig.data.types import MapObject
from emap2lig.main import _require_accelerator, resolve_inference_device


from loguru import logger


def run_fragment_detection(input_map: str, output_dir: str, cfg, emdb_id: str | None):
    """Run FragmentRegSeg on a map and save per-class predictions and masks."""
    # Prepare paths
    input_path = Path(input_map)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce platform accelerator inference.
    device = resolve_inference_device(int(cfg.gpu))

    map_stem = str(input_path.stem).split(".")[0]
    unified_map_path = output_dir / f"{map_stem}_unified.mrc"

    fragment_spatial_size = int(cfg.fragment_detection_model.spatial_size)

    # Load and unify map
    logger.info(f"Loading cryo-EM map from {input_path}")
    raw_obj = parse_mrc(input_path, emdb_id=emdb_id)
    unified_obj = get_unified_mrc(
        raw_obj,
        min_spatial_size=fragment_spatial_size,
        extended_val=cfg.extended_val,
        contour_level=cfg.contour_level,
    )
    to_mrc(unified_obj, unified_map_path, verbose=False)

    # Convert to tensor [1,1,D,H,W]
    input_map_tensor = (
        torch.from_numpy(unified_obj.grid_data).float().unsqueeze(0).unsqueeze(0)
    )

    # Instantiate model
    logger.info("Initializing FragmentRegSeg model from configuration")
    model = instantiate(cfg.fragment_detection_model)

    output_device = torch.device("cpu")

    # Inference
    logger.info(f"Running fragment detection on device: {device}")
    predicted_map = model.sliding_window_inference(
        input_map=input_map_tensor,
        roi_size=fragment_spatial_size,
        batch_size=cfg.detection_batch_size,
        device=device,
        output_device=output_device,
    )

    # Binarize predictions per channel
    binary_map, _ = model._binarize_output(predicted_map)

    # Determine labels (use the labels as defined in the model)
    labels = model.labels

    # Save per-class maps and masks
    for c, label in enumerate(labels):
        # Probability map
        prob_np = predicted_map[0, c, ...].numpy()
        prob_obj = MapObject(
            grid_data=prob_np,
            voxel_size=unified_obj.voxel_size,
            global_origin=unified_obj.global_origin,
            emdb_id=unified_obj.emdb_id,
        )
        prob_path = output_dir / f"{map_stem}_frag_{label}.mrc"
        to_mrc(prob_obj, prob_path, verbose=False)

        # Binary mask
        mask_np = binary_map[0, c, ...].float().numpy()
        mask_obj = MapObject(
            grid_data=mask_np,
            voxel_size=unified_obj.voxel_size,
            global_origin=unified_obj.global_origin,
            emdb_id=unified_obj.emdb_id,
        )
        mask_path = output_dir / f"{map_stem}_frag_{label}_mask.mrc"
        to_mrc(mask_obj, mask_path, verbose=False)

        logger.info(f"Saved {label} prob to {prob_path} and mask to {mask_path}")

    logger.info("Fragment detection completed successfully")
    return 0


app = typer.Typer()


@app.command()
def main(
    input_map: str = typer.Option(
        ..., "--input-map", help="Path to input cryo-EM map file"
    ),
    output_dir: str = typer.Option(
        "./output", "--output-dir", help="Directory to save outputs"
    ),
    gpu: int = typer.Option(0, "--gpu", help="Accelerator device ID"),
    detection_batch_size: int | None = typer.Option(
        None,
        "--detection-batch-size",
        help="Batch size for detection model sliding window inference",
    ),
    emdb_id: str | None = typer.Option(None, "--emdb-id", help="EMDB ID for the map"),
    contour_level: float | None = typer.Option(
        None, "--contour-level", help="Contour level for the map"
    ),
):
    """Run fragment detection using FragmentRegSeg and save predictions."""
    try:
        _require_accelerator(gpu)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu") from exc

    # Load configuration
    overrides = []
    if gpu is not None:
        overrides.append(f"gpu={gpu}")
    if detection_batch_size is not None:
        overrides.append(f"detection_batch_size={detection_batch_size}")
    if contour_level is not None:
        overrides.append(f"contour_level={contour_level}")

    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="emap2lig", overrides=overrides)

    # Run
    return run_fragment_detection(input_map, output_dir, cfg, emdb_id)


if __name__ == "__main__":
    app()
