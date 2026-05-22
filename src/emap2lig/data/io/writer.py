import csv
from typing import Literal
from pathlib import Path

import numpy as np
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import Tensor

from emap2lig.data.io.mmcif import to_mmcif
from emap2lig.data.io.map import to_mrc
from emap2lig.data.simulate import gaussian_blur_and_iou
from emap2lig.data.types import LigandObject, MapObject
from emap2lig.model.seg.threshold import threshold_li

from loguru import logger


def to_cmm(
    path: Path,
    coords: np.ndarray,
    *,
    name: str = "prompt_points",
    rgb: tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius: float = 1.0,
) -> None:
    """Write a Chimera marker file (.cmm) containing one marker per coordinate.

    Parameters
    ----------
    path : Path
        Destination file path (should end in ``.cmm``).
    coords : np.ndarray
        Marker positions, shape ``(N, 3)`` or ``(3,)`` for a single point.
    name : str
        ``<marker_set name="...">`` attribute.
    rgb : tuple[float, float, float]
        Marker colour as (r, g, b) in [0, 1].
    radius : float
        Sphere radius in Angstroms.
    """
    coords = np.atleast_2d(coords)
    r, g, b = rgb
    lines = [f'<marker_set name="{name}">']
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f'  <marker id="{i}" x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"'
            f' r="{r}" g="{g}" b="{b}" radius="{radius}" />'
        )
    lines.append("</marker_set>\n")
    path.write_text("\n".join(lines))


def _resolve_ligand_name(name) -> str:
    """Normalise ``LigandObject.name`` into a filesystem-safe string."""
    if isinstance(name, list):
        return "-".join(name) if len(name) > 1 else name[0]
    if isinstance(name, np.ndarray):
        if name.ndim == 0:
            return str(name.item())
        if name.ndim == 1 and len(name) > 0:
            parts = [str(x) for x in name]
            return "-".join(parts) if len(parts) > 1 else parts[0]
        return str(name)
    if isinstance(name, str):
        return name
    return str(name)


class LigandWriter(BasePredictionWriter):
    """Custom writer for ligand predictions."""

    def __init__(
        self,
        output_dir: Path,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        output_format: Literal["pdb", "mmcif"] = "mmcif",
    ):
        super().__init__(write_interval)

        self.output_dir = output_dir
        self.output_format = output_format
        self.failed = 0
        self.all = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: dict[str, Tensor],
        batch_indices: list[int],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write predictions for a batch.

        Parameters
        ----------
        prediction : dict[str, Tensor]
            Model predictions
        batch : dict[str, Tensor]
            Input batch data
        """
        # Skip if prediction is None or failed
        self.all += 1
        if prediction is None:
            logger.warning(f"Prediction returned None for batch {batch_idx}")
            self.failed += 1
            return
        if prediction.get("exception", False):
            logger.warning(f"Prediction failed for batch {batch_idx}")
            self.failed += 1
            return

        ligand_paths = batch["smiles_path"]
        ligand_identifiers = [f"blob_{oid}" for oid in batch["object_id"]]
        logger.info(f"Shape of smiles paths: {len(ligand_paths)}")
        logger.info(f"Shape of object ids: {len(batch['object_id'])}")

        # Get predictions
        coords = prediction["sampled_atom_coords"]  # [B*M, N, 3]
        logger.info(f"Shape of sampled atom coords: {coords.shape}")
        # Process each ligand in batch
        batch_size = len(ligand_paths)
        multiplicity = coords.shape[0] // batch_size

        logger.info(f"Writing {batch_size} ligands with multiplicity {multiplicity}")

        for batch_idx in range(batch_size):
            # Get ligand info
            ligand_path = ligand_paths[batch_idx]
            ligand_id = ligand_identifiers[batch_idx]

            ligand_object: LigandObject = LigandObject.load(Path(ligand_path))

            center = batch["groundtruth_center"][batch_idx].unsqueeze(0)  # [1, 3]
            pred_mask = batch["atom_mask"][batch_idx].bool()  # [N]

            logger.info(f"Processing ligand {ligand_id} from {ligand_path}")

            # Create output directory for this ligand
            ligand_dir = self.output_dir / ligand_id
            ligand_dir.mkdir(exist_ok=True)

            # Resolve the ligand display name once (same for every model_idx)
            actual_ligand_name = _resolve_ligand_name(ligand_object.name)

            # Count existing conformer CIF files for this ligand so that
            # incremental runs do not overwrite previous predictions.
            existing_cifs = list(
                ligand_dir.glob(f"{ligand_id}_{actual_ligand_name}_*.cif")
            )
            idx_offset = len(existing_cifs)
            if idx_offset > 0:
                logger.info(
                    f"Found {idx_offset} existing conformers for "
                    f"{ligand_id}/{actual_ligand_name}; new indices start at {idx_offset + 1}"
                )

            # Process each sampled conformation
            for model_idx in range(multiplicity):
                pred_idx = batch_idx * multiplicity + model_idx
                logger.info(f"Processing model {model_idx} for ligand {ligand_id}")

                # Get coordinates and mask for this prediction
                pred_coords = coords[pred_idx] + center  # [N, 3]

                # masked_coords
                masked_coords = pred_coords[pred_mask]
                masked_mask = pred_mask[pred_mask]

                # check the size of ligand.atoms["coords"] and masked_coords
                if len(ligand_object.atoms["coords"]) != len(masked_coords):
                    logger.warning(
                        f"Ligand {ligand_id} has {len(ligand_object.atoms['coords'])} atoms, but {len(masked_coords)} atoms in prediction"
                    )

                # Update ligand coordinates
                ligand_object.atoms["coords"] = masked_coords.cpu().numpy()
                ligand_object.atoms["is_present"] = masked_mask.cpu().numpy()

                # Create output name: blob_X_LigandName_Y (used for both .cif and .mrc files)
                outname = (
                    f"{ligand_id}_{actual_ligand_name}_{idx_offset + model_idx + 1}"
                )

                # Calculate consistency_iou using predicted instance_mask_output from model
                consistency_iou_value = None
                if (
                    "instance_mask_output" in prediction
                    and "voxel_size" in batch
                    and "global_origin" in batch
                ):
                    # Use predicted instance mask from model output
                    # prediction["instance_mask_output"] has shape [B*M, 1, D, H, W]
                    instance_mask_tensor = prediction["instance_mask_output"][pred_idx]

                    if instance_mask_tensor.dim() == 4:
                        # [1, D, H, W] -> remove channel dim
                        instance_mask_tensor = instance_mask_tensor[0]  # [D, H, W]

                    if instance_mask_tensor.dim() != 3:
                        raise ValueError(
                            "Unsupported instance_mask_output shape after squeezing: "
                            f"{tuple(instance_mask_tensor.shape)}"
                        )

                    # The instance_mask_output is a probability mask, binarize using Li thresholding
                    instance_mask_probs = instance_mask_tensor.float()
                    threshold = threshold_li(
                        instance_mask_probs.flatten(), initial_guess=0.1
                    )
                    instance_mask_probs_np = instance_mask_probs.cpu().numpy()
                    instance_mask = (
                        instance_mask_probs_np > threshold.cpu().item()
                    ).astype(np.float32)

                    voxel_size = batch["voxel_size"][batch_idx].cpu().numpy()  # [3]
                    global_origin = (
                        (
                            batch["global_origin"][batch_idx]
                            + batch["groundtruth_center"][batch_idx]
                        )
                        .cpu()
                        .numpy()
                    )  # [3]

                    # Save the predicted instance mask as MRC file
                    pred_mask_object = MapObject(
                        grid_data=instance_mask_probs_np,
                        voxel_size=voxel_size,
                        global_origin=global_origin,
                    )
                    pred_mask_path = ligand_dir / f"{outname}_pred_mask.mrc"
                    to_mrc(pred_mask_object, pred_mask_path, verbose=False)
                    logger.info(f"Saved predicted instance mask to {pred_mask_path}")

                    # Quick sanity check: are predicted coords inside the grid bounds?
                    # If most atoms fall outside, IoU will be ~0.0 even with good structures.
                    grid_indices_xyz = (
                        masked_coords.cpu().numpy() - global_origin
                    ) / voxel_size
                    grid_indices_zyx = grid_indices_xyz[:, ::-1]
                    mins = (
                        grid_indices_zyx.min(axis=0) if len(grid_indices_zyx) else None
                    )
                    maxs = (
                        grid_indices_zyx.max(axis=0) if len(grid_indices_zyx) else None
                    )
                    if mins is not None and maxs is not None:
                        d, h, w = instance_mask.shape
                        outside = np.any(
                            (grid_indices_zyx < 0)
                            | (
                                grid_indices_zyx
                                >= np.array([d, h, w], dtype=np.float32)
                            ),
                            axis=1,
                        )
                        outside_frac = float(np.mean(outside)) if len(outside) else 0.0
                        logger.info(
                            "IoU debug for {outname}: grid zyx mins={mins} maxs={maxs} outside_frac={outside_frac} grid_shape={instance_mask.shape}",
                        )

                    # Calculate consistency_iou using the gaussian_blur_and_iou function
                    _pred_mask_3d, consistency_iou_value = gaussian_blur_and_iou(
                        coords=masked_coords.cpu().numpy(),
                        reference_mask=instance_mask,
                        grid_shape=instance_mask.shape,
                        voxel_size=voxel_size,
                        global_origin=global_origin,
                        threshold=0.01,
                    )

                # Save ligand data

                if consistency_iou_value is not None:
                    logger.info(
                        f"Calculated consistency_iou: {consistency_iou_value:.4f} for {outname}"
                    )

                if self.output_format == "mmcif":
                    out_path = ligand_dir / f"{outname}.cif"
                    to_mmcif(ligand_object, out_path)
                else:
                    raise ValueError(f"Unsupported output format: {self.output_format}")

                logger.info(f"Wrote ligand {outname} to {out_path}")

                # Write the prompt point used for this conformer as a .cmm marker file
                if "prompt_points" in batch:
                    prompt_pt = batch["prompt_points"][
                        batch_idx, model_idx
                    ] + center.squeeze(0)
                    cmm_path = ligand_dir / f"{outname}_prompt.cmm"
                    to_cmm(
                        cmm_path,
                        prompt_pt.cpu().numpy(),
                        name=f"{outname}_prompt",
                    )
                    logger.info(f"Wrote prompt point to {cmm_path}")

                iou_val = (
                    float(consistency_iou_value)
                    if consistency_iou_value is not None
                    else 0.0
                )
                self._append_csv(ligand_dir, ligand_id, outname, iou_val)

    @staticmethod
    def _append_csv(
        ligand_dir: Path, blob_id: str, conformer_name: str, iou: float
    ) -> None:
        """Append a single conformer row to the per-blob CSV file.

        Creates the file with a header if it doesn't exist yet.
        """
        csv_path = ligand_dir / f"{blob_id}_results.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["conformer_name", "consistency_iou"])
            w.writerow([conformer_name, iou])

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Print summary statistics."""
        print(f"Number of failed / total predictions: {self.failed} / {self.all}")
