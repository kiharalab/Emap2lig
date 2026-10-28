"""Build stage: diffusion-based ligand structure modeling on detected blobs."""

import logging
import shutil
import uuid
from pathlib import Path

from lightning import Trainer
from torch.utils.data import DataLoader

from ..config import InferenceConfig
from ..data.dataset import LigandModelingDataset, collate_fn
from ..data.featurize import prepare_ligands
from ..data.io.writer import LigandWriter
from ..data.types import LigandRecord
from ..errors import Emap2ligError
from ..layout import OutputLayout
from ..model.pretrained import build_structure_model
from ..utils.device import resolve_device, trainer_device_kwargs
from ..utils.runtime import configure_runtime
from .ranking import rank_conformers

logger = logging.getLogger(__name__)


def run_structure_modeling(
    output_dir: Path | str,
    ligand_records: list[LigandRecord],
    cfg: InferenceConfig,
    blob_ids: list[int] | None = None,
) -> Path:
    """Model each ligand into each blob found by :func:`detect_ligand_objects`.

    Generates ``cfg.multiplicity`` conformers per blob-ligand pair (at most
    ``cfg.max_parallel_multiplicity`` per forward pass), ranks them, and copies
    the best one per blob into ``build_struct/best/``.

    Args:
        output_dir: Output directory of the find stage.
        ligand_records: Ligands to model; see :func:`parse_ligand_list`.
        cfg: Runtime options.
        blob_ids: Model only these blobs, with only *ligand_records*
            (incremental re-modeling).  Default: every blob, with every ligand
            prepared under ``preprocess/ligands/``.

    Returns:
        The ``build_struct/`` directory.
    """
    layout = OutputLayout(Path(output_dir))
    configure_runtime()
    device = resolve_device(cfg.gpu)

    blob_paths = layout.blob_paths(blob_ids)
    if not blob_paths:
        raise Emap2ligError(f"No blob files found in {layout.find_blobs}")
    if blob_ids is not None:
        logger.info(f"Filtered to {len(blob_paths)} blobs (ids={blob_ids})")

    # Incremental runs use an isolated ligands directory so the dataset only
    # sees the newly requested ligands.
    incremental = blob_ids is not None
    ligands_dir = (
        layout.preprocess / f"_ligands_{uuid.uuid4().hex[:8]}"
        if incremental
        else layout.ligands
    )
    try:
        prepare_ligands(ligand_records, ligands_dir)
        if not any(ligands_dir.glob("*.npz")):
            raise Emap2ligError("No ligand could be prepared from the ligand list")

        # One batch per blob-ligand pair; the model chunks the multiplicity
        # internally so Lightning shows a single progress pass.
        dataset = LigandModelingDataset(
            density_object_list=blob_paths,
            ref_mol_dir=ligands_dir,
            multiplicity=cfg.multiplicity,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )

        logger.info("Initializing Emap2lig structure model")
        model = build_structure_model(num_sampling_steps=cfg.num_sampling_steps)
        model.predict_args.multiplicity = cfg.multiplicity
        model.predict_args.max_parallel_multiplicity = cfg.max_parallel_multiplicity

        layout.build_struct.mkdir(parents=True, exist_ok=True)
        writer = LigandWriter(output_dir=layout.build_struct, output_format="mmcif")
        trainer = Trainer(
            **trainer_device_kwargs(device), callbacks=[writer], logger=False
        )
        logger.info(
            f"Running inference on {len(dataset)} combinations with multiplicity "
            f"{cfg.multiplicity} (max parallel {cfg.max_parallel_multiplicity})"
        )
        # LigandWriter persists every batch; returning predictions would keep
        # ~27 MiB of voxel features per conformer on the host.
        trainer.predict(model, dataloaders=dataloader, return_predictions=False)
        logger.info("Inference completed.")
    finally:
        if incremental:
            shutil.rmtree(ligands_dir, ignore_errors=True)

    rank_conformers(layout.build_struct, layout.best)
    return layout.build_struct
