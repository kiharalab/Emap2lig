"""Build stage: diffusion-based ligand structure modeling on detected blobs."""

import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import InferenceConfig
from ..data.dataset import LigandModelingDataset, collate_fn
from ..data.featurize import prepare_ligands
from ..data.io.writer import LigandWriter
from ..data.types import LigandRecord
from ..errors import Emap2ligError
from ..layout import OutputLayout
from ..model.model import Emap2lig
from ..model.pretrained import build_structure_model
from ..progress import ProgressCallback, emit
from ..utils.device import resolve_device
from ..utils.runtime import configure_runtime
from .ranking import rank_conformers

logger = logging.getLogger(__name__)


def run_structure_modeling(
    output_dir: Path | str,
    ligand_records: list[LigandRecord],
    cfg: InferenceConfig,
    blob_ids: list[int] | None = None,
    on_progress: ProgressCallback | None = None,
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
        on_progress: Optional structured-progress callback.

    Returns:
        The ``build_struct/`` directory.

    Raises:
        emap2lig.errors.JobCancelledError: Propagated from *on_progress*.
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
        emit(on_progress, "prepare_ligands", 0, 1, "Preparing ligand features")
        prepare_ligands(ligand_records, ligands_dir)
        if not any(ligands_dir.glob("*.npz")):
            raise Emap2ligError("No ligand could be prepared from the ligand list")
        emit(on_progress, "prepare_ligands", 1, 1, "Ligand features ready")

        # One batch per blob-ligand pair; the model chunks the multiplicity
        # internally so there is a single progress pass.
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
        emit(on_progress, "load_model", 0, 1, "Loading structure model")
        model = build_structure_model(num_sampling_steps=cfg.num_sampling_steps)
        model.predict_args.multiplicity = cfg.multiplicity
        model.predict_args.max_parallel_multiplicity = cfg.max_parallel_multiplicity
        emit(on_progress, "load_model", 1, 1, "Structure model ready")

        layout.build_struct.mkdir(parents=True, exist_ok=True)
        writer = LigandWriter(output_dir=layout.build_struct, output_format="mmcif")
        logger.info(
            f"Running inference on {len(dataset)} combinations with multiplicity "
            f"{cfg.multiplicity} (max parallel {cfg.max_parallel_multiplicity})"
        )
        _predict(model, dataloader, writer, device, on_progress)
        logger.info("Inference completed.")
    finally:
        if incremental:
            shutil.rmtree(ligands_dir, ignore_errors=True)

    emit(on_progress, "rank", 0, 1, "Ranking conformers")
    rank_conformers(layout.build_struct, layout.best)
    emit(on_progress, "rank", 1, 1, "Ranking complete")
    return layout.build_struct


def _batch_label(batch: Any) -> str:
    """Human-readable ligand / blob label from a collated batch."""
    identifiers = batch.get("identifier", batch.get("class_name", ["unknown"]))
    label = identifiers[0] if isinstance(identifiers, list) else identifiers
    return str(label)


def _steps_per_batch(model: Emap2lig) -> int:
    """Diffusion steps emitted per blob-ligand combination.

    Args:
        model: Structure model. Tests may omit ``predict_args``.

    Returns:
        ``ceil(multiplicity / max_parallel) * num_sampling_steps``, or 1 when
        ``predict_args`` is missing.
    """
    args = getattr(model, "predict_args", None)
    if args is None:
        return 1
    multiplicity = max(int(args.multiplicity), 1)
    max_parallel = max(int(args.max_parallel_multiplicity), 1)
    n_steps = max(int(args.num_sampling_steps), 1)
    return math.ceil(multiplicity / max_parallel) * n_steps


def _predict(
    model: Emap2lig,
    dataloader: DataLoader,
    writer: LigandWriter,
    device: torch.device,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Run *model* over *dataloader*, handing each batch to *writer*."""
    # Move the weights outside inference mode: inference-tensor parameters
    # change what the compiled submodules generate, and with it the outputs.
    model.to(device).eval()
    num_batches = len(dataloader)
    steps_per_batch = _steps_per_batch(model)
    total = num_batches * steps_per_batch
    emit(
        on_progress,
        "predict",
        0,
        total,
        f"Running inference on {num_batches} combinations",
    )
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)
            label = _batch_label(batch)

            def _on_step(
                local_step: int, idx: int = batch_idx, msg: str = label
            ) -> None:
                emit(
                    on_progress,
                    "predict",
                    idx * steps_per_batch + local_step,
                    total,
                    f"Predicting {idx + 1}/{num_batches} — {msg}",
                    failed=writer.failed,
                )

            prediction = model.predict_step(
                batch, batch_idx, num_batches, on_step=_on_step
            )
            writer.write(prediction, batch, batch_idx)
            emit(
                on_progress,
                "predict",
                (batch_idx + 1) * steps_per_batch,
                total,
                f"Predicting {batch_idx + 1}/{num_batches} — {label}",
                failed=writer.failed,
            )
    writer.finish()


def _to_device(data: Any, device: torch.device) -> Any:
    """Move the tensors in a (nested) batch to *device*."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _to_device(value, device) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return type(data)(_to_device(value, device) for value in data)
    return data
