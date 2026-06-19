"""Filesystem scanners for Emap2lig output directories.

This module centralizes knowledge of the on-disk output layout produced by the
pipeline so both live jobs and "load existing results" flows can reuse it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .state import BlobInfo, BlobResult, ConformerResult

PREPROCESS_DIR = "preprocess"
FIND_BLOBS_DIR = "find_blobs"
BUILD_STRUCT_DIR = "build_struct"

UNIFIED_MAP_NAME = "unified.mrc"


def unified_map_path(output_dir: Path) -> Path:
    return output_dir / PREPROCESS_DIR / UNIFIED_MAP_NAME


def resolve_blobs_dir(output_dir: Path) -> Path:
    candidate = output_dir / FIND_BLOBS_DIR
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find blobs directory: {candidate}")


def resolve_results_dir(output_dir: Path) -> Path:
    candidate = output_dir / BUILD_STRUCT_DIR
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find modeling output directory: {candidate}")


def scan_blobs(blobs_dir: Path) -> list[BlobInfo]:
    """Scan a ``find_blobs/`` directory and return blob metadata.

    Paths in the returned objects are relative to the job output_dir.
    """
    blobs_dir = Path(blobs_dir)
    blob_dir_name = blobs_dir.name
    infos: list[BlobInfo] = []
    for npz_path in sorted(blobs_dir.glob("blob_*.npz")):
        try:
            blob_id = int(npz_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        num_voxels = 0
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            instance_grid = data.get("instance_grid", None)
            num_voxels = int(instance_grid.sum()) if instance_grid is not None else 0
        except Exception:
            num_voxels = 0

        infos.append(
            BlobInfo(
                id=blob_id,
                num_voxels=num_voxels,
                mask_path=f"{blob_dir_name}/mask_{blob_id}.mrc",
                density_path=f"{blob_dir_name}/blob_{blob_id}.npz",
            )
        )
    return infos


def scan_results(results_dir: Path) -> list[BlobResult]:
    """Scan a ``build_struct/`` directory and return modeling results.

    Paths in the returned objects are relative to the job output_dir.
    """
    results_dir = Path(results_dir)
    results: list[BlobResult] = []
    if not results_dir.exists():
        return results

    base_dir_name = results_dir.name
    for blob_dir in sorted(results_dir.iterdir()):
        if not blob_dir.is_dir() or not blob_dir.name.startswith("blob_"):
            continue

        try:
            blob_id = int(blob_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        csv_path = blob_dir / f"{blob_dir.name}_results.csv"
        conformers: list[ConformerResult] = []
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("conformer_name")
                    if not name:
                        continue
                    try:
                        score = float(row["consistency_iou"])
                    except Exception:
                        score = float("nan")

                    cif_rel = f"{base_dir_name}/{blob_dir.name}/{name}.cif"
                    mask_rel = f"{base_dir_name}/{blob_dir.name}/{name}_pred_mask.mrc"
                    mask_exists = (blob_dir / f"{name}_pred_mask.mrc").exists()

                    conformers.append(
                        ConformerResult(
                            name=name,
                            score=score,
                            cif_path=cif_rel,
                            mask_path=mask_rel if mask_exists else None,
                        )
                    )

        results.append(BlobResult(blob_id=blob_id, conformers=conformers))

    return results
