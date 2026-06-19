"""Stage 2 — structure modeling endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..schemas import JobCreateResponse, LigandSpec
from ..services import (
    _ensure_jobs_root,
    run_modeling,
    validate_gpu_selection,
)
from ..state import create_job, get_job

router = APIRouter(prefix="/api", tags=["model"])


class ModelRequest(BaseModel):
    job_id: str  # detect job to build on
    ligand_list: list[LigandSpec]
    gpu: list[int] | int = 0
    multiplicity: int = 1


class BlobModelRequest(BaseModel):
    detect_job_id: str | None = None
    output_dir: str | None = None
    blob_ids: list[int]
    ligand_list: list[LigandSpec]
    gpu: list[int] | int = 0
    multiplicity: int = 1


@router.post("/model", response_model=JobCreateResponse)
async def start_modeling(
    body: ModelRequest,
    background_tasks: BackgroundTasks,
):
    """Start structure modeling on previously detected blobs (Stage 2)."""
    detect_job = get_job(body.job_id)
    if detect_job is None:
        raise HTTPException(404, f"Job {body.job_id} not found")
    if detect_job.status != "completed":
        raise HTTPException(
            400,
            f"Detection job {body.job_id} is not completed (status={detect_job.status})",
        )

    _ensure_jobs_root()

    # Normalize gpu to list
    gpu_list = body.gpu if isinstance(body.gpu, list) else [body.gpu]
    try:
        validate_gpu_selection(gpu_list)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Create new modeling job, sharing the same output dir
    job = create_job(
        job_type="model",
        output_dir=detect_job.output_dir,
        gpu=gpu_list,
        multiplicity=body.multiplicity,
        detect_job_id=detect_job.id,
    )
    # Copy blob info from detect job
    job.blobs = detect_job.blobs
    job.num_blobs = detect_job.num_blobs

    background_tasks.add_task(run_modeling, job, body.ligand_list)
    return JobCreateResponse(job_id=job.id)


@router.post("/model-blob", response_model=JobCreateResponse)
async def start_blob_modeling(
    body: BlobModelRequest,
    background_tasks: BackgroundTasks,
):
    """Run incremental structure modeling on specific blobs with new ligands.

    Accepts *either* ``detect_job_id`` (from Build tab) or ``output_dir``
    (from Visualization tab) to locate the working directory.
    """
    if not body.blob_ids:
        raise HTTPException(400, "blob_ids must contain at least one blob ID")

    # Resolve output directory from either source
    if body.detect_job_id:
        detect_job = get_job(body.detect_job_id)
        if detect_job is None:
            raise HTTPException(404, f"Detection job {body.detect_job_id} not found")
        if detect_job.status != "completed":
            raise HTTPException(
                400,
                f"Detection job {body.detect_job_id} is not completed "
                f"(status={detect_job.status})",
            )
        out_dir = detect_job.output_dir
        parent_blobs = detect_job.blobs
        parent_num = detect_job.num_blobs
        parent_id = detect_job.id
    elif body.output_dir:
        from pathlib import Path

        out_dir = Path(body.output_dir).expanduser().resolve()
        if not out_dir.is_dir():
            raise HTTPException(404, f"Directory not found: {body.output_dir}")
        parent_blobs = None
        parent_num = None
        parent_id = None
    else:
        raise HTTPException(400, "Either detect_job_id or output_dir must be provided")

    _ensure_jobs_root()

    gpu_list = body.gpu if isinstance(body.gpu, list) else [body.gpu]
    try:
        validate_gpu_selection(gpu_list)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = create_job(
        job_type="model",
        output_dir=out_dir,
        gpu=gpu_list,
        multiplicity=body.multiplicity,
        detect_job_id=parent_id,
    )
    job.blobs = parent_blobs
    job.num_blobs = parent_num

    background_tasks.add_task(run_modeling, job, body.ligand_list, body.blob_ids)
    return JobCreateResponse(job_id=job.id)
