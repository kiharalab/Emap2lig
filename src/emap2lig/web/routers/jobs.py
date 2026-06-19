"""Job status, blob listing, and results endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..results_scan import (
    BUILD_STRUCT_DIR,
    FIND_BLOBS_DIR,
    scan_blobs,
    scan_results,
    unified_map_path,
)
from ..schemas import (
    BlobInfoResponse,
    BlobListResponse,
    BlobResultResponse,
    ConformerResponse,
    JobResponse,
    ResultsResponse,
)
from ..state import Job, _jobs, get_job

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        created_at=job.created_at,
        progress=job.progress,
        error=job.error,
        output_dir=str(job.output_dir),
        num_blobs=job.num_blobs,
        detect_job_id=job.detect_job_id,
        unified_map_url=(
            f"/api/files/{job.id}/{job.unified_map_path}"
            if job.unified_map_path
            else None
        ),
    )


@router.get("/jobs/{job_id}/blobs", response_model=BlobListResponse)
async def get_job_blobs(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.blobs is None:
        raise HTTPException(400, "No blobs available (detection may not be complete)")
    return BlobListResponse(
        blobs=[
            BlobInfoResponse(
                id=b.id,
                num_voxels=b.num_voxels,
                mask_url=f"/api/files/{job_id}/{b.mask_path}",
                density_url=f"/api/files/{job_id}/{b.density_path}",
            )
            for b in job.blobs
        ]
    )


@router.get("/jobs/{job_id}/results", response_model=ResultsResponse)
async def get_job_results(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.results is None:
        raise HTTPException(400, "No results available (modeling may not be complete)")
    return ResultsResponse(
        results=[
            BlobResultResponse(
                blob_id=br.blob_id,
                conformers=[
                    ConformerResponse(
                        name=c.name,
                        score=c.score,
                        cif_url=f"/api/files/{job_id}/{c.cif_path}",
                        mask_url=(
                            f"/api/files/{job_id}/{c.mask_path}"
                            if c.mask_path
                            else None
                        ),
                    )
                    for c in br.conformers
                ],
            )
            for br in job.results
        ]
    )


# ---------------------------------------------------------------------------
# Tutorial: create virtual completed jobs from pre-computed output
# ---------------------------------------------------------------------------

_TUTORIAL_DIR = Path(__file__).resolve().parents[3] / "examples" / "tutorial_output"


@router.post("/tutorial/create-jobs")
async def create_tutorial_jobs():
    """Create virtual completed detect + model jobs from the tutorial output."""
    from datetime import datetime, UTC

    if not _TUTORIAL_DIR.exists():
        raise HTTPException(404, f"Tutorial output not found at {_TUTORIAL_DIR}")

    blobs_dir = _TUTORIAL_DIR / FIND_BLOBS_DIR
    blobs = scan_blobs(blobs_dir) if blobs_dir.is_dir() else []

    results_dir = _TUTORIAL_DIR / BUILD_STRUCT_DIR
    results = scan_results(results_dir) if results_dir.is_dir() else []

    unified = unified_map_path(_TUTORIAL_DIR)
    unified_rel = str(unified.relative_to(_TUTORIAL_DIR)) if unified.exists() else None

    now = datetime.now(UTC)

    detect_job = Job(
        id="tutorial_detect",
        type="detect",
        status="completed",
        created_at=now,
        output_dir=_TUTORIAL_DIR,
        num_blobs=len(blobs),
        blobs=blobs,
        unified_map_path=unified_rel,
    )
    _jobs[detect_job.id] = detect_job

    model_job = Job(
        id="tutorial_model",
        type="model",
        status="completed",
        created_at=now,
        output_dir=_TUTORIAL_DIR,
        results=results,
        detect_job_id=detect_job.id,
    )
    _jobs[model_job.id] = model_job

    return {
        "detect_job_id": detect_job.id,
        "model_job_id": model_job.id,
        "num_blobs": len(blobs),
        "working_dir": str(_TUTORIAL_DIR),
    }
