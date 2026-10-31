"""Job status, blob listing, and results endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ...layout import OutputLayout
from ..results_scan import scan_blobs, scan_results
from ..schemas import (
    BlobInfoResponse,
    BlobListResponse,
    BlobResultResponse,
    ConformerResponse,
    JobResponse,
    JobSummaryResponse,
    ResultsResponse,
)
from ..services import job_to_response, job_to_summary
from ..state import Job, _jobs, get_job, list_jobs

router = APIRouter(prefix="/api", tags=["jobs"])


def _queue_position(request: Request, job_id: str) -> int | None:
    """Return the 1-based queue position of *job_id*, or ``None``."""
    return request.app.state.runner.queue_position(job_id)


@router.get("/jobs", response_model=list[JobSummaryResponse])
async def list_all_jobs(request: Request):
    """Return every in-memory job (newest first)."""
    jobs = sorted(list_jobs(), key=lambda j: j.created_at, reverse=True)
    return [job_to_summary(j, _queue_position(request, j.id)) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, request: Request):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job_to_response(job, _queue_position(request, job.id))


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, request: Request):
    """Cancel a queued job immediately, or a running job at the next progress event."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if not request.app.state.runner.cancel(job):
        raise HTTPException(
            400,
            f"Job {job_id} cannot be cancelled (status={job.status})",
        )
    return job_to_response(job, _queue_position(request, job.id))


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

# Repository checkout only: examples/ is not shipped in the wheel.
_TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "examples" / "tutorial_output"


@router.post("/tutorial/create-jobs")
async def create_tutorial_jobs():
    """Create virtual completed detect + model jobs from the tutorial output."""
    if not _TUTORIAL_DIR.exists():
        raise HTTPException(404, f"Tutorial output not found at {_TUTORIAL_DIR}")

    layout = OutputLayout(_TUTORIAL_DIR)
    blobs_dir = layout.find_blobs
    blobs = scan_blobs(blobs_dir) if blobs_dir.is_dir() else []

    results_dir = layout.build_struct
    results = scan_results(results_dir) if results_dir.is_dir() else []

    unified = layout.unified_map
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
