"""In-memory job store.

Jobs are stored in a plain dict keyed by UUID. Server restart clears all state.
This is intentional — the app targets single-user / small-lab use.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Blob metadata (populated after Stage 1)
# ---------------------------------------------------------------------------


@dataclass
class BlobInfo:
    id: int
    num_voxels: int
    mask_path: str  # relative to job output_dir, e.g. "find_blobs/mask_1.mrc"
    density_path: str  # e.g. "find_blobs/blob_1.npz"


# ---------------------------------------------------------------------------
# Conformer result (populated after Stage 2)
# ---------------------------------------------------------------------------


@dataclass
class ConformerResult:
    name: str
    score: float
    cif_path: str  # relative to job output_dir
    mask_path: str | None = None


@dataclass
class BlobResult:
    blob_id: int
    conformers: list[ConformerResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

JobType = Literal["detect", "model"]
JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class Job:
    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    output_dir: Path
    progress: str = ""
    error: str | None = None

    # Stage 1 outputs
    num_blobs: int | None = None
    blobs: list[BlobInfo] | None = None
    unified_map_path: str | None = None  # relative to output_dir

    # Stage 2 outputs
    results: list[BlobResult] | None = None

    # Link from modeling jobs to the parent detection job
    detect_job_id: str | None = None

    # Input metadata (stored so services can resume)
    input_map_path: Path | None = None
    emdb_id: str | None = None
    contour_level: float | None = None
    gpu: list[int] = field(default_factory=lambda: [0])
    detection_batch_size: int | None = None
    multiplicity: int = 1
    max_parallel_multiplicity: int = 8


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}


def create_job(
    job_type: JobType,
    output_dir: Path,
    **kwargs,
) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12],
        type=job_type,
        status="pending",
        created_at=datetime.now(UTC),
        output_dir=output_dir,
        **kwargs,
    )
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    return list(_jobs.values())
