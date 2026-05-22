"""Stage 1 — detection endpoint."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile

from ..schemas import JobCreateResponse
from ..services import (
    JOBS_ROOT,
    _ensure_jobs_root,
    run_detection,
    validate_gpu_selection,
)
from ..state import create_job

router = APIRouter(prefix="/api", tags=["detect"])


def _map_name_stem(filename: str) -> str:
    """Return a safe stem for output directory naming."""
    stem = Path(filename).name
    lower = stem.lower()
    if lower.endswith(".gz"):
        stem = Path(stem).stem

    # Strip common cryo-EM map suffixes that may remain before .gz
    for ext in (".mrc", ".map", ".ccp4"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "map"


def _default_output_dir(root: Path, map_filename: str) -> Path:
    """Build an auto output dir path: jobs_output/{map_stem}_{timestamp}."""
    stem = _map_name_stem(map_filename)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = root / f"{stem}_{ts}"
    if not base.exists():
        return base

    # Rare collision fallback (same stem + second)
    suffix = 2
    while True:
        candidate = root / f"{stem}_{ts}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


@router.post("/detect", response_model=JobCreateResponse)
async def start_detection(
    background_tasks: BackgroundTasks,
    input_map: UploadFile | None = None,
    input_map_path: str | None = Form(None),
    gpu: str = Form("0"),
    detection_batch_size: int | None = Form(None),
    emdb_id: str | None = Form(None),
    contour_level: float | None = Form(None),
    output_dir: str | None = Form(None),
):
    """Upload a cryo-EM map and start blob detection (Stage 1).

    Accepts either a file upload (``input_map``) or a server-side path
    (``input_map_path``).  Exactly one must be provided.
    """
    _ensure_jobs_root()

    # Parse gpu: accepts a JSON array like "[0,1]" or a bare integer like "0"
    try:
        gpu_parsed = json.loads(gpu)
        gpu_list: list[int] = (
            [int(item) for item in gpu_parsed]
            if isinstance(gpu_parsed, list)
            else [int(gpu_parsed)]
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        gpu_list = [int(gpu)]

    try:
        validate_gpu_selection(gpu_list)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    emdb_clean = emdb_id.strip() if emdb_id else None
    if emdb_clean:
        emdb_clean = re.sub(r"^EMD[-_]?", "", emdb_clean, flags=re.IGNORECASE)
        if not emdb_clean.isdigit():
            raise HTTPException(
                400,
                "Invalid EMDB ID. Use numeric ID (e.g. 30556 or EMD-30556).",
            )

    if emdb_clean and contour_level is not None:
        raise HTTPException(
            400,
            "Provide either emdb_id or contour_level, not both.",
        )
    if not emdb_clean and contour_level is None:
        raise HTTPException(
            400,
            "Provide one normalization source: emdb_id or contour_level.",
        )

    server_path: Path | None = None
    if input_map_path and input_map_path.strip():
        server_path = Path(input_map_path.strip()).expanduser().resolve()
        if not server_path.exists():
            raise HTTPException(404, f"File not found: {server_path}")
        if not server_path.is_file():
            raise HTTPException(400, f"Not a file: {server_path}")
        map_filename = server_path.name
    elif input_map and input_map.filename:
        map_filename = Path(input_map.filename).name
    else:
        raise HTTPException(
            400, "Either input_map (file) or input_map_path must be provided"
        )

    # Use custom output dir if provided, otherwise default to stem+timestamp.
    if output_dir and output_dir.strip():
        job_dir = Path(output_dir.strip()).expanduser().resolve()
    else:
        job_dir = _default_output_dir(JOBS_ROOT, map_filename)

    if job_dir.exists() and not job_dir.is_dir():
        raise HTTPException(
            400, f"Output path exists and is not a directory: {job_dir}"
        )
    job_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input map
    input_map_resolved: Path
    if server_path is not None:
        # Server-side file — use directly
        input_map_resolved = server_path
    elif input_map and input_map.filename:
        # Client upload — save to job dir
        map_path = job_dir / map_filename
        with open(map_path, "wb") as f:
            shutil.copyfileobj(input_map.file, f)
        input_map_resolved = map_path
    else:
        raise HTTPException(
            400, "Either input_map (file) or input_map_path must be provided"
        )

    job = create_job(
        job_type="detect",
        output_dir=job_dir,
        input_map_path=input_map_resolved,
        gpu=gpu_list,
        detection_batch_size=detection_batch_size,
        emdb_id=emdb_clean,
        contour_level=contour_level,
    )

    background_tasks.add_task(run_detection, job)
    return JobCreateResponse(job_id=job.id)
