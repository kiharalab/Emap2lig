"""Thin service layer that wraps emap2lig core functions.

Heavy imports (torch, emap2lig.*) are done lazily inside each function
so the FastAPI module loads instantly.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from ..errors import JobCancelledError
from ..layout import OutputLayout
from ..progress import ProgressCallback, ProgressEvent
from ..utils.storage import get_home, get_model_dir, set_home
from .results_scan import resolve_blobs_dir, scan_blobs, scan_results
from .schemas import (
    JobResponse,
    JobSummaryResponse,
    LigandSpec,
    StageProgressResponse,
)
from .state import Job, JobStatus, StageProgress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured progress → Job.stages
# ---------------------------------------------------------------------------


def apply_progress(job: Job, event: ProgressEvent) -> None:
    """Record *event* on *job*, raising if cancel was requested.

    Args:
        job: In-memory job being updated.
        event: Pipeline progress event.

    Raises:
        JobCancelledError: ``job.cancel_event`` is set.
    """
    if job.cancel_event.is_set():
        raise JobCancelledError("Job cancelled")
    now = datetime.now(UTC)
    stage = next((s for s in job.stages if s.stage == event.stage), None)
    if stage is None:
        for previous in job.stages:
            if previous.status == "running":
                previous.status = "completed"
                previous.finished_at = now
                if previous.total is not None:
                    previous.current = previous.total
        stage = StageProgress(stage=event.stage, status="running", started_at=now)
        job.stages.append(stage)
    stage.current = event.current
    stage.total = event.total
    stage.failed = event.failed
    stage.message = event.message
    if event.total is not None and event.total > 0 and event.current >= event.total:
        stage.status = "completed"
        stage.finished_at = now
    job.progress = event.message


def job_progress_callback(job: Job) -> ProgressCallback:
    """Return a pipeline callback that writes structured progress onto *job*."""

    def _callback(event: ProgressEvent) -> None:
        apply_progress(job, event)

    return _callback


def job_to_response(job: Job, queue_position: int | None = None) -> JobResponse:
    """Serialize *job* to the public job payload."""
    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        created_at=job.created_at,
        progress=job.progress,
        stages=[
            StageProgressResponse(
                stage=s.stage,
                status=s.status,
                current=s.current,
                total=s.total,
                failed=s.failed,
                message=s.message,
                started_at=s.started_at,
                finished_at=s.finished_at,
            )
            for s in job.stages
        ],
        error=job.error,
        output_dir=str(job.output_dir),
        num_blobs=job.num_blobs,
        detect_job_id=job.detect_job_id,
        unified_map_url=(
            f"/api/files/{job.id}/{job.unified_map_path}"
            if job.unified_map_path
            else None
        ),
        queue_position=queue_position,
        cancel_requested=job.cancel_event.is_set(),
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def job_to_summary(job: Job, queue_position: int | None = None) -> JobSummaryResponse:
    """Serialize *job* to the list-view payload."""
    return JobSummaryResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        created_at=job.created_at,
        output_dir=str(job.output_dir),
        detect_job_id=job.detect_job_id,
        queue_position=queue_position,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOBS_ROOT = Path("jobs_output")


def _ensure_jobs_root() -> Path:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    return JOBS_ROOT


def write_ligand_yaml(specs: list[LigandSpec], output_dir: Path, job_id: str) -> Path:
    """Write frontend ligand specs as a CLI-compatible YAML file."""
    from emap2lig.data.ligand_list import write_ligand_list
    from emap2lig.data.types import LigandRecord

    preprocess_dir = OutputLayout(output_dir).preprocess
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    records: list[LigandRecord] = []
    for spec in specs:
        blobs = sorted({int(b) for b in spec.blob_id}) if spec.blob_id else None
        if spec.type == "CCD":
            name = (spec.name or "").strip().upper()
            if name:
                records.append(LigandRecord(type="CCD", name=name, blobs=blobs))
        elif spec.type == "SMILES":
            smiles = (spec.smiles or "").strip()
            if smiles:
                records.append(
                    LigandRecord(type="SMILES", name="", blobs=blobs, smiles=smiles)
                )
        else:
            residues = {
                int(rid): str(code).strip().upper()
                for rid, code in (spec.residues or {}).items()
                if str(code).strip()
            }
            if residues:
                records.append(
                    LigandRecord(
                        type="BRANCHED",
                        name="",
                        blobs=blobs,
                        residues=residues,
                        bonds=spec.bonds,
                    )
                )

    return write_ligand_list(records, preprocess_dir / f"ligand_list_{job_id}.yaml")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _job_config(job: Job):
    """Runtime options for *job*; unset fields keep the pipeline defaults."""
    from emap2lig.config import InferenceConfig

    options = {
        "gpu": job.gpu[0] if job.gpu else 0,  # one device per stage
        "contour_level": job.contour_level,
        "multiplicity": job.multiplicity,
        "max_parallel_multiplicity": job.max_parallel_multiplicity,
        "detection_batch_size": job.detection_batch_size,
        "num_sampling_steps": job.num_sampling_steps,
    }
    return InferenceConfig(**{k: v for k, v in options.items() if v is not None})


def _finish(
    job: Job, status: JobStatus, progress: str, error: str | None = None
) -> None:
    """Mark *job* terminal with *status*, *progress*, and optional *error*."""
    now = datetime.now(UTC)
    if status in {"cancelled", "failed"}:
        for stage in job.stages:
            if stage.status == "running":
                stage.status = "failed"
                stage.finished_at = now
    job.status = status
    job.progress = progress
    job.error = error
    job.finished_at = now


async def run_detection(job: Job) -> None:
    """Run Stage 1 detection in a background thread."""
    job.status = "running"
    if job.started_at is None:
        job.started_at = datetime.now(UTC)
    job.progress = "Loading configuration..."
    try:
        validate_gpu_selection(job.gpu)
        blobs_dir = await asyncio.to_thread(_detect_sync, job)

        blobs = scan_blobs(blobs_dir)
        job.blobs = blobs
        job.num_blobs = len(blobs)

        layout = OutputLayout(job.output_dir)
        if not layout.unified_map.exists():
            raise FileNotFoundError(
                "Detection completed but unified map was not found at "
                f"{layout.unified_map}"
            )
        job.unified_map_path = str(layout.unified_map.relative_to(job.output_dir))
        _finish(job, "completed", f"Detected {len(blobs)} blobs")
    except JobCancelledError:
        _finish(job, "cancelled", "Cancelled")
    except Exception as e:
        logger.exception("Detection failed")
        _finish(job, "failed", job.progress or "Detection failed", str(e))


def _detect_sync(job: Job) -> Path:
    from emap2lig.pipeline import detect_ligand_objects

    cfg = _job_config(job)
    job.progress = "Running blob detection..."
    return detect_ligand_objects(
        job.input_map_path,
        job.output_dir,
        cfg,
        job.emdb_id,
        on_progress=job_progress_callback(job),
    )


async def run_modeling(
    job: Job,
    ligand_specs: list[LigandSpec],
    blob_ids: list[int] | None = None,
) -> None:
    """Run Stage 2 modeling in a background thread.

    When *blob_ids* is provided, only those blobs are (re)modeled.
    """
    job.status = "running"
    if job.started_at is None:
        job.started_at = datetime.now(UTC)
    job.progress = (
        "Preparing incremental ligand YAML..."
        if blob_ids is not None
        else "Preparing ligand YAML..."
    )
    try:
        validate_gpu_selection(job.gpu)
        resolve_blobs_dir(job.output_dir)
        ligand_yaml_path = write_ligand_yaml(ligand_specs, job.output_dir, job.id)
        results_dir = await asyncio.to_thread(
            _model_sync, job, ligand_yaml_path, blob_ids
        )

        job.results = scan_results(results_dir)
        _finish(
            job,
            "completed",
            (
                "Incremental modeling complete"
                if blob_ids is not None
                else "Modeling complete"
            ),
        )
    except JobCancelledError:
        _finish(job, "cancelled", "Cancelled")
    except Exception as e:
        logger.exception("Modeling failed")
        _finish(job, "failed", job.progress or "Modeling failed", str(e))


def _model_sync(
    job: Job, ligand_list_path: Path, blob_ids: list[int] | None = None
) -> Path:
    from emap2lig.data.ligand_list import parse_ligand_list
    from emap2lig.pipeline import run_structure_modeling

    cfg = _job_config(job)
    job.progress = "Parsing ligand YAML..."
    ligand_records = parse_ligand_list(ligand_list_path)
    job.progress = "Running structure modeling..."
    return run_structure_modeling(
        job.output_dir,
        ligand_records,
        cfg,
        blob_ids=blob_ids,
        on_progress=job_progress_callback(job),
    )


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


async def download_models() -> dict[str, bool]:
    """Download model weights via huggingface_hub into the model directory. Returns cache status."""
    return await asyncio.to_thread(_download_models_sync)


def _download_models_sync() -> dict[str, bool]:
    from emap2lig.utils.weights import MODEL_FILES, fetch_weights

    results = {}
    for key, filename in MODEL_FILES.items():
        try:
            fetch_weights(filename)
            results[key] = True
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")
            results[key] = False
    return results


# ---------------------------------------------------------------------------
# Model download progress (background thread + pollable status)
# ---------------------------------------------------------------------------

_download_lock = threading.Lock()
_download_state: dict[str, str | bool | None] = {"active": False, "error": None}
_hf_file_sizes: dict[str, int | None] = {}


def _query_hf_sizes() -> dict[str, int | None]:
    """Best-effort per-file sizes from the HF Hub metadata (cached)."""
    if _hf_file_sizes:
        return _hf_file_sizes
    try:
        from huggingface_hub import HfApi

        from emap2lig.utils.weights import MODEL_FILES, WEIGHTS_REPO_ID

        info = HfApi().model_info(WEIGHTS_REPO_ID, files_metadata=True)
        by_name = {s.rfilename: (s.size or 0) for s in info.siblings}
        for fn in MODEL_FILES.values():
            _hf_file_sizes[fn] = by_name.get(fn)
    except Exception as e:
        logger.warning(f"Could not fetch HF file sizes: {e}")
        from emap2lig.utils.weights import MODEL_FILES

        for fn in MODEL_FILES.values():
            _hf_file_sizes[fn] = None
    return _hf_file_sizes


def _in_flight_bytes(models_dir: Path) -> int:
    """Bytes of the currently in-flight HF download.

    huggingface_hub 2.x names in-flight files ``<etag>.<random>.incomplete``
    under ``<models_dir>/.cache/huggingface/download/``. Downloads run
    serially, so at most one exists; the newest wins.
    """
    dl_dir = models_dir / ".cache" / "huggingface" / "download"
    try:
        newest: tuple[float, Path] | None = None
        for p in dl_dir.glob("*.incomplete"):
            mtime = p.stat().st_mtime
            if newest is None or mtime > newest[0]:
                newest = (mtime, p)
        return newest[1].stat().st_size if newest else 0
    except OSError:
        return 0


def start_model_download() -> bool:
    """Kick off a background download if one is not already running."""
    with _download_lock:
        if _download_state["active"]:
            return False
        _download_state["active"] = True
        _download_state["error"] = None

    def _run() -> None:
        try:
            results = _download_models_sync()
            if not all(results.values()):
                _download_state["error"] = (
                    "one or more downloads failed (see server log)"
                )
        except Exception as e:  # surface the failure in /download/status
            _download_state["error"] = str(e)
        finally:
            with _download_lock:
                _download_state["active"] = False

    threading.Thread(target=_run, daemon=True, name="model-download").start()
    return True


def get_download_status() -> dict:
    """Per-file download progress: cached flag, total size, downloaded bytes."""
    from emap2lig.utils.weights import MODEL_FILES, get_model_path

    sizes = _query_hf_sizes()
    files: list[dict] = []
    total = downloaded = 0
    in_flight_assigned = False
    for key, filename in MODEL_FILES.items():
        target = get_model_path(filename)
        cached = target.exists()
        size = sizes.get(filename)
        if cached and not size:
            size = target.stat().st_size
        if cached:
            done = size or 0
        elif not in_flight_assigned:
            # Serial downloads: the newest .incomplete belongs to the first
            # uncached file in order.
            done = _in_flight_bytes(get_model_dir())
            in_flight_assigned = True
        else:
            done = 0
        files.append(
            {
                "key": key,
                "filename": filename,
                "cached": cached,
                "size_bytes": size,
                "downloaded_bytes": done if done else 0,
            }
        )
        total += size or 0
        downloaded += done or 0
    return {
        "downloading": bool(_download_state["active"]),
        "error": _download_state["error"],
        "files": files,
        "total_bytes": total or None,
        "downloaded_bytes": downloaded or None,
    }


def check_model_cache() -> dict[str, bool]:
    """Check if models are already downloaded to the model directory."""
    try:
        from emap2lig.utils.weights import check_weights

        return check_weights()
    except Exception:
        return {"detection_model": False, "structure_model": False}


# ---------------------------------------------------------------------------
# Cache directory info
# ---------------------------------------------------------------------------


def get_cache_info() -> dict:
    """Return cache directory path and per-model file info.

    The "cache_dir" is the root directory (e.g. ``~/.emap2lig/``).
    Model weights live inside ``<cache_dir>/models/``.
    """
    try:
        from emap2lig.utils.weights import MODEL_FILES, get_model_path

        cache_dir = get_home()

        models = []
        for key, filename in MODEL_FILES.items():
            fpath = get_model_path(filename)
            exists = fpath.exists()
            size_mb = round(fpath.stat().st_size / (1024 * 1024), 1) if exists else None
            models.append(
                {
                    "key": key,
                    "filename": filename,
                    "path": str(fpath),
                    "exists": exists,
                    "size_mb": size_mb,
                }
            )
        return {"cache_dir": str(cache_dir), "models": models}
    except Exception as e:
        logger.warning(f"Failed to get cache info: {e}")
        return {"cache_dir": str(get_home()), "models": []}


def set_cache_dir(new_dir: str) -> dict:
    """Update the cache root directory (runtime only).

    Model weights will be stored under ``<new_dir>/models/``.
    """
    set_home(new_dir)
    get_model_dir().mkdir(parents=True, exist_ok=True)
    return get_cache_info()


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def list_gpus() -> dict:
    """Detect the platform-supported inference accelerator."""
    try:
        from emap2lig.utils.device import list_accelerators

        return list_accelerators()
    except Exception:
        return {
            "cuda_available": False,
            "mps_available": False,
            "accelerator": None,
            "gpus": [],
        }


def validate_gpu_selection(gpu_ids: list[int]) -> None:
    """Validate selected accelerator IDs for the current platform."""
    from emap2lig.utils.device import validate_device_ids

    validate_device_ids(gpu_ids)
