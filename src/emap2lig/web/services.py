"""Thin service layer that wraps emap2lig core functions.

Heavy imports (torch, emap2lig.*) are done lazily inside each function
so the FastAPI module loads instantly.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from ..utils.storage import get_home, get_model_dir, set_home
from ..layout import OutputLayout
from .results_scan import resolve_blobs_dir, scan_blobs, scan_results
from .schemas import LigandSpec
from .state import Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress capturing helpers
# ---------------------------------------------------------------------------


class _JobLogHandler(logging.Handler):
    """Logging handler that forwards ``emap2lig`` log messages to ``job.progress``."""

    def __init__(self, job: Job):
        super().__init__(level=logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        line = record.getMessage().strip()
        if line:
            self.job.progress = line


class _TeeWriter:
    """Stream wrapper that writes to the original stream *and* updates ``job.progress``.

    Handles both newline-delimited output and carriage-return progress bars
    (e.g. ``tqdm``).

    .. warning:: Replacing ``sys.stdout`` is *not* thread-safe.  This is
       acceptable for the single-user / small-lab use case of the app.
    """

    def __init__(self, original, job: Job):
        self.original = original
        self.job = job
        self._buf = ""

    def write(self, text: str) -> int:
        self.original.write(text)
        self._buf += text
        # Process complete lines
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._update(line)
        # Handle carriage-return overwrites (tqdm, etc.)
        while "\r" in self._buf:
            line, self._buf = self._buf.split("\r", 1)
            self._update(line)
        return len(text)

    def _update(self, line: str) -> None:
        line = line.strip()
        if line:
            self.job.progress = line

    def flush(self) -> None:
        if hasattr(self.original, "flush"):
            self.original.flush()

    # Delegate everything else (fileno, isatty, …) to the real stream.
    def __getattr__(self, name: str):
        return getattr(self.original, name)


class _ProgressCapture:
    """Context manager: captures logging + stdout/stderr, piping into ``job.progress``.

    Usage::

        with _ProgressCapture(job):
            run_heavy_work()
    """

    def __init__(self, job: Job):
        self.job = job
        self._handler = _JobLogHandler(job)
        self._pkg_logger = logging.getLogger("emap2lig")
        self._old_level = self._pkg_logger.level
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        # Make sure INFO records are emitted even if logging is unconfigured.
        if self._pkg_logger.getEffectiveLevel() > logging.INFO:
            self._pkg_logger.setLevel(logging.INFO)
        self._pkg_logger.addHandler(self._handler)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._old_stdout, self.job)  # type: ignore[assignment]
        sys.stderr = _TeeWriter(self._old_stderr, self.job)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old_stdout  # type: ignore[assignment]
        sys.stderr = self._old_stderr  # type: ignore[assignment]
        self._pkg_logger.removeHandler(self._handler)
        self._pkg_logger.setLevel(self._old_level)
        return False


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


async def run_detection(job: Job) -> None:
    """Run Stage 1 detection in a background thread."""
    job.status = "running"
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

        job.status = "completed"
        job.progress = f"Detected {len(blobs)} blobs"
    except Exception as e:
        logger.exception("Detection failed")
        job.status = "failed"
        job.error = str(e)


def _detect_sync(job: Job) -> Path:
    from emap2lig.pipeline import detect_ligand_objects

    with _ProgressCapture(job):
        cfg = _job_config(job)
        job.progress = "Running blob detection..."
        return detect_ligand_objects(
            job.input_map_path, job.output_dir, cfg, job.emdb_id
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
        job.status = "completed"
        job.progress = (
            "Incremental modeling complete"
            if blob_ids is not None
            else "Modeling complete"
        )
    except Exception as e:
        logger.exception("Modeling failed")
        job.status = "failed"
        job.error = str(e)


def _model_sync(
    job: Job, ligand_list_path: Path, blob_ids: list[int] | None = None
) -> Path:
    from emap2lig.data.ligand_list import parse_ligand_list
    from emap2lig.pipeline import run_structure_modeling

    with _ProgressCapture(job):
        cfg = _job_config(job)
        job.progress = "Parsing ligand YAML..."
        ligand_records = parse_ligand_list(ligand_list_path)
        job.progress = "Running structure modeling..."
        return run_structure_modeling(
            job.output_dir, ligand_records, cfg, blob_ids=blob_ids
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
