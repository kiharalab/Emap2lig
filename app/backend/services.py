"""Thin service layer that wraps emap2lig core functions.

Heavy imports (torch, emap2lig.*) are done lazily inside each function
so the FastAPI module loads instantly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

from .results_scan import (
    PREPROCESS_DIR,
    resolve_blobs_dir,
    resolve_results_dir,
    scan_blobs,
    scan_results,
    unified_map_path,
)
from .schemas import LigandSpec
from .state import Job

from loguru import logger


# ---------------------------------------------------------------------------
# Progress capturing helpers
# ---------------------------------------------------------------------------


def _make_job_sink(job: Job):
    """Create a loguru sink that forwards log messages to ``job.progress``."""

    def _sink(message):
        line = message.record["message"].strip()
        if line:
            job.progress = line

    return _sink


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
        self._sink_id: int | None = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._sink_id = logger.add(
            _make_job_sink(self.job),
            filter=lambda record: record["name"].startswith("emap2lig"),
            level="INFO",
        )
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._old_stdout, self.job)  # type: ignore[assignment]
        sys.stderr = _TeeWriter(self._old_stderr, self.job)  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old_stdout  # type: ignore[assignment]
        sys.stderr = self._old_stderr  # type: ignore[assignment]
        if self._sink_id is not None:
            logger.remove(self._sink_id)
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
    preprocess_dir = output_dir / PREPROCESS_DIR
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = preprocess_dir / f"ligand_list_{job_id}.yaml"

    items: list[dict] = []
    for spec in specs:
        if spec.type == "CCD":
            name = (spec.name or "").strip().upper()
            if not name:
                continue
            item: dict = {"CCD": name}
        elif spec.type == "SMILES":
            smiles = (spec.smiles or "").strip()
            if not smiles:
                continue
            item = {"SMILES": smiles}
        else:
            residues = spec.residues or {}
            if not residues:
                continue
            residue_lines = [
                f"{rid}. {str(code).strip().upper()}"
                for rid, code in sorted(
                    ((int(k), v) for k, v in residues.items()),
                    key=lambda kv: kv[0],
                )
                if str(code).strip()
            ]
            if not residue_lines:
                continue
            branched: dict = {"residues": residue_lines}
            if spec.bonds:
                branched["bonds"] = spec.bonds
            item = {"BRANCHED": branched}

        if spec.blob_id:
            blob_ids = sorted({int(b) for b in spec.blob_id})
            if blob_ids:
                item["blob_id"] = blob_ids

        items.append(item)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(items, f, sort_keys=False)

    return yaml_path


# ---------------------------------------------------------------------------
# Stage 1 — Detection
# ---------------------------------------------------------------------------


async def run_detection(job: Job) -> None:
    """Run Stage 1 detection in a background thread."""
    job.status = "running"
    job.progress = "Loading configuration..."
    try:
        validate_gpu_selection(job.gpu)
        # Use first GPU for detection (single-device operation)
        primary_gpu = job.gpu[0] if job.gpu else 0
        status, blobs_dir = await asyncio.to_thread(
            _detect_sync,
            job=job,
            input_map=str(job.input_map_path),
            output_dir=str(job.output_dir),
            gpu=primary_gpu,
            detection_batch_size=job.detection_batch_size,
            contour_level=job.contour_level,
            emdb_id=job.emdb_id,
        )
        if status != 0:
            job.status = "failed"
            job.error = "Detection returned non-zero status"
            return

        # Populate blob info
        blobs = scan_blobs(blobs_dir) if blobs_dir is not None else []
        job.blobs = blobs
        job.num_blobs = len(blobs)

        # Find unified map
        unified_map = unified_map_path(job.output_dir)
        if not unified_map.exists():
            raise FileNotFoundError(
                f"Detection completed but unified map was not found at {unified_map}"
            )
        job.unified_map_path = str(unified_map.relative_to(job.output_dir))

        job.status = "completed"
        job.progress = f"Detected {len(blobs)} blobs"
    except Exception as e:
        logger.exception("Detection failed")
        job.status = "failed"
        job.error = str(e)


def _detect_sync(
    *,
    job: Job,
    input_map: str,
    output_dir: str,
    gpu: int,
    detection_batch_size: int | None,
    contour_level: float | None,
    emdb_id: str | None,
) -> tuple[int, Path | None]:
    from emap2lig.main import detect_ligand_objects, load_config

    with _ProgressCapture(job):
        job.progress = "Loading model configuration..."
        cfg = load_config(
            gpu=gpu,
            detection_batch_size=detection_batch_size,
            contour_level=contour_level,
        )
        job.progress = "Running blob detection..."
        return detect_ligand_objects(input_map, output_dir, cfg, emdb_id)


# ---------------------------------------------------------------------------
# Stage 2 — Structure Modeling
# ---------------------------------------------------------------------------


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
        blobs_dir = resolve_blobs_dir(job.output_dir)
        ligand_yaml_path = write_ligand_yaml(ligand_specs, job.output_dir, job.id)

        # Use first GPU for load_config, pass full list for Trainer devices
        primary_gpu = job.gpu[0] if job.gpu else 0
        status = await asyncio.to_thread(
            _model_sync,
            job=job,
            blobs_dir=str(blobs_dir),
            output_dir=str(job.output_dir),
            ligand_list_path=str(ligand_yaml_path),
            gpu=primary_gpu,
            multiplicity=job.multiplicity,
            blob_ids=blob_ids,
        )
        if status != 0:
            job.status = "failed"
            job.error = "Modeling returned non-zero status"
            return

        # Populate results
        job.results = scan_results(resolve_results_dir(job.output_dir))
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
    *,
    job: Job,
    blobs_dir: str,
    output_dir: str,
    ligand_list_path: str,
    gpu: int,
    multiplicity: int,
    blob_ids: list[int] | None = None,
) -> int:
    from emap2lig.main import load_config, parse_ligand_list, run_structure_modeling

    with _ProgressCapture(job):
        job.progress = "Loading model configuration..."
        cfg = load_config(gpu=gpu)
        job.progress = "Parsing ligand YAML..."
        ligand_records = parse_ligand_list(Path(ligand_list_path))
        job.progress = "Running structure modeling..."
        return run_structure_modeling(
            blobs_dir=blobs_dir,
            output_dir=output_dir,
            ligand_records=ligand_records,
            cfg=cfg,
            gpu=gpu,
            multiplicity=multiplicity,
            blob_ids=blob_ids,
        )


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


async def download_models() -> dict[str, bool]:
    """Download model weights via huggingface_hub to ~/.emap2lig/models/. Returns cache status."""
    return await asyncio.to_thread(_download_models_sync)


def _download_models_sync() -> dict[str, bool]:
    from emap2lig.main import MODEL_FILES, download_weights

    results = {}
    for key, filename in MODEL_FILES.items():
        try:
            download_weights(filename=filename)
            results[key] = True
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")
            results[key] = False
    return results


def check_model_cache() -> dict[str, bool]:
    """Check if models are already downloaded to ~/.emap2lig/models/."""
    try:
        from emap2lig.main import check_weights

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
        from emap2lig.main import MODEL_DIR, MODEL_FILES, get_model_path

        # MODEL_DIR is ``~/.emap2lig/models`` — expose its parent as the cache root
        cache_dir = MODEL_DIR.parent

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
        return {"cache_dir": str(Path.home() / ".emap2lig"), "models": []}


def set_cache_dir(new_dir: str) -> dict:
    """Update the cache root directory (runtime only).

    Model weights will be stored under ``<new_dir>/models/``.
    """
    import emap2lig.main as emap_main

    cache_root = Path(new_dir).expanduser().resolve()
    model_dir = cache_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    emap_main.MODEL_DIR = model_dir
    return get_cache_info()


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def list_gpus() -> dict:
    """Detect available CUDA GPUs."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False, "gpus": []}

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "id": i,
                    "name": props.name,
                    "memory_mb": props.total_memory // (1024 * 1024),
                }
            )
        return {"cuda_available": True, "gpus": gpus}
    except Exception:
        return {"cuda_available": False, "gpus": []}


def validate_gpu_selection(gpu_ids: list[int]) -> None:
    """Validate that selected GPU IDs are usable CUDA devices."""
    gpu_info = list_gpus()
    if not gpu_info["cuda_available"] or not gpu_info["gpus"]:
        raise ValueError(
            "CUDA GPU is required for inference. CPU mode is not supported."
        )

    if not gpu_ids:
        raise ValueError(
            "No GPU selected. Please select at least one CUDA GPU in Setup."
        )

    available_ids = {gpu["id"] for gpu in gpu_info["gpus"]}
    invalid_ids = sorted({gpu_id for gpu_id in gpu_ids if gpu_id not in available_ids})
    if invalid_ids:
        available_str = ", ".join(str(i) for i in sorted(available_ids))
        invalid_str = ", ".join(str(i) for i in invalid_ids)
        raise ValueError(
            f"Invalid GPU id(s): {invalid_str}. Available CUDA GPU id(s): {available_str}."
        )
