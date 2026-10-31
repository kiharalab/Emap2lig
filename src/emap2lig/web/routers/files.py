"""Static file serving for job outputs (MRC, CIF, NPZ, CSV) + directory browsing."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import tarfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from ...layout import OutputLayout
from ..results_scan import scan_blobs, scan_results
from ..services import JOBS_ROOT
from ..state import get_job

router = APIRouter(prefix="/api", tags=["files"])

# Files above this size are served from a cached .gz sibling (built once) with
# a truthful Content-Encoding header; the browser transparently decompresses.
_GZIPPABLE_MIN_BYTES = 1_000_000
_GZIPPABLE_SUFFIXES = {".mrc", ".map", ".npz", ".cif", ".csv"}


def _gzip_cached(target: Path) -> tuple[Path, int] | None:
    """Return ``(gz_path, original_size)`` for a large compressible file.

    The .gz sibling is cached under the system temp directory (keyed by path
    and mtime) so user output directories are not polluted. Returns ``None``
    for small files, already-compressed files, or when the cache cannot be
    written.
    """
    try:
        original_size = target.stat().st_size
        if (
            target.suffix.lower() == ".gz"
            or original_size < _GZIPPABLE_MIN_BYTES
            or target.suffix.lower() not in _GZIPPABLE_SUFFIXES
        ):
            return None
        import hashlib
        import tempfile

        cache_root = Path(tempfile.gettempdir()) / "emap2lig-gzcache"
        cache_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(
            f"{target}|{target.stat().st_mtime_ns}".encode()
        ).hexdigest()[:24]
        gz = cache_root / f"{target.stem}-{key}.gz"
        if not gz.exists():
            tmp = gz.with_name(gz.name + f".tmp.{os.getpid()}")
            with (
                open(target, "rb") as src,
                gzip.open(tmp, "wb", compresslevel=1) as dst,
            ):
                shutil.copyfileobj(src, dst, length=256 * 1024)
            tmp.replace(gz)
        return gz, original_size
    except OSError:
        return None


def _served_file_response(
    request: Request, target: Path, media_type: str, filename: str | None = None
) -> Response:
    """Serve a file with a stable ETag and 304 negotiation.

    The ETag is derived from the *original* file's path/mtime/size (not the
    gz cache file), so it stays stable across rebuilds of the cache, and
    unchanged files are re-served as 304 with zero body — which is what keeps
    tab switches and scene rebuilds from re-downloading maps.
    """
    headers = {}
    cached = _gzip_cached(target)
    if cached is None:
        serve_path, size = target, target.stat().st_size
    else:
        gz_path, size = cached
        serve_path = gz_path
        headers["Content-Encoding"] = "gzip"
        headers["X-Original-Size"] = str(size)

    etag = f'"{hashlib.sha1(f"{target}|{target.stat().st_mtime_ns}|{size}".encode()).hexdigest()[:24]}"'
    headers["ETag"] = etag
    headers["Cache-Control"] = "no-cache"

    if request.headers.get("if-none-match", "").strip() == etag:
        return Response(status_code=304, headers=headers)

    return FileResponse(
        path=serve_path, media_type=media_type, filename=filename, headers=headers
    )


def _allowed_roots() -> list[Path]:
    """Return the allow-listed roots for server-side browsing/serving.

    Defaults to the current user's home directory and ``jobs_output/``.
    Can be extended via ``EMAP2LIG_ALLOWED_ROOTS`` (comma-separated paths).
    """
    roots = {Path.home().resolve(), JOBS_ROOT.resolve()}
    extra = os.environ.get("EMAP2LIG_ALLOWED_ROOTS", "")
    for part in extra.split(","):
        p = part.strip()
        if not p:
            continue
        roots.add(Path(p).expanduser().resolve())
    return sorted(roots, key=lambda p: str(p))


def _is_allowed_path(target: Path) -> bool:
    roots = _allowed_roots()
    return any(target.is_relative_to(root) for root in roots)


def _require_allowed_path(target: Path) -> None:
    if _is_allowed_path(target):
        return
    allowed = ", ".join(str(r) for r in _allowed_roots())
    raise HTTPException(
        403,
        "Path is outside allowed roots. "
        f"Allowed roots: {allowed}. "
        "Set EMAP2LIG_ALLOWED_ROOTS to extend.",
    )


def _parent_if_allowed(target: Path) -> str | None:
    parent = target.parent
    if parent == target:
        return None
    return str(parent) if _is_allowed_path(parent) else None


# ---------------------------------------------------------------------------
# Server-side directory browser
# ---------------------------------------------------------------------------


class DirEntry(BaseModel):
    name: str
    is_dir: bool


class BrowseResponse(BaseModel):
    path: str
    parent: str | None
    entries: list[DirEntry]


@router.get("/browse-dirs", response_model=BrowseResponse)
async def browse_directories(
    path: str = Query(default="~", description="Absolute path or ~ to browse"),
    include_files: bool = Query(default=False, description="Include files in listing"),
    extensions: str = Query(
        default="",
        description="Comma-separated list of extensions to include (e.g. .mrc,.map,.gz)",
    ),
):
    """List directories (and optionally files) under a server path.

    WARNING: This endpoint is intended for localhost-only use.
    Paths are restricted to allow-listed roots (by default, your home directory
    and the app's ``jobs_output/`` directory).

    Used by the frontend to let users pick an output directory or a file.
    """
    target = Path(path).expanduser().resolve()
    _require_allowed_path(target)

    if not target.exists():
        # Return empty listing for non-existent paths (e.g. new output dir)
        parent = _parent_if_allowed(target)
        return BrowseResponse(path=str(target), parent=parent, entries=[])
    if not target.is_dir():
        raise HTTPException(400, f"Not a directory: {target}")

    ext_set = (
        {e.strip().lower() for e in extensions.split(",") if e.strip()}
        if extensions
        else set()
    )

    entries: list[DirEntry] = []
    try:
        for child in sorted(target.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append(DirEntry(name=child.name, is_dir=True))
            elif include_files:
                # Check suffix — handle double extensions like .map.gz
                suffixes = "".join(child.suffixes).lower()
                if not ext_set or any(suffixes.endswith(ext) for ext in ext_set):
                    entries.append(DirEntry(name=child.name, is_dir=False))
    except PermissionError as err:
        raise HTTPException(403, f"Permission denied: {target}") from err

    parent = _parent_if_allowed(target)

    return BrowseResponse(
        path=str(target),
        parent=parent,
        entries=entries,
    )


@router.get("/cwd")
async def get_cwd():
    """Return the server's current working directory."""
    return {"cwd": os.getcwd()}


@router.get("/serve-path")
async def serve_server_file(
    request: Request,
    path: str = Query(..., description="Absolute file path on the server"),
):
    """Serve an arbitrary file from the server filesystem.

    WARNING: This endpoint is intended for localhost-only use.
    Access is restricted to allow-listed roots (by default, your home directory
    and the app's ``jobs_output/`` directory).

    Used to preview MRC/map files that already exist on the server
    before submitting a job.  Files are served as-is (including .gz);
    the client (Molstar) handles gzip decompression natively.
    """
    target = Path(path).expanduser().resolve()
    _require_allowed_path(target)

    if not target.exists():
        raise HTTPException(404, f"File not found: {target}")
    if not target.is_file():
        raise HTTPException(400, f"Not a file: {target}")

    suffix = target.suffix.lower()
    media_type = _MIME.get(suffix, "application/octet-stream")
    return _served_file_response(request, target, media_type, filename=target.name)


# MIME types for structural biology formats
_MIME = {
    ".mrc": "application/octet-stream",
    ".map": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".cif": "chemical/x-mmcif",
    ".csv": "text/csv",
    ".gz": "application/gzip",  # truthful type; gzip middleware skips recompression
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
}


@router.get("/files/{job_id}/{file_path:path}")
async def serve_file(request: Request, job_id: str, file_path: str):
    """Serve a file from a job's output directory.

    The *file_path* is relative to the job's output_dir.
    Requests are constrained to stay within the job directory.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")

    try:
        output_root = job.output_dir.resolve()
        full_path = (output_root / file_path).resolve()
        if not full_path.is_relative_to(output_root):
            raise HTTPException(400, "Invalid path")
    except Exception as e:
        raise HTTPException(400, "Invalid path") from e

    if not full_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")

    suffix = full_path.suffix.lower()
    media_type = _MIME.get(suffix, "application/octet-stream")
    return _served_file_response(
        request, full_path, media_type, filename=full_path.name
    )


@router.get("/load-results")
async def load_results_from_dir(
    path: str = Query(..., description="Absolute path to an output directory"),
):
    """Load blobs and CSV results from an existing output directory.

    Used by the Visualization tab to inspect results from previous runs.
    """
    target = Path(path).expanduser().resolve()
    _require_allowed_path(target)
    if not target.exists():
        raise HTTPException(404, f"Directory not found: {target}")
    if not target.is_dir():
        raise HTTPException(400, f"Not a directory: {target}")

    layout = OutputLayout(target)
    blobs_dir = layout.find_blobs
    blob_infos = scan_blobs(blobs_dir) if blobs_dir.is_dir() else []
    blobs = [
        {
            "id": b.id,
            "num_voxels": b.num_voxels,
            "mask_url": f"/api/serve-path?path={target / b.mask_path}",
            "density_url": f"/api/serve-path?path={target / b.density_path}",
        }
        for b in blob_infos
    ]

    # Scan outputs
    outputs_dir = layout.build_struct
    blob_results = scan_results(outputs_dir) if outputs_dir.is_dir() else []
    results = [
        {
            "blob_id": br.blob_id,
            "conformers": [
                {
                    "name": c.name,
                    "score": c.score,
                    "cif_url": f"/api/serve-path?path={target / c.cif_path}",
                    "mask_url": f"/api/serve-path?path={target / c.mask_path}"
                    if c.mask_path
                    else None,
                }
                for c in br.conformers
            ],
        }
        for br in blob_results
    ]

    # Unified map
    unified_map = layout.unified_map
    unified_map_url = (
        f"/api/serve-path?path={unified_map}" if unified_map.exists() else None
    )

    return {
        "blobs": blobs,
        "results": results,
        "unified_map_path": unified_map_url,
    }


# ---------------------------------------------------------------------------
# Download selected files as a single .tar.gz archive
# ---------------------------------------------------------------------------


class DownloadArchiveRequest(BaseModel):
    """List of API URLs (relative, as used in the browser) to bundle."""

    urls: list[str]


def _resolve_api_url(url: str) -> Path | None:
    """Resolve an internal API URL to an actual file path on disk.

    Supports two patterns used by this app:
      - /api/serve-path?path=<abs_path>
      - /api/files/<job_id>/<relative_path>
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)

    if parsed.path == "/api/serve-path":
        qs = parse_qs(parsed.query)
        paths = qs.get("path", [])
        if paths:
            target = Path(paths[0]).expanduser().resolve()
            if not _is_allowed_path(target):
                return None
            return target

    if parsed.path.startswith("/api/files/"):
        # /api/files/<job_id>/<relative>
        parts = parsed.path.split("/", 4)  # ['', 'api', 'files', job_id, relative]
        if len(parts) >= 5:
            job_id = parts[3]
            relative = parts[4]
            job = get_job(job_id)
            if job:
                try:
                    output_root = job.output_dir.resolve()
                    candidate = (output_root / relative).resolve()
                    if candidate.is_relative_to(output_root):
                        return candidate
                except Exception:
                    return None

    return None


@router.post("/download-archive")
async def download_archive(body: DownloadArchiveRequest):
    """Create a tar.gz archive of the requested files and stream it back.

    Accepts the same API URLs the frontend uses for serving files.
    Each file is stored in the archive with just its basename.
    """
    if not body.urls:
        raise HTTPException(400, "No files specified")

    resolved: list[Path] = []
    for url in body.urls:
        fp = _resolve_api_url(url)
        if fp is None or not fp.exists() or not fp.is_file():
            raise HTTPException(404, f"Could not resolve file: {url}")
        resolved.append(fp)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        seen_names: dict[str, int] = {}
        for fp in resolved:
            # Deduplicate basenames
            name = fp.name
            if name in seen_names:
                seen_names[name] += 1
                stem = fp.stem
                suffix = "".join(fp.suffixes)
                name = f"{stem}_{seen_names[name]}{suffix}"
            else:
                seen_names[name] = 0
            tar.add(str(fp), arcname=name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/gzip",
        headers={"Content-Disposition": "attachment; filename=emap2lig_results.tar.gz"},
    )
