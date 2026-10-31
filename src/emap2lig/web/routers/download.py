"""Model download, cache management, and GPU info endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import (
    CacheInfoResponse,
    DownloadStatusResponse,
    GpuListResponse,
    ModelStatusResponse,
    SetCacheDirRequest,
)
from ..services import (
    check_model_cache,
    get_cache_info,
    get_download_status,
    list_gpus,
    set_cache_dir,
    start_model_download,
)

router = APIRouter(prefix="/api", tags=["download"])


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Check if model weights are cached locally."""
    status = check_model_cache()
    return ModelStatusResponse(**status)


@router.post("/download-model", response_model=DownloadStatusResponse)
async def trigger_model_download():
    """Start downloading missing model weights in the background.

    Poll ``/api/download/status`` for per-file progress (bytes/total).
    """
    start_model_download()
    return get_download_status()


@router.get("/download/status", response_model=DownloadStatusResponse)
async def download_status():
    """Per-file download progress (cached flag, total size, downloaded bytes)."""
    return get_download_status()


@router.get("/cache-info", response_model=CacheInfoResponse)
async def get_cache_info_endpoint():
    """Return cache directory path and per-model file info."""
    return CacheInfoResponse(**get_cache_info())


@router.post("/cache-dir", response_model=CacheInfoResponse)
async def update_cache_dir(body: SetCacheDirRequest):
    """Update the model cache directory (runtime only)."""
    return CacheInfoResponse(**set_cache_dir(body.cache_dir))


@router.get("/gpus", response_model=GpuListResponse)
async def list_gpus_endpoint():
    """List available CUDA GPUs."""
    return GpuListResponse(**list_gpus())
