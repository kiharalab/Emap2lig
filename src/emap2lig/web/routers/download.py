"""Model download, cache management, and GPU info endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import (
    CacheInfoResponse,
    GpuListResponse,
    ModelStatusResponse,
    SetCacheDirRequest,
)
from ..services import (
    check_model_cache,
    download_models,
    get_cache_info,
    list_gpus,
    set_cache_dir,
)

router = APIRouter(prefix="/api", tags=["download"])


@router.get("/model-status", response_model=ModelStatusResponse)
async def get_model_status():
    """Check if model weights are cached locally."""
    status = check_model_cache()
    return ModelStatusResponse(**status)


@router.post("/download-model", response_model=ModelStatusResponse)
async def trigger_model_download():
    """Download model weights from HuggingFace Hub.

    Idempotent — skips files already cached.
    """
    status = await download_models()
    return ModelStatusResponse(**status)


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
