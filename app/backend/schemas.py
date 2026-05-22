"""Pydantic request / response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ligand specification (frontend → backend)
# ---------------------------------------------------------------------------


class LigandSpec(BaseModel):
    type: Literal["CCD", "SMILES", "BRANCHED"]
    name: str | None = None
    smiles: str | None = None
    residues: dict[int, str] | None = None
    bonds: list[list[int | str]] | None = None
    blob_id: list[int] | None = None


# ---------------------------------------------------------------------------
# Job responses
# ---------------------------------------------------------------------------


class BlobInfoResponse(BaseModel):
    id: int
    num_voxels: int
    mask_url: str
    density_url: str


class ConformerResponse(BaseModel):
    name: str
    score: float
    cif_url: str
    mask_url: str | None = None


class BlobResultResponse(BaseModel):
    blob_id: int
    conformers: list[ConformerResponse] = Field(default_factory=list)


class JobResponse(BaseModel):
    id: str
    type: str
    status: str
    created_at: datetime
    progress: str
    error: str | None = None
    output_dir: str | None = None
    num_blobs: int | None = None
    detect_job_id: str | None = None
    unified_map_url: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str


class BlobListResponse(BaseModel):
    blobs: list[BlobInfoResponse]


class ResultsResponse(BaseModel):
    results: list[BlobResultResponse]


# ---------------------------------------------------------------------------
# Model / cache info
# ---------------------------------------------------------------------------


class ModelStatusResponse(BaseModel):
    detection_model: bool  # True if cached
    structure_model: bool


class ModelWeightInfo(BaseModel):
    key: str
    filename: str
    path: str
    exists: bool
    size_mb: float | None = None  # file size in MB, None if not exists


class CacheInfoResponse(BaseModel):
    cache_dir: str
    models: list[ModelWeightInfo]


class SetCacheDirRequest(BaseModel):
    cache_dir: str


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------


class GpuInfo(BaseModel):
    id: int
    name: str
    memory_mb: int


class GpuListResponse(BaseModel):
    cuda_available: bool
    gpus: list[GpuInfo]
