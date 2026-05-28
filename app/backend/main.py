"""FastAPI application factory for Emap2lig Web GUI."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routers import detect, download, files, jobs, model

# Resolved once at import time so it works regardless of cwd.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Browsers must revalidate index.html after frontend dist rebuilds so script
# tags point at the current hashed bundles under /assets/.
_SPA_INDEX_CACHE_HEADERS = {"Cache-Control": "no-cache"}


def _resolve_frontend_dist_file(dist_root: Path, full_path: str) -> Path | None:
    """Resolve a requested SPA path to a real file inside *dist_root*.

    Returns ``None`` when the path is empty, points outside *dist_root*, or does
    not exist as a file.
    """
    if not full_path:
        return None
    try:
        dist_root = dist_root.resolve()
        candidate = (dist_root / full_path).resolve()
    except Exception:
        return None
    if not candidate.is_relative_to(dist_root):
        return None
    return candidate if candidate.is_file() else None


def _spa_index_response(dist_root: Path) -> FileResponse:
    """Return ``index.html`` with cache headers that force revalidation.

    Args:
        dist_root: Built frontend directory containing ``index.html``.

    Returns:
        File response for the SPA shell with ``Cache-Control: no-cache``.
    """
    return FileResponse(
        dist_root / "index.html",
        headers=_SPA_INDEX_CACHE_HEADERS,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Emap2lig",
        description="Web GUI for cryo-EM ligand detection and structure modeling",
        version="0.3.3",
    )

    # CORS — allow the Vite dev server (port 5173) and any localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routers ──────────────────────────────────────────────
    app.include_router(detect.router)
    app.include_router(model.router)
    app.include_router(jobs.router)
    app.include_router(files.router)
    app.include_router(download.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/version")
    async def version():
        try:
            pkg_version = importlib.metadata.version("emap2lig")
        except importlib.metadata.PackageNotFoundError:
            pkg_version = "unknown"
        return {"version": pkg_version}

    # ── Serve built frontend (SPA) ───────────────────────────────
    if _FRONTEND_DIST.is_dir():
        # Static assets (JS, CSS, images) under /assets
        assets = _FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets)),
                name="frontend-assets",
            )

        # SPA catch-all: anything that is NOT /api/* returns index.html
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Try to serve the exact file first (e.g. favicon, vite.svg),
            # but never allow path traversal outside frontend/dist.
            candidate = _resolve_frontend_dist_file(_FRONTEND_DIST, full_path)
            if candidate is not None:
                if candidate.name == "index.html":
                    return FileResponse(candidate, headers=_SPA_INDEX_CACHE_HEADERS)
                return FileResponse(candidate)
            return _spa_index_response(_FRONTEND_DIST)

    return app


app = create_app()
