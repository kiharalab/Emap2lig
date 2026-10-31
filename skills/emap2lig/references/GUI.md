---
title: Emap2lig Web GUI Reference
---

# Emap2lig Web GUI Reference

Detailed reference for the Emap2lig web GUI — architecture, API endpoints,
and troubleshooting.

## Architecture

```
src/emap2lig/web/
├── cli.py                # Typer launcher (`emap2lig-gui`)
├── app.py                # FastAPI app factory
├── routers/
│   ├── detect.py         # POST /api/detect (Stage 1)
│   ├── model.py          # POST /api/model, POST /api/model-blob (Stage 2)
│   ├── jobs.py           # GET /api/jobs/..., job status polling
│   ├── files.py          # GET /api/files/... (serve output files)
│   └── download.py       # Model download + cache/gpu endpoints
├── schemas.py
├── services.py
├── state.py
├── results_scan.py
└── frontend/
    └── dist/              # Pre-built assets (from the emap2lig-webui npm package)
```

The web UI source is not distributed; `dist/` is gitignored and fetched from the npm
package `emap2lig-webui` at the version pinned in `.webui-version`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Submit detection job (Stage 1) |
| `POST` | `/api/model` | Submit modeling for all blobs (Stage 2) |
| `POST` | `/api/model-blob` | Submit modeling for a single blob |
| `GET` | `/api/jobs` | List in-memory jobs |
| `GET` | `/api/jobs/{id}` | Poll job status (stages, queue_position) |
| `POST` | `/api/jobs/{id}/cancel` | Cancel a queued or running job |
| `GET` | `/api/files/...` | Serve output files to Mol* viewer |
| `GET` | `/api/download/status` | Model download status |
| `POST` | `/api/download/model` | Trigger model download |
| `GET` | `/api/gpu` | GPU availability |

## Job Model

Jobs run on a single-worker FIFO queue (`web/runner.py`) so two GPU stages
cannot overlap. Status progresses:
`queued` → `running` → `completed` / `failed` / `cancelled`.

Pipeline stages emit structured `ProgressEvent`s; the job payload includes
`stages` (current/total/failed per stage) and `queue_position` while queued.
`POST /api/jobs/{id}/cancel` drops a queued job immediately, or sets a cancel
flag so a running job stops at the next progress event.

Job state is stored in-memory (no database). Restarting the server loses
active jobs but results on disk remain accessible via the Visualization tab.

## Mol* Viewer Scene Groups

| Group | Contents |
|-------|----------|
| `maps` | Unified map objects |
| `find-blobs` | Blob masks from detection |
| `user-upload` | User-added structures/maps |
| `build` | Modeled structures and masks |

## Development Commands

```bash
# Backend dev (hot reload)
uv run --group web fastapi dev emap2lig.web.app:app

# Fetch frontend/dist/ from npm (required in a checkout)
uv run python scripts/sync_frontend.py

# Or copy dist/ from a local Emap2lig-web checkout
uv run python scripts/sync_frontend.py --local
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | `cli.py` auto-frees the port; or use `--port` |
| Frontend not loading | Ensure `frontend/dist/` exists; run `uv run python scripts/sync_frontend.py` |
| Model download fails | Check internet; weights stored in `~/.emap2lig/models/` |
| Job stuck at queued | Another job is using the GPU; cancel it or wait |
| Job stuck at running | Restart server; check GPU availability |
| Mol* shows nothing | Verify map at correct contour level; check file paths |
