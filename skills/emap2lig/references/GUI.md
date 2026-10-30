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
| `GET` | `/api/jobs/{id}` | Poll job status |
| `GET` | `/api/jobs/{id}/logs` | Get job logs |
| `GET` | `/api/files/...` | Serve output files to Mol* viewer |
| `GET` | `/api/download/status` | Model download status |
| `POST` | `/api/download/model` | Trigger model download |
| `GET` | `/api/gpu` | GPU availability |

## Job Model

Jobs run in background threads (`asyncio.to_thread`). Status progresses:
`pending` → `running` → `completed` / `failed`.

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
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | `cli.py` auto-frees the port; or use `--port` |
| Frontend not loading | Ensure `frontend/dist/` exists; run `uv run python scripts/sync_frontend.py` |
| Model download fails | Check internet; weights stored in `~/.emap2lig/models/` |
| Job stuck at pending | Restart server; check GPU availability |
| Mol* shows nothing | Verify map at correct contour level; check file paths |
