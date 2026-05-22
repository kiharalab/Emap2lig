---
title: Emap2lig Web GUI Reference
---

# Emap2lig Web GUI Reference

Detailed reference for the Emap2lig web GUI — architecture, API endpoints,
and troubleshooting.

## Architecture

```
app/
├── start.py              # One-command launcher (npm build + uvicorn)
├── backend/
│   ├── main.py           # FastAPI app factory
│   ├── routers/
│   │   ├── detect.py     # POST /api/detect (Stage 1)
│   │   ├── model.py      # POST /api/model, POST /api/model-blob (Stage 2)
│   │   ├── jobs.py       # GET /api/jobs/..., job status polling
│   │   ├── files.py      # GET /api/files/... (serve output files)
│   │   └── download.py   # Model download + cache/gpu endpoints
│   ├── schemas.py        # Pydantic request/response models
│   ├── services.py       # Wrappers around emap2lig core functions
│   ├── state.py          # In-memory job store
│   └── results_scan.py   # On-disk output layout scanning
└── frontend/
    └── src/
        ├── App.tsx                  # Tab routing
        ├── components/
        │   ├── SetupTab.tsx         # Model cache + GPU
        │   ├── FindTab.tsx          # Detection workflow
        │   ├── BuildTab.tsx         # Modeling workflow
        │   ├── VisualizationTab.tsx # Load existing output
        │   ├── MolstarViewer.tsx    # Mol* 3D viewer
        │   ├── ObjectUpload.tsx     # Upload structures/maps
        │   ├── ResultsTable.tsx     # Conformer table
        │   ├── BlobList.tsx         # Per-blob assignment
        │   ├── LigandInput.tsx      # Ligand YAML input
        │   ├── MapInput.tsx         # Map input
        │   ├── RunOptionsPanel.tsx  # Detection options panel
        │   ├── JobStatus.tsx        # Job progress
        │   └── tutorial/            # Interactive tutorial
        └── lib/
            └── api.ts              # API client
```

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
uv run --group web fastapi dev backend.main:app --app-dir app

# Frontend dev (Vite dev server)
cd app/frontend
npm install
npm run dev
npm run build
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | `start.py` auto-frees the port; or use `--port` |
| Frontend not loading | Run `--rebuild` to force frontend rebuild |
| Model download fails | Check internet; weights stored in `~/.emap2lig/models/` |
| Job stuck at pending | Restart server; check GPU availability |
| Mol* shows nothing | Verify map at correct contour level; check file paths |
