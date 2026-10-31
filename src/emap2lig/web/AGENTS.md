# WEB GUI — KNOWLEDGE BASE

**Status:** Functional (v0.4.0)

## OVERVIEW

Web GUI for Emap2lig inference at `src/emap2lig/web/`. Find and Build are separate tabs.

Backend: **FastAPI**. Frontend: **React + shadcn/ui** with **Mol\***.

The web UI **source** is not distributed. `frontend/dist/` is gitignored and fetched
from the npm package `emap2lig-webui` (pinned in `/.webui-version`) by
`scripts/sync_frontend.py`. The `emap2lig-gui` launcher lives in
`emap2lig/cli/gui.py`; output paths come from `emap2lig.layout.OutputLayout`.

## STRUCTURE

```
src/emap2lig/web/
├── app.py              # FastAPI app factory (job-runner lifespan)
├── runner.py           # Single-GPU FIFO queue + cooperative cancel
├── routers/            # detect, model, jobs, files, download
├── schemas.py
├── services.py         # Stage runners; structured progress → Job.stages
├── state.py
├── results_scan.py
└── frontend/
    ├── dist/           # Pre-built static assets (packaged in wheel)
    └── .gitignore      # Ignores everything except dist/
```

Jobs are queued on a single worker so two GPU stages cannot overlap. Progress is
native `ProgressEvent`s from the pipeline (not log scraping). Poll `GET /api/jobs/{id}`
for `stages` / `queue_position`; `POST /api/jobs/{id}/cancel` cancels queued jobs
immediately and running jobs at the next progress event.

## COMMANDS

```bash
# Run GUI (needs frontend/dist/: run scripts/sync_frontend.py once in a checkout)
uv run --group web emap2lig-gui

# PyPI install
pip install "emap2lig[web]"
emap2lig-gui

# Backend dev
uv run --group web fastapi dev emap2lig.web.app:app

# Fetch frontend/dist/ from npm (pinned version in .webui-version)
uv run python scripts/sync_frontend.py

# Copy dist/ from a local Emap2lig-web checkout (optional --build)
uv run python scripts/sync_frontend.py --local
uv run python scripts/sync_frontend.py --local /path/to/Emap2lig-web --build
```

## CONVENTIONS

- Keep heavy ML imports lazy (inside service functions).
- Never hand-edit `frontend/dist/`; sync from npm (`.webui-version`) or `--local`.
- Do not inline version into generated JS chunks; assets are content-hashed, and index.html carries `<meta name="emap2lig-version">`.
- **Packaging**: only `frontend/dist/` is published (wheel/sdist); sync it before `uv build`.
- **Transfer**: `GZipMiddleware` (level 6, min 1KB) compresses API/file responses;
  `/assets/` responses carry immutable cache headers for hashed filenames. Keep both.

## FRONTEND SYNC

`scripts/sync_frontend.py` installs `frontend/dist/` from either npm
(`emap2lig-webui@<.webui-version>`, integrity-checked) or `--local` (a sibling
`Emap2lig-web` checkout, or `$EMAP2LIG_WEBUI` / an explicit path). `--local` does
not rewrite `.webui-version`. `.github/workflows/publish.yml` always uses npm.
