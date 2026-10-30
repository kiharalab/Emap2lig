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
├── app.py              # FastAPI app factory
├── routers/            # detect, model, jobs, files, download
├── schemas.py
├── services.py
├── state.py
├── results_scan.py
└── frontend/
    ├── dist/           # Pre-built static assets (packaged in wheel)
    └── .gitignore      # Ignores everything except dist/
```

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
```

## CONVENTIONS

- Keep heavy ML imports lazy (inside service functions).
- Never hand-edit `frontend/dist/`; change `.webui-version` and run `scripts/sync_frontend.py`.
- Do not inline version into generated JS chunks; assets are content-hashed, and index.html carries `<meta name="emap2lig-version">`.
- **Packaging**: only `frontend/dist/` is published (wheel/sdist); sync it before `uv build`.

## FRONTEND SYNC

`scripts/sync_frontend.py` downloads `emap2lig-webui@<.webui-version>` from the npm
registry, verifies its integrity hash, and replaces `frontend/dist/`. No token needed.
`.github/workflows/publish.yml` runs it before `uv build`.
