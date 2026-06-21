# WEB GUI — KNOWLEDGE BASE

**Status:** Functional (v0.4.0)

## OVERVIEW

Web GUI for Emap2lig inference at `src/emap2lig/web/`. Find and Build are separate tabs.

Backend: **FastAPI**. Frontend: **React + shadcn/ui** with **Mol\***.

Frontend **source** is in private repo **shuuul/Emap2lig-web**. This repo tracks
only `frontend/dist/` (synced via CI PR).

## STRUCTURE

```
src/emap2lig/web/
├── cli.py              # Typer launcher (`emap2lig-gui` entry point)
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
# Run GUI (pre-built dist/; no npm required)
uv run --group web emap2lig-gui

# PyPI install
pip install "emap2lig[web]"
emap2lig-gui

# Backend dev
uv run --group web fastapi dev emap2lig.web.app:app

# Frontend dev (separate repo)
git clone git@github.com:shuuul/Emap2lig-web.git
cd Emap2lig-web && npm install && npm run dev
# Build into this repo:
# EMAP2LIG_DIST_DIR=../Emap2Lig/src/emap2lig/web/frontend/dist npm run build
```

## CONVENTIONS

- Keep heavy ML imports lazy (inside service functions).
- Align `Emap2lig-web` `package.json` version with root `pyproject.toml` before release.
- Do not inline version into generated JS chunks; only `dist/index.html` cache-bust query.
- **Packaging**: only `frontend/dist/` is published (wheel/sdist).

## CI SYNC

`shuuul/Emap2lig-web` workflow `.github/workflows/sync-dist.yml` opens PRs against
`kiharalab/Emap2Lig` updating `src/emap2lig/web/frontend/dist/`.
Requires `EMAP2LIG_SYNC_TOKEN` secret (PAT with write access to kiharalab/Emap2Lig).
