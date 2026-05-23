# Web GUI

Browser-based interface in `app/` with Mol* visualization. Find and Build are
separate tabs, matching the two-stage pipeline.

## Prerequisites

- Python **3.12** and [uv](https://docs.astral.sh/uv/)
- NVIDIA CUDA GPU (same as CLI)
- Clone the repository — the GUI is **not** available via `uv tool install`

## Install and run

```bash
git clone https://github.com/kiharalab/Emap2lig.git
cd Emap2lig
uv sync --group web
uv run --group web python app/start.py
```

Open `http://localhost:40427` (default).

The repository includes a pre-built `app/frontend/dist/`. **Node.js and npm are
not required** for normal use.

### Options

```bash
# Custom port
uv run --group web python app/start.py --port 8080

# Headless (no browser auto-open)
uv run --group web python app/start.py --no-browser
```

| Option | Description |
|--------|-------------|
| `--port INT` | Port (default: `40427`) |
| `--no-browser` | Do not open a browser tab |
| `--rebuild` | Rebuild frontend from source (requires npm; see below) |

## Workflow

- **Setup** — model cache path, download weights, GPU selection
- **Find** — input map, detection options, inspect blobs in Mol*
- **Build** — ligands (global or per-blob), run modeling, view conformers
- **Visualization** — load an existing output directory

Input formats match the CLI: [Input formats](input-format.md).

## Frontend development (maintainers)

React source under `app/frontend/src/` is **not** published on GitHub; only
`app/frontend/dist/` is tracked. To change the UI:

1. Use a full checkout that includes `package.json` and `src/`.
2. Install Node.js **18+** and npm.
3. `cd app/frontend && npm install && npm run build`
4. Commit updated `dist/` artifacts, or run `app/start.py --rebuild` locally.

Without source, `--rebuild` cannot succeed on a dist-only clone.

## Architecture

For API routes and component layout, see
[`skills/emap2lig/references/GUI.md`](../skills/emap2lig/references/GUI.md).
