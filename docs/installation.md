# Installation

Emap2lig targets **Python 3.12** and **uv** for dependency management.

## Option A: CLI only (`uv tool install`)

Install `emap2lig` and `fragment-detect` on your PATH without cloning the
repository:

```bash
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig
```

Verify:

```bash
emap2lig --help
fragment-detect --help
```

Update to the latest release:

```bash
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig --reinstall
```

This path does **not** include the Web GUI (see Option B).

## Option B: Clone the repository

Use this for development, `uv run` workflows, or the Web GUI:

```bash
git clone https://github.com/kiharalab/Emap2lig.git
cd Emap2lig
uv sync
```

Run CLI commands from the repo:

```bash
uv run emap2lig --help
uv run fragment-detect --help
```

### Web GUI dependencies

The GUI needs the `web` dependency group. The public repository ships a pre-built
`app/frontend/dist/`; **Node.js is not required** to start the server.

```bash
uv sync --group web
uv run --group web python app/start.py
```

See [Web GUI](web-gui.md) for ports, headless mode, and frontend development.

## Model weights

Weights and CCD data download automatically on first run to `~/.emap2lig/models/`.
See [Model weights](models.md) for files, licensing, and maintainer update steps.

## Hardware (local only)

Local inference requires an NVIDIA GPU. See [README — Local usage](../README.md#local).
