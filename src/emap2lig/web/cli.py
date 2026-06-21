#!/usr/bin/env python3
"""One-command launcher for the Emap2lig web GUI.

Usage (from the repo root):
    uv run --group web emap2lig-gui [OPTIONS]

Or:
    uv run --group web python -m emap2lig.web.cli [OPTIONS]

Options:
    --port INT       Port to serve on (default: 40427)
    --rebuild        Force-rebuild the frontend even if dist/ exists
    --no-browser     Don't auto-open the browser

What it does:
    1. Uses pre-built frontend/dist/ when present (no npm required)
    2. Optionally installs npm deps and rebuilds (--rebuild, or missing dist/)
    3. Frees the chosen port if something stale is occupying it
    4. Opens http://localhost:<port> in the default browser
    5. Starts the FastAPI/Uvicorn server
"""

from __future__ import annotations

import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

WEB_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = WEB_DIR / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
DEFAULT_PORT = 40427

cli = typer.Typer(add_completion=False)


def _check_npm() -> str:
    """Return the path to npm, or exit with a helpful message."""
    npm = shutil.which("npm")
    if npm is None:
        print(
            "ERROR: npm is not installed. "
            "Install Node.js from https://nodejs.org/ and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    return npm


def _npm_install(npm: str) -> None:
    """Run `npm install` if node_modules/ doesn't exist yet."""
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.is_dir():
        print("  npm dependencies already installed, skipping.")
        return
    print("  Installing npm dependencies...")
    subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)


def _npm_build(npm: str, *, rebuild: bool = False) -> None:
    """Run `npm run build` to produce frontend/dist/."""
    if DIST_DIR.is_dir() and not rebuild:
        print("  Frontend already built, skipping. (use --rebuild to force)")
        return
    package_json = FRONTEND_DIR / "package.json"
    if not package_json.is_file():
        print(
            "ERROR: frontend source is not available in this checkout.\n"
            "  PyPI installs and kiharalab/Emap2Lig ship frontend/dist/ only.\n"
            "  Build from shuuul/Emap2lig-web and copy dist/, or remove --rebuild.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("  Building frontend...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)


def _prepare_frontend(*, rebuild: bool) -> None:
    """Ensure frontend/dist exists, using npm only when necessary."""
    if DIST_DIR.is_dir() and not rebuild:
        print("  Using pre-built frontend (emap2lig/web/frontend/dist/).")
        return

    if not DIST_DIR.is_dir():
        print("  No pre-built frontend found; building from source...")

    npm = _check_npm()
    _npm_install(npm)
    _npm_build(npm, rebuild=rebuild)


def _port_in_use(port: int) -> bool:
    """Check if a TCP port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port(port: int) -> None:
    """Kill any process listening on *port* (macOS/Linux)."""
    if not _port_in_use(port):
        return
    print(f"  Port {port} is in use — killing stale process...")
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in out.splitlines():
            try:
                import os

                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(0.5)
        if _port_in_use(port):
            print(f"  WARNING: could not free port {port}", file=sys.stderr)
        else:
            print(f"  Port {port} freed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"  WARNING: port {port} is occupied and could not be freed automatically.\n"
            f"  Kill the process manually, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)


def _open_browser(url: str, delay: float = 1.5) -> None:
    """Open the URL in the default browser after a short delay."""

    def _open():
        time.sleep(delay)
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


@cli.command()
def main(
    port: Annotated[
        int,
        typer.Option(help="Port to serve on."),
    ] = DEFAULT_PORT,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild", help="Force-rebuild the frontend even if dist/ exists."
        ),
    ] = False,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Don't auto-open the browser."),
    ] = False,
) -> None:
    """Launch the Emap2lig web GUI."""
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    # ── 1. Frontend build ────────────────────────────────────────
    print("[1/4] Preparing frontend...")
    _prepare_frontend(rebuild=rebuild)
    if not DIST_DIR.is_dir():
        print(
            "ERROR: frontend/dist/ is missing after prepare step.",
            file=sys.stderr,
        )
        sys.exit(1)
    print()

    # ── 2. Free port ─────────────────────────────────────────────
    print(f"[2/4] Checking port {port}...")
    _free_port(port)
    print()

    # ── 3. Open browser ──────────────────────────────────────────
    if no_browser:
        print(f"[3/4] Skipping browser (--no-browser). Visit {url}")
    else:
        print(f"[3/4] Opening {url} in browser...")
        _open_browser(url)
    print()

    # ── 4. Start backend ─────────────────────────────────────────
    print(f"[4/4] Starting Emap2lig server on {url}")
    print("       Press Ctrl+C to stop.\n")

    import uvicorn

    uvicorn.run(
        "emap2lig.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
