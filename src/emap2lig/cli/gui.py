#!/usr/bin/env python3
"""One-command launcher for the Emap2lig web GUI.

Usage (from the repo root):
    uv run --group web emap2lig-gui [OPTIONS]

Or:
    uv run --group web python -m emap2lig.web [OPTIONS]

Options:
    --port INT       Port to serve on (default: 40427)
    --no-browser     Don't auto-open the browser

What it does:
    1. Checks that the pre-built frontend/dist/ is present (no npm required;
       the frontend source lives in shuuul/Emap2lig-web)
    2. Frees the chosen port if something stale is occupying it
    3. Opens http://localhost:<port> in the default browser
    4. Starts the FastAPI/Uvicorn server
"""

from __future__ import annotations

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

DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "frontend" / "dist"
DEFAULT_PORT = 40427

cli = typer.Typer(add_completion=False)


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
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Don't auto-open the browser."),
    ] = False,
) -> None:
    """Launch the Emap2lig web GUI."""
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    # ── 1. Frontend build ────────────────────────────────────────
    print("[1/4] Checking frontend...")
    if not DIST_DIR.is_dir():
        print(
            f"ERROR: pre-built frontend not found at {DIST_DIR}.\n"
            "  Build it from shuuul/Emap2lig-web "
            "(EMAP2LIG_DIST_DIR=<this path> npm run build).",
            file=sys.stderr,
        )
        sys.exit(1)
    print("  Using pre-built frontend (emap2lig/web/frontend/dist/).")
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

    from emap2lig.utils.log import setup_logging

    setup_logging()

    uvicorn.run(
        "emap2lig.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    cli()
