"""Sync ``web/frontend/dist/`` from npm or a local Emap2lig-web checkout.

The web UI source is not distributed. Only the pre-built bundle is copied into
this repository, either from the ``emap2lig-webui`` npm package or from a sibling
frontend checkout.

Usage:
    uv run python scripts/sync_frontend.py
        Fetch the version pinned in ``.webui-version`` from npm.

    uv run python scripts/sync_frontend.py 0.7.0
        Fetch 0.7.0 from npm and pin it in ``.webui-version``.

    uv run python scripts/sync_frontend.py --local
        Copy ``dist/`` from ``../Emap2lig-web`` (or ``$EMAP2LIG_WEBUI``).
        Does not change ``.webui-version``.

    uv run python scripts/sync_frontend.py --local /path/to/Emap2lig-web --build
        Run ``bun run build`` in that checkout, then copy ``dist/``.

Set ``NPM_CONFIG_REGISTRY`` to use a registry other than registry.npmjs.org.
Set ``EMAP2LIG_WEBUI`` to override the default local checkout path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO

PACKAGE = "emap2lig-webui"
ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / ".webui-version"
DIST_DIR = ROOT / "src" / "emap2lig" / "web" / "frontend" / "dist"
TAR_PREFIX = "package/dist/"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")
DEFAULT_LOCAL_NAME = "Emap2lig-web"


def _fetch(url: str) -> bytes:
    """Download *url* and return the response body.

    Args:
        url: HTTP URL to fetch.

    Returns:
        Raw response bytes.
    """
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _verify_integrity(data: bytes, integrity: str) -> None:
    """Check *data* against an npm ``sha512-`` integrity string.

    Args:
        data: Tarball bytes.
        integrity: npm ``dist.integrity`` value.

    Raises:
        SystemExit: If the algorithm is unsupported or the digest does not match.
    """
    algorithm, _, expected = integrity.partition("-")
    if algorithm != "sha512":
        sys.exit(f"Unsupported integrity algorithm: {integrity!r}")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if actual != expected:
        sys.exit("Tarball integrity check failed; refusing to use it.")


def _extract_dist(data: bytes, dest: Path) -> int:
    """Extract ``package/dist/`` from an npm tarball into *dest*.

    Args:
        data: gzip-compressed npm package tarball.
        dest: Directory that will receive the dist files.

    Returns:
        Number of files extracted.

    Raises:
        SystemExit: If the tarball is missing ``dist/index.html`` or contains
            unsafe or unexpected entries.
    """
    count = 0
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if not member.name.startswith(TAR_PREFIX):
                continue
            if not member.isfile():
                sys.exit(f"Unexpected non-file entry in tarball: {member.name}")
            target = (root / member.name[len(TAR_PREFIX) :]).resolve()
            if not target.is_relative_to(root):
                sys.exit(f"Unsafe path in tarball: {member.name}")
            extracted: BinaryIO | None = tar.extractfile(member)
            if extracted is None:
                sys.exit(f"Cannot read tarball member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())
            count += 1
    if not (root / "index.html").is_file():
        sys.exit("Tarball has no dist/index.html; wrong package?")
    return count


def _install_dist(src: Path, dest: Path) -> int:
    """Replace *dest* with the contents of *src*.

    Copy into a temporary directory first so a failure leaves *dest* intact.

    Args:
        src: Built frontend directory that must contain ``index.html``.
        dest: Target ``frontend/dist`` directory.

    Returns:
        Number of files installed.

    Raises:
        SystemExit: If *src* has no ``index.html``.
    """
    if not (src / "index.html").is_file():
        sys.exit(f"No index.html in {src}; build the frontend first or pass --build.")
    files = [path for path in src.rglob("*") if path.is_file()]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copytree(src, tmp_path, dirs_exist_ok=True)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(tmp_path, dest)
    return len(files)


def _load_package_json(source: Path) -> dict[str, object]:
    """Load ``package.json`` from a frontend checkout.

    Args:
        source: Frontend checkout root.

    Returns:
        Parsed package object.

    Raises:
        SystemExit: If the file is missing or not a JSON object.
    """
    package_json = source / "package.json"
    if not package_json.is_file():
        sys.exit(f"{source} has no package.json; not an Emap2lig-web checkout.")
    try:
        payload = json.loads(package_json.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"{package_json} is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        sys.exit(f"{package_json} must contain a JSON object.")
    return payload


def resolve_local_webui(explicit: str | None, *, root: Path = ROOT) -> Path:
    """Resolve a local Emap2lig-web checkout.

    Lookup order: *explicit* path, ``EMAP2LIG_WEBUI``, then
    ``<repo-parent>/Emap2lig-web``.

    Args:
        explicit: Path from ``--local PATH``, or ``None``/empty for the default.
        root: Emap2lig repository root.

    Returns:
        Absolute path to the frontend checkout.

    Raises:
        SystemExit: If the resolved path does not exist.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
    elif os.environ.get("EMAP2LIG_WEBUI"):
        resolved = Path(os.environ["EMAP2LIG_WEBUI"]).expanduser().resolve()
    else:
        resolved = (root.parent / DEFAULT_LOCAL_NAME).resolve()
    if not resolved.is_dir():
        sys.exit(
            f"Local frontend checkout not found at {resolved}.\n"
            "  Pass --local PATH or set EMAP2LIG_WEBUI."
        )
    return resolved


def _require_frontend_checkout(source: Path) -> dict[str, object]:
    """Reject a directory that is not the Emap2lig-web frontend.

    Args:
        source: Candidate checkout root.

    Returns:
        Parsed ``package.json`` object.

    Raises:
        SystemExit: If ``package.json`` is missing or the package name differs.
    """
    payload = _load_package_json(source)
    if "name" not in payload or not isinstance(payload["name"], str):
        sys.exit(f"{source}/package.json has no string 'name' field.")
    name = payload["name"]
    if name != PACKAGE:
        sys.exit(f"{source} is package {name!r}, expected {PACKAGE!r}.")
    return payload


def build_local(source: Path) -> None:
    """Run ``bun run build`` in *source*.

    Args:
        source: Emap2lig-web checkout.

    Raises:
        SystemExit: If ``bun`` is missing or the build fails.
    """
    bun = shutil.which("bun")
    if bun is None:
        sys.exit("bun not found on PATH; install Bun or omit --build and copy dist/.")
    print(f"Building frontend in {source} ...")
    result = subprocess.run([bun, "run", "build"], cwd=source, check=False)
    if result.returncode != 0:
        sys.exit(f"bun run build failed with exit {result.returncode}")


def sync_from_local(source: Path, dest: Path, *, build: bool = False) -> int:
    """Copy ``dist/`` from a local frontend checkout into *dest*.

    Args:
        source: Emap2lig-web checkout root.
        dest: Target ``frontend/dist`` directory.
        build: If True, run ``bun run build`` before copying.

    Returns:
        Number of files copied.
    """
    payload = _require_frontend_checkout(source)
    if build:
        build_local(source)
    count = _install_dist(source / "dist", dest)
    if "version" in payload and isinstance(payload["version"], str):
        version = payload["version"]
    else:
        version = "unknown"
    print(f"Synced {count} files from {source} ({PACKAGE}@{version}) into {dest}")
    return count


def sync_from_npm(
    version: str,
    dest: Path,
    version_file: Path,
    *,
    registry: str | None = None,
) -> int:
    """Download ``emap2lig-webui@*version*`` and install it into *dest*.

    Args:
        version: npm package version.
        dest: Target ``frontend/dist`` directory.
        version_file: ``.webui-version`` path to rewrite after a successful sync.
        registry: npm registry origin. Defaults to ``NPM_CONFIG_REGISTRY`` or
            ``https://registry.npmjs.org``.

    Returns:
        Number of files installed.

    Raises:
        SystemExit: If *version* is invalid or the tarball is rejected.
    """
    if not VERSION_RE.fullmatch(version):
        sys.exit(f"Invalid version: {version!r}")

    origin = (
        registry
        or os.environ.get("NPM_CONFIG_REGISTRY")
        or "https://registry.npmjs.org"
    ).rstrip("/")
    meta = json.loads(_fetch(f"{origin}/{PACKAGE}/{version}"))
    tarball = _fetch(meta["dist"]["tarball"])
    _verify_integrity(tarball, meta["dist"]["integrity"])

    with tempfile.TemporaryDirectory() as tmp:
        count = _extract_dist(tarball, Path(tmp))
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(tmp, dest)

    version_file.write_text(f"{version}\n")
    print(f"Synced {count} files from {PACKAGE}@{version} into {dest}")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run python scripts/sync_frontend.py\n"
            "  uv run python scripts/sync_frontend.py 0.7.0\n"
            "  uv run python scripts/sync_frontend.py --local\n"
            "  uv run python scripts/sync_frontend.py --local ../Emap2lig-web --build\n"
        ),
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="npm version to sync and pin (default: the one in .webui-version)",
    )
    parser.add_argument(
        "--local",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "Copy dist/ from a local Emap2lig-web checkout instead of npm. "
            "PATH defaults to $EMAP2LIG_WEBUI or ../Emap2lig-web. "
            "Does not change .webui-version."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="With --local, run 'bun run build' before copying dist/.",
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    *,
    dest: Path = DIST_DIR,
    version_file: Path = VERSION_FILE,
    root: Path = ROOT,
) -> None:
    """Dispatch to the npm or local sync path.

    Args:
        args: Parsed CLI arguments.
        dest: Target ``frontend/dist`` directory.
        version_file: Pin file used by the npm path.
        root: Emap2lig repository root, used to resolve the default local path.
    """
    if args.build and args.local is None:
        sys.exit("--build requires --local")
    if args.local is not None and args.version:
        sys.exit("Pass either an npm version or --local, not both.")

    if args.local is not None:
        source = resolve_local_webui(args.local or None, root=root)
        sync_from_local(source, dest, build=args.build)
        return

    version = args.version or version_file.read_text().strip()
    sync_from_npm(version, dest, version_file)


def main() -> None:
    """CLI entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
