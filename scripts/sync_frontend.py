"""Sync ``web/frontend/dist/`` from the pinned ``emap2lig-webui`` npm package.

The web UI source is not distributed. Only the pre-built bundle is published to npm
as ``emap2lig-webui`` and copied into this repository.

Usage:
    uv run python scripts/sync_frontend.py          # sync the version in .webui-version
    uv run python scripts/sync_frontend.py 0.7.0    # sync 0.7.0, then pin it

Set ``NPM_CONFIG_REGISTRY`` to use a registry other than registry.npmjs.org.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PACKAGE = "emap2lig-webui"
ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / ".webui-version"
DIST_DIR = ROOT / "src" / "emap2lig" / "web" / "frontend" / "dist"
TAR_PREFIX = "package/dist/"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _verify_integrity(data: bytes, integrity: str) -> None:
    algorithm, _, expected = integrity.partition("-")
    if algorithm != "sha512":
        sys.exit(f"Unsupported integrity algorithm: {integrity!r}")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if actual != expected:
        sys.exit("Tarball integrity check failed; refusing to use it.")


def _extract_dist(data: bytes, dest: Path) -> int:
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
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
            count += 1
    if not (root / "index.html").is_file():
        sys.exit("Tarball has no dist/index.html; wrong package?")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to sync and pin (default: the one in .webui-version)",
    )
    args = parser.parse_args()

    version = args.version or VERSION_FILE.read_text().strip()
    if not VERSION_RE.fullmatch(version):
        sys.exit(f"Invalid version: {version!r}")

    registry = (
        os.environ.get("NPM_CONFIG_REGISTRY") or "https://registry.npmjs.org"
    ).rstrip("/")
    meta = json.loads(_fetch(f"{registry}/{PACKAGE}/{version}"))
    tarball = _fetch(meta["dist"]["tarball"])
    _verify_integrity(tarball, meta["dist"]["integrity"])

    # Extract completely before touching DIST_DIR so a failure leaves it intact.
    with tempfile.TemporaryDirectory() as tmp:
        count = _extract_dist(tarball, Path(tmp))
        shutil.rmtree(DIST_DIR, ignore_errors=True)
        shutil.copytree(tmp, DIST_DIR)

    VERSION_FILE.write_text(f"{version}\n")
    print(f"Synced {count} files from {PACKAGE}@{version} into {DIST_DIR}")


if __name__ == "__main__":
    main()
