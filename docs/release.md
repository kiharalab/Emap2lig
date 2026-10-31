# Release process

Releases are published by hand. Bump the version, push an annotated tag, then
run the **Publish** workflow from GitHub Actions.

## How to publish

1. On `main`, bump the Python package version in `pyproject.toml` and run
   `uv lock`.
2. Update `CHANGELOG.md`.
3. If the release should ship a newer web UI, publish `emap2lig-webui` to npm
   first, then run `scripts/sync_frontend.py <version>` and commit
   `.webui-version` (see [Web GUI](web-gui.md#frontend-bundle)).
4. Commit and push `main`.
5. Create and push the matching annotated tag:

   ```bash
   git tag -a v0.7.1 -m "v0.7.1"
   git push origin v0.7.1
   ```

6. In GitHub Actions, run **Publish** and set **tag** to that tag
   (`v0.7.1`). The workflow checks out the tag, fetches
   `emap2lig-webui@<.webui-version>` from npm, builds the sdist and wheel,
   creates or updates the GitHub Release, and uploads to PyPI.

Pushing a tag does not publish by itself.

## Version files

| File | Role |
| --- | --- |
| `pyproject.toml` | Python package version; must match the `v*` tag |
| `uv.lock` | Regenerated after the version bump |
| `CHANGELOG.md` | Hand-written release notes |
| `.webui-version` | npm `emap2lig-webui` pin fetched by `publish.yml` |
