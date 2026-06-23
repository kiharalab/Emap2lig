# Release process

Emap2lig uses [release-please](https://github.com/googleapis/release-please) to
prepare releases from Conventional Commit history.

## How releases are created

1. Merge normal pull requests into `main` using Conventional Commit messages.
   - `feat(...)` creates a minor release candidate.
   - `fix(...)` and `perf(...)` create patch release candidates.
   - `docs(...)`, `test(...)`, `chore(...)`, `ci(...)`, `build(...)`, and
     `refactor(...)` are included as supporting changes when relevant but do not
     normally trigger a release by themselves.
2. The **Release Please** workflow opens or updates a release pull request.
3. Review the release pull request. It updates:
   - `CHANGELOG.md`
   - `pyproject.toml`
   - `.release-please-manifest.json`
4. Because this project tracks `uv.lock`, run `uv lock` on the release pull
   request if the version changed, then commit the lockfile update to that same
   release PR.
5. Merge the release PR. release-please creates the version tag and GitHub
   Release.
6. The tag-triggered **Publish** workflow builds distributions, uploads them to
   the GitHub Release, and publishes to PyPI.

## Writing release-friendly commits

Release notes are generated from commit messages, so make user-facing commits
specific enough to stand alone in `CHANGELOG.md`.

Good examples:

```text
feat(cli): cap parallel multiplicity during build inference
fix(web): preserve existing results during incremental blob modeling
perf(model): reduce peak memory for large conformer sampling
```

Avoid vague implementation-only summaries such as `update main.py` or
`fix stuff`. If a change has important defaults, migration notes, or operational
impact, add a short commit body so release-please can include useful context.
