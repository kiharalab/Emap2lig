# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-12 (updated 2026-05-22)
**Branch:** main
**Release:** 0.4.2
**Repository:** https://github.com/kiharalab/Emap2lig

## OVERVIEW

Inference pipeline for ligand structure modeling from cryo-EM maps. Two-stage architecture: 3D CNN segmentation (MUNet) detects ligand density blobs, then diffusion-based transformer (PairFormer + AtomDiffusion) generates atomic coordinates. ~9.7k lines Python in `src/emap2lig/` (~11.8k total including `web/`), ~9.3k lines TypeScript. Includes a web GUI (FastAPI + React + Mol*).

**Reference**: Upstream training codebase is in the `reference/` submodule (no longer bundled — kept in git history).

## STRUCTURE

```
src/emap2lig/                    # Core inference pipeline (~9.7k lines)
├── main.py                      # CLI + programmatic API (1411 lines)
├── emap2lig.yaml                # Hydra config (at package root, not configs/)
├── frag.py                      # Fragment segmentation CLI (166 lines)
├── data/                        # IO, dataset, chemistry (see data/AGENTS.md)
│   ├── types.py                 # Core dataclasses: MapObject, LigandObject, DensityObject
│   ├── dataset.py               # LigandModelingDataset (579 lines, 149-dim atom features)
│   ├── map.py                   # MRC processing: resample, normalize, crop (512 lines)
│   ├── ccd.py                   # CCD retrieval (HuggingFace), ETKDG conformers
│   ├── simulate.py              # Numba-accelerated density simulation for IoU scoring
│   ├── const.py                 # Chemical constants (128 elements, 7 chiralities, 5 bond types)
│   ├── transforms.py            # Coordinate augmentation, atom name encoding
│   ├── download.py              # EMDB contour level fetching
│   └── io/                      # mmCIF/MRC readers/writers
├── model/                       # Neural network architecture (see model/AGENTS.md)
│   ├── model.py                 # Emap2lig LightningModule (461 lines)
│   ├── layers/                  # Primitive operations (see layers/AGENTS.md)
│   ├── modules/                 # Diffusion, conformer embedder, pairformer, instance seg (see modules/AGENTS.md)
│   └── seg/                     # MUNet segmentation (see seg/AGENTS.md)
├── web/                         # Web GUI — see web/AGENTS.md
│   ├── cli.py                   # Typer launcher (emap2lig-gui entry point)
│   ├── app.py                   # FastAPI app factory
│   ├── routers/                 # detect, model, jobs, files, download
│   ├── schemas.py               # Pydantic request/response models
│   ├── services.py              # Wrappers around emap2lig core functions
│   ├── state.py                 # In-memory job store
│   ├── results_scan.py          # On-disk output layout scanning
│   └── frontend/
│       └── dist/                # Pre-built static assets (source in shuuul/Emap2lig-web)
└── *.py                         # Utilities

examples/                        # Sample input files (EMD-30556, EMD-7783)
```

## ENTRY POINTS

```toml
# pyproject.toml [project.scripts]
emap2lig       = "emap2lig.main:app"                # Typer CLI — full pipeline
fragment-detect = "emap2lig.frag:app"               # Fragment segmentation only
emap2lig-gui   = "emap2lig.web.cli:cli"            # Web GUI launcher
```

Note: `extract-yaml` entry point was removed — use the PDB→YAML utility from the upstream training codebase if needed.

## PROGRAMMATIC API (main.py)

| Function | Stage | Signature |
|----------|-------|-----------|
| `load_config(gpu, detection_batch_size, contour_level)` | Config | → `cfg` (Hydra config) |
| `detect_ligand_objects(input_map, output_dir, cfg, emdb_id)` | Stage 1 | → `(status, blobs_dir)` |
| `run_structure_modeling(blobs_dir, output_dir, ligand_records, cfg, gpu, multiplicity)` | Stage 2 | → `status` |
| `parse_ligand_list(ligand_list_path)` | Utility | → `list[LigandRecord]` |
| `prepare_ligand_dataset(ligand_records, output_dir, ligands_dir)` | Utility | → `(status, ligands_dir)` |

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| User documentation | `docs/` | Installation, CLI, Web GUI, input/output formats |
| Run inference | `uv run emap2lig --input-map X --output-dir Y --ligand-list Z` | Typer CLI |
| Fragment detection | `uv run fragment-detect --input-map X` | Lightweight segmentation |
| Model architecture | `src/emap2lig/model/model.py` | Emap2lig LightningModule |
| Segmentation model | `src/emap2lig/model/seg/model.py` | MUNetRegSeg / FragmentRegSeg |
| Data pipeline | `src/emap2lig/data/dataset.py` | 149-dim atom features |
| Core data types | `src/emap2lig/data/types.py` | NumpySerializable, structured arrays |
| Map processing | `src/emap2lig/data/map.py` | Resample to 1.0 Å/voxel |
| Diffusion sampling | `src/emap2lig/model/modules/diffusion.py` | EDM-based, 20 steps default |
| Web GUI | `src/emap2lig/web/` | `uv run emap2lig-gui` — see `web/AGENTS.md` |
| Model weights | `~/.emap2lig/models/` | HuggingFace `KiharaLab/Emap2lig` |

## COMMANDS

```bash
# Install
uv sync

# Install with web GUI dependencies
uv sync --group web

# Lint & format
uv run ruff check --fix && uv run ruff format

# Pre-commit (ruff, ruff-format, pyupgrade, trailing-whitespace, uv-lock)
prek --all-files

# Run web GUI (uses pre-built frontend/dist/ on http://localhost:40427)
uv run --group web emap2lig-gui
```

## CONVENTIONS (THIS PROJECT)

- **Python**: 3.12 (uv-managed, uv_build build backend)
- **Version source of truth**: `pyproject.toml` (`[project].version`) for the Python package; align `Emap2lig-web` `package.json` before a release. After changing the Python version, run `uv lock`. Do not hand-edit generated frontend assets for version changes — merge the CI sync PR from `shuuul/Emap2lig-web`. Only `web/frontend/dist/` is included in published packages.
- **Commits**: do not add AI/agent co-author trailers (for example `Co-authored-by: Amp ...`) to commit messages.
- **Formatter**: Ruff — 88 chars, double quotes, space indent
- **Types**: Use `|` for unions, lowercase `list`/`dict`
- **Imports**: Ruff isort (`combine-as-imports = true`, `split-on-trailing-comma = true`)
- **Lint rules**: `E`, `F`, `B`, `UP`, `RUF` selected
- **Lint ignores**: `E501` (line length), `B006`/`B008` (mutable defaults), `F821`/`F722` (jaxtyping annotations), `B905`, `RUF001-003` (unicode), `F841` (unused vars), `E731` (lambda), `E741` (ambiguous names)
- **Precision**: `tf32` on CUDA, BF16 mixed precision
- **Serialization**: NPZ for data objects (`NumpySerializable`), safetensors for model weights
- **CI**: GitHub Actions for lint (PR/push) and PyPI publish (tags); frontend dist sync PRs come from `shuuul/Emap2lig-web`

## ANTI-PATTERNS (THIS PROJECT)

- `type: ignore`: 10+ occurrences in data handling
- Hardcoded magic numbers in distogram boundaries
- `strict=False` in `load_state_dict()` — may hide structural drift
- Model storage path hardcoded to `~/.emap2lig/models/` (no env var override)

## OUTPUT STRUCTURE

```
output_dir/
├── preprocess/
│   ├── {stem}_unified.mrc        # Resampled map (1.0 Å/voxel)
│   └── ligands/
│       └── {LIGAND}.npz          # LigandObject serialization
├── find_maps/
│   ├── backbone.mrc              # Backbone probability map
│   ├── sidechain.mrc             # Sidechain probability map
│   ├── ligand.mrc                # Ligand probability map
│   ├── ligand_mask.mrc           # Binary ligand mask
│   └── ...                       # Additional detection label maps
├── find_blobs/                   # Detected density regions
│   ├── blob_N.npz               # DensityObject (cropped density + instance mask)
│   └── mask_N.mrc               # Binary mask per blob
└── build_struct/                 # Predicted structures
    ├── blob_N/                   # Per-blob: CIF + pred_mask.mrc + results.csv + prompt.cmm
    └── best/                     # Top prediction per blob (by consistency IoU)
```

## NOTES

- Multiplicity sampling: generates N conformations per blob
- Performance limit: exits if >100 blobs detected
- Model weights downloaded from HuggingFace Hub on first run
- Tests require network access (HF Hub checkpoint download)
