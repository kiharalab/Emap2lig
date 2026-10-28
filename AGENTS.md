# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-12 (updated 2026-05-22)
**Branch:** main
**Release:** 0.5.0
**Repository:** https://github.com/kiharalab/Emap2lig

## OVERVIEW

Inference pipeline for ligand structure modeling from cryo-EM maps. Two-stage architecture: 3D CNN segmentation (MUNet) detects ligand density blobs, then diffusion-based transformer (PairFormer + AtomDiffusion) generates atomic coordinates. ~9.5k lines Python in `src/emap2lig/` (~11.5k total including `web/`), ~9.3k lines TypeScript. Includes a web GUI (FastAPI + React + Mol*).

**Reference**: Upstream training codebase is in the `reference/` submodule (no longer bundled — kept in git history).

## STRUCTURE

```
src/emap2lig/                    # Core inference pipeline
├── __init__.py                  # Lazy public API (InferenceConfig, pipeline stages, errors)
├── config.py                    # InferenceConfig (user options) + fixed checkpoint constants
├── errors.py                    # Emap2ligError, NoBlobsDetectedError, TooManyBlobsError, ...
├── layout.py                    # OutputLayout: every path of an output directory
├── pipeline/                    # Stage orchestration (no CLI/web knowledge)
│   ├── preprocess.py            # load_unified_map (resample/normalize + save)
│   ├── detect.py                # Find: segmentation → find_maps/ → find_blobs/
│   ├── build.py                 # Build: ligands + blobs → predict loop → build_struct/
│   ├── ranking.py               # rank_conformers: sort CSVs, copy best CIFs
│   └── fragments.py             # run_fragment_detection
├── cli/                         # Typer entry points: main (emap2lig), frag, gui
├── utils/                       # storage (EMAP2LIG_HOME), weights (HF), device, runtime, log
├── data/                        # IO, dataset, chemistry (see data/AGENTS.md)
│   ├── types.py                 # Core dataclasses: MapObject, LigandObject, DensityObject
│   ├── ligand_list.py           # Ligand list YAML ↔ LigandRecord
│   ├── featurize.py             # LigandRecord → LigandObject (atom/bond features)
│   ├── dataset.py               # LigandModelingDataset (149-dim atom features)
│   ├── map.py                   # MRC processing: resample, normalize, crop
│   ├── ccd.py                   # CCD retrieval (HuggingFace), ETKDG conformers
│   ├── simulate.py              # Vectorized NumPy density simulation for IoU scoring
│   ├── const.py                 # Chemical constants (128 elements, 7 chiralities, 5 bond types)
│   ├── transforms.py            # Coordinate augmentation, atom name encoding
│   ├── download.py              # EMDB contour level fetching
│   └── io/                      # mmCIF/MRC readers/writers
├── model/                       # Neural network architecture (see model/AGENTS.md)
│   ├── model.py                 # Emap2lig nn.Module
│   ├── pretrained.py            # Factories for the released checkpoints
│   ├── layers/                  # Primitive operations (see layers/AGENTS.md)
│   ├── modules/                 # Diffusion, conformer embedder, pairformer, instance seg
│   └── seg/                     # SlidingWindowSegmenter → MUNetRegSeg / FragmentRegSeg
└── web/                         # Web GUI (FastAPI + pre-built frontend/dist/) — see web/AGENTS.md

examples/                        # Sample input files (EMD-30556, EMD-7783)
```

## ENTRY POINTS

```toml
# pyproject.toml [project.scripts]
emap2lig        = "emap2lig.cli.main:app"           # Typer CLI — full pipeline
fragment-detect = "emap2lig.cli.frag:app"           # Fragment segmentation only
emap2lig-gui    = "emap2lig.cli.gui:cli"            # Web GUI launcher
```

Note: `extract-yaml` entry point was removed — use the PDB→YAML utility from the upstream training codebase if needed.

## PROGRAMMATIC API (`from emap2lig import ...`)

| Function | Stage | Signature |
|----------|-------|-----------|
| `InferenceConfig(gpu, contour_level, detection_batch_size, multiplicity, ...)` | Config | validated dataclass |
| `detect_ligand_objects(input_map, output_dir, cfg, emdb_id=None)` | Stage 1 | → `find_blobs/` path; raises `NoBlobsDetectedError` / `TooManyBlobsError` |
| `run_structure_modeling(output_dir, ligand_records, cfg, blob_ids=None)` | Stage 2 | → `build_struct/` path; raises `Emap2ligError` |
| `parse_ligand_list(path)` / `write_ligand_list(records, path)` | Utility | ↔ `list[LigandRecord]` |
| `prepare_ligands(records, ligands_dir)` | Utility | → written `.npz` paths (failures skipped) |

Stages raise exceptions (never `sys.exit`); `cli/common.run_stage` maps
`Emap2ligError` to exit code 1 and the web services mark the job failed.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| User documentation | `docs/` | Installation, CLI, Web GUI, input/output formats |
| Run inference | `uv run emap2lig --input-map X --output-dir Y --ligand-list Z` | `cli/main.py` → `pipeline/` |
| Fragment detection | `uv run fragment-detect --input-map X` | Lightweight segmentation |
| Model architecture | `src/emap2lig/model/model.py` | Emap2lig nn.Module |
| Segmentation model | `src/emap2lig/model/seg/model.py` | SlidingWindowSegmenter base; MUNetRegSeg / FragmentRegSeg |
| Data pipeline | `src/emap2lig/data/dataset.py` | 149-dim atom features |
| Core data types | `src/emap2lig/data/types.py` | NumpySerializable, structured arrays |
| Map processing | `src/emap2lig/data/map.py` | Resample to 1.0 Å/voxel |
| Diffusion sampling | `src/emap2lig/model/modules/diffusion.py` | EDM-based, 20 steps default |
| Web GUI | `src/emap2lig/web/` | `uv run emap2lig-gui` — see `web/AGENTS.md` |
| Model weights | `~/.emap2lig/models/` (`utils/storage.py`, `utils/weights.py`, `EMAP2LIG_HOME`) | HuggingFace `KiharaLab/Emap2lig` |

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

- **Python**: 3.12–3.14 (uv-managed, uv_build build backend)
- **Version source of truth**: `pyproject.toml` (`[project].version`) for the Python package; align `Emap2lig-web` `package.json` before a release. After changing the Python version, run `uv lock`. Do not hand-edit generated frontend assets for version changes — merge the CI sync PR from `shuuul/Emap2lig-web`. Only `web/frontend/dist/` is included in published packages.
- **Commits**: do not add AI/agent co-author trailers (for example `Co-authored-by: Amp ...`) to commit messages.
- **Formatter**: Ruff — 88 chars, double quotes, space indent
- **Types**: Use `|` for unions, lowercase `list`/`dict`
- **Imports**: Ruff isort (`combine-as-imports = true`, `split-on-trailing-comma = true`)
- **Lint rules**: `E`, `F`, `B`, `UP`, `RUF` selected
- **Lint ignores**: `E501` (line length), `B006`/`B008` (mutable defaults), `B905`, `RUF001-003` (unicode), `F841` (unused vars), `E731` (lambda), `E741` (ambiguous names)
- **Precision**: FP32 inference (TF32 matmuls on CUDA via `utils/runtime.py`), no autocast
- **Serialization**: NPZ for data objects (`NumpySerializable`), safetensors for model weights
- **CI**: GitHub Actions for lint (PR/push) and PyPI publish (tags); frontend dist sync PRs come from `shuuul/Emap2lig-web`

## ANTI-PATTERNS (THIS PROJECT)

- `type: ignore`: 10+ occurrences in data handling
- Hardcoded magic numbers in distogram boundaries
- `strict=False` in `load_state_dict()` — may hide structural drift

## OUTPUT STRUCTURE

```
output_dir/
├── preprocess/
│   ├── unified.mrc               # Resampled map (1.0 Å/voxel)
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
- Performance limit: `TooManyBlobsError` (exit code 1) if >100 blobs detected
- Model weights downloaded from HuggingFace Hub on first run
- Tests require network access (HF Hub checkpoint download)
