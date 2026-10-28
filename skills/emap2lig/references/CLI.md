---
title: Emap2lig CLI Reference
---

# Emap2lig CLI Reference

Detailed reference for the Emap2lig command-line interface, including the
programmatic API and config file options.

## Commands

### `emap2lig` — Full Inference Pipeline

```bash
uv run emap2lig [OPTIONS]
```

Runs both stages: (1) detect ligand density blobs, (2) model atomic structures
for each blob.

**All options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-map` | `TEXT` | *required* | Path to input cryo-EM map file |
| `--output-dir` | `TEXT` | `./output` | Output directory |
| `--gpu` | `INTEGER` | `0` | CUDA GPU device ID |
| `--detection-batch-size` | `INTEGER` | 16 | Batch size for sliding window |
| `--emdb-id` | `TEXT` | — | EMDB ID for contour level lookup |
| `--contour-level` | `FLOAT` | — | Manual contour level override |
| `--ligand-list` | `PATH` | required | Path to ligand list YAML |
| `--multiplicity` | `INTEGER` | `1` | Conformers per blob per ligand |
| `--seed` | `INTEGER` | `42` | Random seed |

### `fragment-detect` — Detection Only

```bash
uv run fragment-detect [OPTIONS]
```

Runs only the segmentation stage (FragmentRegSeg model). Saves probability
maps without structure modeling.

**All options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-map` | `TEXT` | *required* | Path to input cryo-EM map file |
| `--output-dir` | `TEXT` | `./output` | Output directory |
| `--gpu` | `INTEGER` | `0` | CUDA GPU device ID |
| `--detection-batch-size` | `INTEGER` | config default | Batch size for sliding window |
| `--emdb-id` | `TEXT` | — | EMDB ID for contour level lookup |
| `--contour-level` | `FLOAT` | — | Manual contour level override |

> **Tip**: Use `--help` with any command to see up-to-date options:
> `uv run emap2lig --help`

## Programmatic API (Python)

Import from the top-level package for scripting:

```python
from emap2lig import (
    InferenceConfig,
    detect_ligand_objects,
    parse_ligand_list,
    run_structure_modeling,
)

cfg = InferenceConfig(gpu=0, detection_batch_size=4, multiplicity=1)

# Stage 1: detect blobs (raises NoBlobsDetectedError / TooManyBlobsError)
blobs_dir = detect_ligand_objects("path/to/map.mrc", "./output", cfg, emdb_id="30556")

# Stage 2: model structures into the blobs under ./output
ligand_records = parse_ligand_list("ligands.yaml")
build_dir = run_structure_modeling("./output", ligand_records, cfg)
```

**Utility functions:**

| Function | Description |
|----------|-------------|
| `parse_ligand_list(path)` | Parse a YAML ligand list → `list[LigandRecord]` |
| `write_ligand_list(records, path)` | Write records back as a YAML ligand list |
| `prepare_ligands(records, ligands_dir)` | Featurize ligands into `<ligands_dir>/<name>.npz` |

## Configuration

There is no config file. Defaults live in `src/emap2lig/config.py`
(`InferenceConfig`); adjust them via CLI flags, the web GUI, or `InferenceConfig(...)`:

| Field | CLI Equivalent | Description |
|-------|---------------|-------------|
| `gpu` | `--gpu` | CUDA device ID (default: 0) |
| `detection_batch_size` | `--detection-batch-size` | Batch size for detection (default: 16) |
| `contour_level` | `--contour-level` | Manual contour level (default: auto) |
| `multiplicity` | `--multiplicity` | Conformers per blob per ligand (default: 1) |
| `max_parallel_multiplicity` | `--max-parallel-multiplicity` | Conformers per forward pass (default: 8) |
| `num_sampling_steps` | `--num-sampling-steps` | Diffusion sampling steps (default: 100) |

## Supported Input Formats

- **Map files**: `.mrc`, `.map.gz` (CCP4/MRC format)
- **Ligand list**: YAML (see SKILL.md for format)
- **EMDB contour levels**: Auto-fetched via `--emdb-id`
