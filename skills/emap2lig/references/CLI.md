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
| `--detection-batch-size` | `INTEGER` | config default (16) | Batch size for sliding window |
| `--emdb-id` | `TEXT` | — | EMDB ID for contour level lookup |
| `--contour-level` | `FLOAT` | — | Manual contour level override |
| `--ligand-list` | `TEXT` | — | Path to ligand list YAML |
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

Import from `emap2lig.main` for scripting:

```python
from emap2lig.main import load_config, detect_ligand_objects, run_structure_modeling

# Load config
cfg = load_config(gpu=0, detection_batch_size=4, contour_level=None)

# Stage 1: detect blobs
status, blobs_dir = detect_ligand_objects(
    input_map="path/to/map.mrc",
    output_dir="./output",
    cfg=cfg,
    emdb_id="30556",
)

# Stage 2: model structures
ligand_records = [{"ccd": "HEM"}, {"ccd": "FAD"}]
status = run_structure_modeling(
    blobs_dir=blobs_dir,
    output_dir="./output",
    ligand_records=ligand_records,
    cfg=cfg,
    gpu=0,
    multiplicity=1,
)
```

**Utility functions:**

| Function | Description |
|----------|-------------|
| `parse_ligand_list(path)` | Parse a YAML ligand list → `list[LigandRecord]` |
| `prepare_ligand_dataset(records, output_dir, ligands_dir)` | Prepare ligand data for modeling |

## Config File

The Hydra config file is at `src/emap2lig/emap2lig.yaml` in the repository.
Key settings that can be overridden at runtime:

| Config Key | CLI Equivalent | Description |
|------------|---------------|-------------|
| `gpu` | `--gpu` | CUDA device ID |
| `detection_batch_size` | `--detection-batch-size` | Batch size for detection |
| `spatial_size` | — | Crop size for sliding window (default: 48) |

## Supported Input Formats

- **Map files**: `.mrc`, `.map.gz` (CCP4/MRC format)
- **Ligand list**: YAML (see SKILL.md for format)
- **EMDB contour levels**: Auto-fetched via `--emdb-id`
