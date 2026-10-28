# Programmatic API

Everything is importable from the top-level package:

```python
from emap2lig import (
    InferenceConfig,
    detect_ligand_objects,
    parse_ligand_list,
    run_structure_modeling,
)

cfg = InferenceConfig(gpu=0, multiplicity=4, max_parallel_multiplicity=8)
blobs_dir = detect_ligand_objects("map.mrc", "output/", cfg, emdb_id="30556")
ligand_records = parse_ligand_list("ligands.yaml")
build_dir = run_structure_modeling("output/", ligand_records, cfg)
```

| Name | Stage | Returns |
|------|-------|---------|
| `InferenceConfig(...)` | Config | runtime options (validated on construction) |
| `detect_ligand_objects(input_map, output_dir, cfg, emdb_id=None)` | Find | `find_blobs/` directory |
| `parse_ligand_list(path)` / `write_ligand_list(records, path)` | Utility | `list[LigandRecord]` / path |
| `run_structure_modeling(output_dir, ligand_records, cfg, blob_ids=None)` | Build | `build_struct/` directory |
| `run_fragment_detection(input_map, output_dir, cfg, emdb_id=None)` | Fragments | output directory |
| `OutputLayout(root)` | Utility | paths of every output file |

`InferenceConfig` fields and defaults:

| Field | Default | Meaning |
|-------|---------|---------|
| `gpu` | `0` | Accelerator device ID (CUDA on Linux; `0`/`-1` select MPS on macOS) |
| `contour_level` | `None` | Map contour level; `None` fetches it from EMDB or estimates it |
| `detection_batch_size` | `16` | Sliding-window batch size of the find stage |
| `multiplicity` | `1` | Conformers per blob per ligand |
| `max_parallel_multiplicity` | `8` | Conformers generated per forward pass (peak memory) |
| `num_sampling_steps` | `100` | Diffusion sampling steps |

`run_structure_modeling` reads the blobs written by `detect_ligand_objects` into
the same `output_dir`. Pass `blob_ids` to re-model only some blobs with only the
given ligands.

## Errors

Pipeline failures raise subclasses of `emap2lig.Emap2ligError`:

| Exception | When |
|-----------|------|
| `NoBlobsDetectedError` | The find stage found no blob of usable size |
| `TooManyBlobsError` | More than 100 blobs (likely noise); blobs are still saved |
| `Emap2ligError` | No blobs on disk, or no ligand could be prepared, before building |

A single ligand that cannot be prepared (unknown CCD code, invalid SMILES) is
skipped with a warning (`LigandPreparationError` is logged, not raised).

Model architecture and checkpoints are fixed in `emap2lig.model.pretrained`.

See also [CLI reference](cli.md) and [Output structure](output.md).
