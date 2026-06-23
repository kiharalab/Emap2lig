# Programmatic API

Import from `emap2lig.main`:

```python
from emap2lig.main import (
    detect_ligand_objects,
    load_config,
    parse_ligand_list,
    run_structure_modeling,
)

cfg = load_config(gpu=0)
status, blobs_dir = detect_ligand_objects(
    "map.mrc", "output/", cfg, emdb_id="30556"
)
ligand_records = parse_ligand_list("ligands.yaml")
status = run_structure_modeling(
    blobs_dir,
    "output/",
    ligand_records,
    cfg,
    gpu=0,
    multiplicity=4,
    max_parallel_multiplicity=8,
)
```

| Function | Stage | Returns |
|----------|-------|---------|
| `load_config(gpu, ...)` | Config | Hydra `cfg` |
| `detect_ligand_objects(input_map, output_dir, cfg, emdb_id)` | Find | `(status, blobs_dir)` |
| `parse_ligand_list(path)` | Utility | `list[LigandRecord]` |
| `run_structure_modeling(blobs_dir, output_dir, ligand_records, cfg, gpu, multiplicity, max_parallel_multiplicity=8)` | Build | `status` |

`max_parallel_multiplicity` only limits how many conformers are generated in one
forward pass; `multiplicity` remains the total conformer count. Results are
written to the same output layout and sorted normally after inference.

See also [CLI reference](cli.md) and [Output structure](output.md).
