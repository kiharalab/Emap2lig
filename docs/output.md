# Output Structure

Default layout under `--output-dir`:

```text
output_dir/
├── preprocess/
│   ├── unified.mrc
│   └── ligands/
│       └── {LIGAND}.npz
├── find_maps/
│   ├── backbone.mrc
│   ├── sidechain.mrc
│   ├── sugar.mrc
│   ├── phosphate.mrc
│   ├── base.mrc
│   ├── ligand.mrc
│   └── ligand_mask.mrc
├── find_blobs/
│   ├── blob_N.npz
│   └── mask_N.mrc
└── build_struct/
    ├── blob_N/
    │   ├── blob_N_{LIG}_M.cif
    │   ├── blob_N_{LIG}_M_pred_mask.mrc
    │   ├── blob_N_{LIG}_M_prompt.cmm
    │   └── blob_N_results.csv
    └── best/
        └── blob_N_blob_N_{LIG}_M.cif
```

## Results CSV

`blob_N_results.csv` columns:

| Column | Description |
|--------|-------------|
| `conformer_name` | `blob_N_{LIGAND}_{M}` |
| `consistency_iou` | IoU between predicted structure density and mask (0–1, higher is better) |

## Behavior

- `--multiplicity` generates multiple conformers per ligand–blob pair
- Candidates are ranked by `consistency_iou`
- Per-conformer Chimera markers: `blob_N/{conformer_name}_prompt.cmm`
- Top result per blob is copied to `build_struct/best/`
- Pipeline exits early if more than **100** blobs are detected
