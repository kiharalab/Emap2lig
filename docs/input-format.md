# Input Formats

## Cryo-EM map

CLI flag: `--input-map`

| Format | Notes |
|--------|--------|
| `.mrc` | Standard MRC2014; orthogonal cell (90° angles) |
| `.map.gz` | Gzip-compressed map |

The pipeline reads the map via `mrcfile`, then:

1. **Resamples** to **1.0 Å/voxel** (`preprocess/unified.mrc`)
2. **Normalizes** using a contour level (see [Map normalization](#map-normalization))
3. **Crops** to density content for detection

Non-orthogonal maps are rejected at read time.

Web GUI and [KiharaLab web server](web-server.md) accept the same map types through
their upload interfaces.

## Ligand list (YAML)

CLI flag: `--ligand-list`

A YAML **list** of ligand entries. Each item is one of: **CCD**, **SMILES**, or
**BRANCHED**.

### CCD ligands

```yaml
- CCD: ATP
```

Uses the Chemical Component Dictionary (reference conformers from HuggingFace).

### SMILES ligands

```yaml
- SMILES: CCO
```

RDKit builds 3D coordinates from the SMILES string.

### Branched ligands

```yaml
- BRANCHED:
    residues:
      - "1. NAG"
      - "2. NAG"
    bonds:
      - [1, "C1", 2, "O4"]
```

- `residues`: `"index. three-letter-code"`
- `bonds`: `[res1_idx, atom1, res2_idx, atom2]`

### Example files

Simple (`examples/emd_30556.yaml`):

```yaml
- CCD: HEM
- CCD: FAD
- CCD: NDP
- CCD: NAG
```

Branched (`examples/emd_7783.yaml`):

```yaml
- BRANCHED:
    residues:
      - 1. NAG
      - 2. NAG
    bonds:
      - [2, "C1", 1, "O4"]
- CCD: NAG
- CCD: CO3
```

### Optional blob assignment

After Find, restrict which blobs each ligand uses:

```yaml
- CCD: ATP
  blob_id: 1

- SMILES: CCO
  blob_id: [2, 3]

- BRANCHED:
    residues:
      - "1. NAG"
      - "2. NAG"
    bonds:
      - [1, "C1", 2, "O4"]
  blob_id: [4, 5, 6]
```

`blob_id` accepts a single integer or a list. Alias: `blobs`.

## Map normalization

Contour level controls density scaling before detection. Provide **one** of:

| CLI flag | Behavior |
|----------|----------|
| `--emdb-id` | Fetch recommended contour level from EMDB (e.g. `30556`) |
| `--contour-level` | Use a numeric level directly |

If neither is set, the pipeline still resamples to 1.0 Å/voxel but uses default
normalization without an EMDB contour.

Web GUI exposes the same choices in the Find run options panel.
