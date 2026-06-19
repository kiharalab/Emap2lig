---
name: emap2lig
description: >
  Ligand structure modeling from cryo-EM density maps. Use when working with
  cryo-EM maps (MRC files) to detect ligand density blobs and predict
  atomic structures. Covers the full Emap2lig inference pipeline (CLI) and
  the web GUI with Mol* 3D visualization. Triggers on mentions of
  emap2lig, cryo-EM ligand modeling, density map analysis, or requests to
  run the Emap2lig pipeline.
license: GPL-3.0
metadata:
  author: Kihara Lab
  repository: https://github.com/kiharalab/Emap2lig
  version: "0.3.4"
compatibility: >
  Requires Python 3.12, uv, and a CUDA-capable GPU for local inference.
  Web GUI: pip install "emap2lig[web]" or clone + uv sync --group web
  (pre-built dist/, no Node.js for normal use).
  Model weights download automatically from HuggingFace Hub on first run (network required).
---

# Emap2lig Skill

Detect and model ligands in cryo-EM density maps using the Emap2lig inference
pipeline. Two-stage architecture: (1) 3D CNN segmentation (MUNet) finds ligand
density blobs, (2) diffusion-based transformer (PairFormer + AtomDiffusion)
generates atomic coordinates for each blob.

## Quick Start

```bash
# Option A: Install as a CLI tool (globally available)
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig

# Then run directly
emap2lig \
  --input-map examples/emd_30556.map.gz \
  --output-dir outputs_30556 \
  --emdb-id 30556 \
  --gpu 0 \
  --detection-batch-size 4 \
  --ligand-list examples/emd_30556.yaml \
  --multiplicity 1

# --- or ---

# Option B: Clone the repo (development or local workflows)
git clone https://github.com/kiharalab/Emap2lig.git
cd Emap2lig
uv sync
uv run emap2lig --input-map examples/emd_30556.map.gz ...

# Option C: PyPI with Web GUI
pip install "emap2lig[web]"
emap2lig-gui
```

Output goes to `outputs_30556/` — see [Output Structure](#output-structure) below.


## When to Use CLI vs Web GUI

| Task | Recommended Tool | Reason |
|------|-----------------|--------|
| Batch / headless / server | **CLI** | Scriptable, no browser needed |
| Interactive exploration | **Web GUI** | Mol* 3D viewer, blob inspection |
| Quick prototyping | **Web GUI** | Visual feedback, per-blob assignment |
| Multi-conformer sampling | **CLI** | Better for parameter sweeps |
| Visualizing results | **Web GUI** | Mol* scene groups for maps, masks, structures |

## CLI Usage

### Full Pipeline

```bash
uv run emap2lig --input-map <MAP> --output-dir <OUTPUT> --ligand-list <LIGANDS_YAML> [OPTIONS]
```

> **Note**: If you used `uv tool install`, run `emap2lig` directly. If you
 > cloned the repo, prefix with `uv run` (e.g., `uv run emap2lig ...`).

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--input-map` | Path to cryo-EM map file (`.map.gz`, `.mrc`) |
| `--output-dir` | Output directory (default: `./output`) |
| `--ligand-list` | Path to ligand YAML file |

**Optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `0` | CUDA GPU device ID |
| `--detection-batch-size` | config default (16) | Batch size for sliding window inference |
| `--emdb-id` | — | EMDB ID for automatic contour level lookup |
| `--contour-level` | — | Manual contour level override |
| `--multiplicity` | `1` | Number of conformers per blob per ligand |
| `--seed` | `42` | Random seed |

### Fragment Detection Only

Use `fragment-detect` when you only need the segmentation stage (no structure
modeling):

```bash
uv run fragment-detect --input-map <MAP> [OPTIONS]
```

This runs the lightweight FragmentRegSeg model and saves probability maps.

### Step-by-Step Workflow

1. **Prepare input**: Get your cryo-EM map (MRC format) and create a ligand
   list YAML (see [Ligand List Format](#ligand-list-format) below).
2. **Run detection** (Stage 1): `emap2lig` segments the map into density blobs.
   Results go to `find_maps/` and `find_blobs/`.
3. **Run modeling** (Stage 2): The pipeline automatically proceeds to structure
   prediction for each blob. Results in `build_struct/`.
4. **Inspect outputs**: Check the `.cif` files in `build_struct/blob_*/` and
   the `build_struct/best/` directory for top predictions.

> **Tip**: Run detection first with `fragment-detect` if you want to preview
> blobs before committing to the full modeling pipeline.

## Web GUI Usage

### Starting the GUI

The Web GUI ships with pre-built `frontend/dist/`. Install the `web` extra from
PyPI or clone the repository for development.

```bash
# PyPI
pip install "emap2lig[web]"
emap2lig-gui

# Clone
uv sync --group web
uv run --group web emap2lig-gui
```

This starts the server at **http://localhost:40427** (configurable with `--port`).

**Options:**

| Option | Description |
|--------|-------------|
| `--port INT` | Port to serve on (default: 40427) |
| `--rebuild` | Force-rebuild frontend even if `dist/` exists |
| `--no-browser` | Don't auto-open browser |

### GUI Workflow

The GUI has four tabs:

**Setup tab:**
- Configure model cache path
- Download model weights (from HuggingFace)
- Select GPU device

**Find tab (Stage 1 — Detection):**
- Upload or specify path to cryo-EM map
- Set detection parameters (contour level, batch size)
- Submit detection job
- Visualize unified map and blob masks in Mol*
- Click **Build From These Blobs** to pass context to Build tab

**Build tab (Stage 2 — Modeling):**
- Uses detection results from Find tab
- Two ligand input modes:
  - **Global Ligands**: ligands apply to all detected blobs
  - **Per-Blob Assignment**: explicit blob→ligand mapping
- Submit modeling job
- View results in Mol* (cycle through conformers)
- Check rows to show/hide structures and masks

**Visualization tab:**
- Load existing output directories
- Inspect previously completed runs
- Upload additional structures/maps for overlay

### Running Headless

```bash
uv run --group web emap2lig-gui --no-browser --port 8080
```

The server is accessible at `http://localhost:8080` for remote access.

## Ligand List Format

Ligands are specified in a YAML file. Two types are supported:

### Simple Ligands (CCD codes)

```yaml
- CCD: HEM
- CCD: FAD
- CCD: NDP
- CCD: NAG
```

CCD codes are from the PDB Chemical Component Dictionary. Common examples:
`HEM` (heme), `FAD` (flavin adenine dinucleotide), `NAD`/`NDP` (nicotinamide
adenine dinucleotide), `NAG` (N-acetylglucosamine), `ATP`/`ADP`, `CO3`
(carbonate ion).

### Branched Ligands (glycans, linked residues)

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

The `BRANCHED` type defines a tree of residues connected by explicit bonds.
Each bond is `[residue_idx, atom_name, residue_idx, atom_name]`.

## Output Structure

```
output_dir/
├── preprocess/
│   ├── {stem}_unified.mrc     # Resampled map (1.0 Å/voxel)
│   └── ligands/               # LigandObject serialization
├── find_maps/
│   ├── backbone.mrc            # Backbone probability map
│   ├── sidechain.mrc           # Sidechain probability map
│   ├── ligand.mrc              # Ligand probability map
│   ├── ligand_mask.mrc         # Binary ligand mask
│   └── ...                     # Additional detection maps
├── find_blobs/                 # Detected density regions
│   ├── blob_N.npz              # DensityObject (cropped density + mask)
│   └── mask_N.mrc              # Binary mask per blob
└── build_struct/               # Predicted structures
    ├── blob_N/                 # Per-blob results
    │   ├── blob_N_LIG_N.cif    # Modeled structure (mmCIF format)
    │   ├── blob_N_LIG_N_pred_mask.mrc   # Predicted density mask
    │   ├── blob_N_results.csv  # Per-conformer scores
    │   └── blob_N_LIG_N_prompt.cmm      # ChimeraX visualization script
    └── best/                   # Top prediction per blob (by consistency IoU)
```

## Examples

### Basic run with EMDB ID (auto contour level)

```bash
uv run emap2lig \
  --input-map examples/emd_30556.map.gz \
  --output-dir outputs_30556 \
  --emdb-id 30556 \
  --gpu 0 \
  --detection-batch-size 4 \
  --ligand-list examples/emd_30556.yaml
```

### Branched ligand with multiplicity sampling

```bash
uv run emap2lig \
  --input-map examples/emd_7783.map.gz \
  --output-dir outputs_7783 \
  --emdb-id 7783 \
  --gpu 0 \
  --detection-batch-size 4 \
  --ligand-list examples/emd_7783.yaml \
  --multiplicity 5
```

### Using a manual contour level

```bash
uv run emap2lig \
  --input-map my_map.mrc \
  --output-dir my_output \
  --contour-level 0.035 \
  --ligand-list ligands.yaml
```

### Fragment detection only (no modeling)

```bash
uv run fragment-detect \
  --input-map examples/emd_30556.map.gz \
  --output-dir frag_output \
  --emdb-id 30556
```

## Tips and Common Issues

- **First run**: Model weights download automatically from HuggingFace
  (`KiharaLab/Emap2lig`) — needs internet. Subsequent runs are offline.
- **GPU memory**: Reduce `--detection-batch-size` (e.g., 4) if you hit CUDA
  OOM errors.
- **No GPU**: The pipeline supports CPU fallback but is significantly slower.
- **Multiplicity**: Higher values (>3) generate more conformers per blob but
  increase runtime linearly.
- **Blob limit**: The pipeline exits if >100 blobs are detected.
- **Contour level**: If you don't know the contour level, use `--emdb-id` to
  auto-fetch it from EMDB. For custom maps, try levels between 0.02–0.05.
- **Output inspection**: Use ChimeraX with the generated `.cmm` prompt files
  for interactive visualization of individual blob results.

## References

- [Detailed CLI Reference](references/CLI.md) — All commands, options, and
  programmatic API
- [Web GUI Reference](references/GUI.md) — Full GUI workflow, API endpoints,
  and troubleshooting
- [Project Repository](https://github.com/kiharalab/Emap2lig)
- [Model Weights (HuggingFace)](https://huggingface.co/KiharaLab/Emap2lig)

## Acknowledgements

Emap2lig builds upon several excellent open-source projects:

- **[Boltz](https://github.com/jwohlwend/boltz)** — Diffusion-based
  biomolecular interaction modeling framework.
- **[Mol\*](https://molstar.org/)** — Molecular visualization library used in
  the Web GUI.
- **[PyTorch](https://pytorch.org/)** — Deep learning framework.

## Citation

```bibtex
@article{li2026direct,
  title        = {Direct Detection and Atomic Modeling of Ligands in Cryo-EM Maps Using Deep Learning},
  author       = {Li, Shu and Jain, Anika and Kagaya, Yuki and Park, Joon Hong and Kihara, Daisuke},
  journal      = {bioRxiv},
  year         = {2026},
  doi          = {10.64898/2026.06.01.729423},
  url          = {https://www.biorxiv.org/content/10.64898/2026.06.01.729423v1},
  note         = {Preprint}
}
```
