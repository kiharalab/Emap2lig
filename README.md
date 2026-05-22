# Emap2lig

[![Kihara Lab](https://img.shields.io/badge/Kihara%20Lab-Purdue%20University-B1810B)](https://kiharalab.org/)
[![GitHub](https://img.shields.io/badge/GitHub-kiharalab%2FEmap2lig-181717?logo=github&logoColor=white)](https://github.com/kiharalab/Emap2lig)
[![HuggingFace Model](https://img.shields.io/badge/Model%20Weights-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/KiharaLab/Emap2lig)
<br/>
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://docs.python.org/3/) [![CUDA](https://img.shields.io/badge/CUDA-12%2F13-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit) [![Package manager: uv](https://img.shields.io/badge/Package%20manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/) [![Lint: Ruff](https://img.shields.io/badge/Lint-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/) [![Pre-commit: prek](https://img.shields.io/badge/Pre--commit-prek-FAB040?logo=pre-commit&logoColor=white)](https://github.com/j178/prek)

Official Emap2lig inference pipeline for finding ligand density blobs and building atomic ligand structures in cryo-EM maps.

- Stage 1 (**Find**): segment ligand density blobs from cryo-EM maps.
- Stage 2 (**Build**): generate ligand atomic coordinates from blobs.

> [!IMPORTANT]
> Emap2lig requires an NVIDIA CUDA GPU (CUDA 12/13). CPU inference is not supported.

## Hardware Requirements

- **GPU**: NVIDIA GPU with **8 GB+ VRAM**, Post-Ampere architecture (RTX 30xx / 40xx / 50xx series or higher)
- **CUDA**: CUDA 12 / 13 compatible driver
## Quick Start

```bash
# Install emap2lig as a CLI tool (globally available)
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig

# Run the pipeline
emap2lig \
  --input-map examples/emd_30556.map.gz \
  --output-dir outputs_30556 \
  --ligand-list examples/emd_30556.yaml \
  --emdb-id 30556
```

> **Web GUI**: The web GUI requires a different installation — see the [Web GUI section](#usage-web-gui) below.

## Sections (Toggle)

<details>
<summary><strong>Installation</strong></summary>

### Option A: Quick CLI (uv tool install)

Install emap2lig as a globally available CLI tool. No cloning needed.

```bash
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig
```

After installation, the `emap2lig` and `fragment-detect` commands are
available on your PATH:

```bash
emap2lig --help
fragment-detect --help
```

To update to the latest version:

```bash
uv tool install --from git+https://github.com/kiharalab/Emap2lig emap2lig --reinstall
```

### Option B: Clone and develop (full source)

Use this if you want to modify the source code or run the Web GUI.

```bash
git clone https://github.com/kiharalab/Emap2lig.git
cd Emap2lig
uv sync
```

Run CLI commands from the repo:

```bash
uv run emap2lig --help
uv run fragment-detect --help
```

### Web GUI (requires clone)

The Web GUI needs extra Python dependencies and Node.js for the frontend build:

```bash
git clone https://github.com/kiharalab/Emap2lig.git
cd Emap2lig
uv sync --group web
```

Then start the server:

```bash
uv run --group web python app/start.py
```

See [Usage: Web GUI](#usage-web-gui) below for details.

</details>

<details open>
<summary><strong>Usage: CLI</strong></summary>

### Main Command

```bash
uv run emap2lig --input-map <MAP> --output-dir <OUTPUT_DIR> --ligand-list <LIGANDS_YAML> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|---|---|
| `--input-map` | Path to cryo-EM map file (`.map.gz`, `.mrc`) |
| `--output-dir` | Directory to save output files (default: `./output`) |
| `--ligand-list` | Path to ligand YAML file |

### Optional Arguments

| Argument | Description |
|---|---|
| `--gpu` | CUDA GPU device ID (default: `0`) |
| `--detection-batch-size` | Sliding-window batch size for detection (default config: `16`) |
| `--emdb-id` | EMDB ID for automatic contour level lookup |
| `--contour-level` | Manual contour level |
| `--multiplicity` | Number of conformers per ligand-blob pair (default: `1`) |
| `--seed` | Random seed (default: `42`) |

Use either `--contour-level` or `--emdb-id` for best map normalization.

### Example Commands

```bash
# Simple ligands
uv run emap2lig --input-map examples/emd_30556.map.gz --gpu 0 --emdb-id 30556 --detection-batch-size 4 --output-dir outputs_30556 --ligand-list examples/emd_30556.yaml

# Branched ligands
uv run emap2lig --input-map examples/emd_7783.map.gz --gpu 0 --emdb-id 7783 --detection-batch-size 4 --output-dir outputs_7783 --ligand-list examples/emd_7783.yaml --multiplicity 1
```

</details>

<details>
<summary><strong>Usage: Web GUI</strong></summary>

A browser-based GUI is available in `app/` with Mol* visualization. **Note**: The web GUI requires cloning the repository — it is not available via `uv tool install` since it needs npm/frontend build steps.

### Prerequisites

- Python `3.12+` with `uv`
- Node.js `18+` and npm (frontend build)

### Install

```bash
uv sync --group web
```

### Run

```bash
uv run --group web python app/start.py
```

Then open `http://localhost:40427`.

### Customizing Ports and Headless Mode

```bash
# Custom port
uv run --group web python app/start.py --port 8080

# Headless mode (no browser auto-open, useful for servers)
uv run --group web python app/start.py --no-browser

# Combine options
uv run --group web python app/start.py --port 9000 --no-browser
```

| Option | Description |
|---|---|
| `--port INT` | Port to serve on (default: `40427`) |
| `--no-browser` | Don't auto-open the browser |
| `--rebuild` | Force-rebuild the frontend even if `dist/` exists |

### GUI Workflow

- **Setup**: model cache, download, GPU selection.
- **Find**: run detection and inspect blobs.
- **Build**: assign ligands and run structure modeling.
- **Visualization**: inspect existing output directories.

</details>

<details>
<summary><strong>Ligand List Format (YAML)</strong></summary>

### Supported Types

1. CCD ligands
```yaml
- CCD: ATP
```

2. SMILES ligands
```yaml
- SMILES: CCO
```

3. BRANCHED ligands
```yaml
- BRANCHED:
    residues:
      - "1. NAG"
      - "2. NAG"
    bonds:
      - [1, "C1", 2, "O4"]
```

### Example Files

Simple ligands (`examples/emd_30556.yaml`):

```yaml
- CCD: HEM
- CCD: FAD
- CCD: NDP
- CCD: NAG
```

Branched ligands (`examples/emd_7783.yaml`):

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

### Optional Blob Metadata

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

Format notes:
- `residues`: `"index. three-letter-code"`
- `bonds`: `[res1_idx, atom1, res2_idx, atom2]`
- `blob_id`: optional mapping to detected blobs (also accepts `blobs` as alias)

</details>

<details>
<summary><strong>Output Structure</strong></summary>

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

`blob_N_results.csv` columns:
- `conformer_name`: `blob_N_{LIGAND}_{M}`
- `consistency_iou`: IoU score between predicted structure density and predicted mask (`0-1`, higher is better)

Key behavior:
- Generates `multiplicity` conformers per ligand-blob pair
- Ranks candidates by `consistency_iou`
- Writes per-conformer prompt markers to `blob_N/{conformer_name}_prompt.cmm`
  (`<marker_set name="{conformer_name}_prompt">` in each CMM)
- Copies the top result per blob into `build_struct/best/`
- Exits early when more than 100 blobs are detected

</details>

<details>
<summary><strong>Programmatic API</strong></summary>

```python
from emap2lig.main import (
    detect_ligand_objects,
    load_config,
    parse_ligand_list,
    run_structure_modeling,
)

cfg = load_config(gpu=0)
status, blobs_dir = detect_ligand_objects("map.mrc", "output/", cfg, emdb_id="30556")
ligand_records = parse_ligand_list("ligands.yaml")
status = run_structure_modeling(
    blobs_dir,
    "output/",
    ligand_records,
    cfg,
    gpu=0,
    multiplicity=4,
)
```

</details>

<details>
<summary><strong>Fragment Detection (Optional)</strong></summary>

Lightweight fragment detector for when you only need fragment-class maps (5-membered ring, 6-membered ring, etc.).

### Command

```bash
uv run fragment-detect --input-map <MAP> [OPTIONS]
```

### Options

| Option | Description |
|---|---|
| `--output-dir` | Output directory (default: `./output`) |
| `--gpu` | CUDA GPU device ID (default: `0`) |
| `--detection-batch-size` | Sliding-window batch size |
| `--emdb-id` | EMDB ID for automatic contour level |
| `--contour-level` | Manual contour level |

### Outputs

```text
{stem}_frag_{label}.mrc        # probability map per fragment class
{stem}_frag_{label}_mask.mrc   # binary mask per fragment class
```

</details>

<details>
<summary><strong>Model Weights (HuggingFace Hub)</strong></summary>

Model weights, CCD reference data, and license are hosted on HuggingFace Hub:

| Resource | HF File | Used By |
|---|---|---|
| Detection model | `emap2lig-find-v0.0.1.safetensors` | `MUNetRegSeg` (map segmentation) |
| Fragment model | `emap2lig-frag.safetensors` | `FragmentRegSeg` (fragment-only detection) |
| Structure model | `emap2lig-build-v0.0.1.safetensors` | `Emap2lig` (diffusion-based modeling) |
| CCD dictionary | `ccd/ccd_dict_250523.pkl` | `get_ccd_dict()` (reference conformers) |
| License | `LICENSE.md` | Academic and Non-Profit Research License ⚠️

### Repository

```
https://huggingface.co/KiharaLab/Emap2lig
```

Files are automatically downloaded on first run to `~/.emap2lig/models/` via `huggingface_hub`.

### License

> [!IMPORTANT]
 > The **model weights** on HuggingFace are governed by a custom **Academic and Non-Profit Research License Agreement** — this is **separate and different** from the code license.
 >
 > - **Source code** (this repository): [GNU General Public License v3.0 (GPL-3.0)](LICENSE)
 > - **Model weights** (HuggingFace Hub): [Academic and Non-Profit Research License](https://huggingface.co/KiharaLab/Emap2lig/blob/main/LICENSE.md)
 >
 > The model weights are **not** licensed under GPL-3.0. If you use the model weights (via inference or fine-tuning), the HuggingFace license terms apply. Please review both licenses carefully.

### Syncing Updates

When releasing new model weights or CCD data:

1. Upload files to [KiharaLab/Emap2lig](https://huggingface.co/KiharaLab/Emap2lig) on HuggingFace.
2. Update the corresponding `filename` / `repo_id` in `src/emap2lig/emap2lig.yaml` (3 places: `detection_model`, `fragment_detection_model`, `model`).
3. Update `REPO_ID` in `src/emap2lig/main.py` and `repo_id` in `src/emap2lig/data/ccd.py`.
4. Update the CCD dictionary date string in `src/emap2lig/data/ccd.py` (`get_ccd_dict(date="...")`) if a new CCD version is uploaded.

</details>

## Latest Updates

- **2026-05-22: uv Tool Installation**
  - Emap2lig can now be installed globally via `uv tool install` — no cloning
    needed for CLI usage.
  - Added [Agent Skill](skills/emap2lig/) following the agentskills.io
    specification for AI-agent-guided usage.

- **2026-01-12: v0.3.1 Release**
  - Detection model update.
  - Per-blob ligand assignment in Web GUI.
  - Web GUI tutorial system.

- **2025-11-05: v0.3.0 Release**
  - Initial public release with CLI and Web GUI.
  - Two-stage pipeline: MUNet segmentation + PairFormer/AtomDiffusion modeling.

## Acknowledgements

Emap2lig builds upon and is inspired by several excellent open-source projects:

- **[Boltz](https://github.com/jwohlwend/boltz)** (Wohlwend et al.) — A
  diffusion-based biomolecular interaction modeling framework. Emap2lig's
  structure prediction approach is inspired by diffusion-based modeling
  techniques pioneered in the Boltz family of models.

- **[Mol\*](https://molstar.org/)** (Sehnal et al.) — An open-source
  molecular visualization library used for 3D rendering of cryo-EM maps and
  predicted ligand structures in the Emap2lig Web GUI.

- **[Hugging Face Hub](https://huggingface.co/KiharaLab/Emap2lig)** — Model
  weight and data distribution platform.

If you use Emap2lig in your research, please cite our work (see below) and
the relevant dependencies above.

## Citation

If you use Emap2lig in your research, please cite the following:

```bibtex
TBD
```
