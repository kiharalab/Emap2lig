# KiharaLab Web Server

Run Emap2lig without a local GPU or installation. Computation runs on Kihara Lab
servers; upload inputs in the browser and download results when jobs finish.

## Endpoints

| Stage | URL | Description |
|-------|-----|-------------|
| **Find** | [em.kiharalab.org/algorithm/Emap2lig-Find](https://em.kiharalab.org/algorithm/Emap2lig-Find) | Upload a cryo-EM map and detect ligand density blobs |
| **Build** | [em.kiharalab.org/algorithm/Emap2lig-Build](https://em.kiharalab.org/algorithm/Emap2lig-Build) | Upload Find outputs and generate atomic ligand structures |

## Workflow

1. **Find** — upload your map (see [Cryo-EM map](input-format.md#cryo-em-map) for
   supported formats). Download blob masks and related artifacts.
2. **Build** — upload the Find results plus ligand definitions (same YAML ideas as
   the [ligand list](input-format.md#ligand-list-yaml) used locally). Download
   modeled structures and scores.

## Compared to local runs

| | Web server | Local ([Usage](../README.md#usage)) |
|---|------------|--------------------------------------|
| Accelerator | Not required | Linux/CUDA or macOS/MPS required |
| Install | None | `uv` + Python 3.12 |
| Stages | Separate Find and Build pages | Full pipeline via CLI, Web GUI, or API |

For hardware, installation, and programmatic control, use the [local](installation.md)
guides instead.
