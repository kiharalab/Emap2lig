# DATA MODULE KNOWLEDGE BASE

**Last Reviewed:** 2026-05-22
**Scope Note:** Updated to reflect current codebase state.

## OVERVIEW
Data handling for cryo-EM maps, ligand chemistry, and tokenization for structural modeling.

## KEY COMPONENTS
- `types.py`: Core dataclasses for molecules (`LigandObject`, `DensityObject`, `MapObject`, `CIFObject`, `LigandRecord`). Implements `NumpySerializable` for NPZ-based persistence. Defines structured Numpy arrays for atomic features (element, charge, coords, chirality, ring) and bond features (type, ring membership).
- `dataset.py`: `LigandModelingDataset` implementation for tokenizing ligands into graph representations using RDKit and Gemmi. Handles mapping localized density regions (blobs) to neural network inputs with 149-dim atom features and 9-dim bond features.
- `map.py`: Standardized MRC map processing pipeline including resampling (to 1.0 Å/voxel), normalization via EMDB contour levels, and automated spatial cropping based on content density.
- `ccd.py`: Interface for Chemical Component Dictionary (CCD) retrieval from Hugging Face and RDKit-based 3D coordinate generation using ETKDG (v2/v3) methods for reference conformers.
- `simulate.py`: Numba-accelerated density simulation from atomic coordinates using a binary sphere kernel. Used for structural verification and Mask IoU-based scoring of predicted models.
- `io/mmcif.py`: High-fidelity mmCIF generation via `modelcif` and `ihm`. Ensures output compliance with PDBx/mmCIF standards for modeling metadata and entity/assembly definitions.
- `io/map.py`: Low-level MRC file I/O using `mrcfile`. Handles voxel size extraction, origin detection, and reordering axes to standard ZYX orientation for consistency.
- `io/writer.py`: PyTorch Lightning `BasePredictionWriter` for batch-wise serialization of predicted ligand structures (mmCIF), simulated density maps (MRC), Chimera marker files (CMM), and confidence summary tables (CSV).
- `const.py`: Hardcoded chemical constants including atomic numbers, bond type definitions, and chirality types used for one-hot encoding of chemical features.
- `download.py`: Utility for fetching contour levels from EMDB servers to automate map normalization during the data preparation phase.
- `transforms.py`: Coordinate augmentation and atom name encoding utilities for training-time data transformation.

## FILE FORMATS
- `.mrc`/`.map`: Cryo-EM density maps (standardized to 1.0 Å/voxel, float32 grid data). Handled via `mrcfile` with orientation and origin normalization.
- `.cif`: Standard mmCIF for structural outputs, entity metadata, and parsing reference protein/ligand data. Compatible with modern visualizers like PyMOL and ChimeraX.
- `.npz`: Compressed storage for serialized `DensityObject` (segmented blobs) and `LigandObject` (molecule graph and chemical features) for efficient I/O.
- `.pkl`: Cached CCD dictionaries and pre-computed molecular conformer metadata for ligand references, typically downloaded and updated from Hugging Face Hub.
- `.yaml`: Input ligand list format supporting CCD codes, SMILES strings, and complex branched residue definitions (e.g., oligosaccharides or glycosylation).
- `.csv`: Sorted confidence summary tables aggregating Mask IoU scores for multiple generated conformations per blob, providing a ranked list of predictions.
- `.cmm`: Chimera marker files with per-conformer prompt markers for visualization in UCSF ChimeraX.

## TECH STACK
- **Chemistry**: RDKit, Gemmi, pdbeccdutils
- **Cryo-EM**: mrcfile, Gemmi (mmCIF parsing only)
- **Formatters**: modelcif, ihm
- **Performance**: Numba, Numpy, Jaxtyping
