#!/usr/bin/env python3
"""Emap2lig CLI and programmatic inference API."""

import csv
import platform
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from loguru import logger
from rdkit import Chem
from skimage import measure
from torch.utils.data import DataLoader
import typer

import rdkit
from huggingface_hub import hf_hub_download

from emap2lig.data.ccd import CCDFetchError, get_ccd_mol, get_conformer_from_smiles
from emap2lig.data.const import bond_type_ids, chirality_type_ids
from emap2lig.data.dataset import LigandModelingDataset, collate_fn
from emap2lig.data.io.map import parse_mrc, to_mrc
from emap2lig.data.io.writer import LigandWriter
from emap2lig.data.map import crop_mrcs, get_unified_mrc
from emap2lig.data.types import (
    Atom,
    Bond,
    DensityObject,
    LigandObject,
    LigandRecord,
    MapObject,
)

# ---------------------------------------------------------------------------
# Model weights directory & metadata
# ---------------------------------------------------------------------------
MODEL_DIR = Path.home() / ".emap2lig" / "models"
REPO_ID = "KiharaLab/Emap2lig"
MODEL_FILES: dict[str, str] = {
    "detection_model": "emap2lig-find-v0.0.1.safetensors",
    "structure_model": "emap2lig-build-v0.0.1.safetensors",
}

# Set default pickle properties
pickle_option = rdkit.Chem.PropertyPickleOptions.AllProps
rdkit.Chem.SetDefaultPickleProperties(pickle_option)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 2048
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    torch.autograd.set_detect_anomaly(True)


def _mps_available() -> bool:
    return bool(
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _macos_version_at_least(major: int, minor: int) -> bool:
    version = platform.mac_ver()[0]
    if not version:
        return False
    parts = version.split(".")
    try:
        current = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return False
    return current >= (major, minor)


def resolve_inference_device(gpu: int = 0) -> torch.device:
    """Resolve the supported accelerator by platform.

    Linux uses CUDA. macOS uses MPS when available. Other platforms and CPU-only
    environments are not supported for inference.
    """
    if sys.platform.startswith("linux"):
        if gpu < 0:
            raise ValueError(
                "CPU inference is not supported. Please provide --gpu with a CUDA "
                "device ID (0..N-1)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this Linux environment. Emap2lig "
                "requires an NVIDIA GPU (CUDA 12/13) on Linux."
            )
        device_count = torch.cuda.device_count()
        if gpu >= device_count:
            raise ValueError(
                f"Invalid GPU device ID {gpu}. Available CUDA devices: 0..{device_count - 1}."
            )
        return torch.device(f"cuda:{gpu}")

    if sys.platform == "darwin":
        if gpu not in (0, -1):
            raise ValueError("MPS exposes a single device. Use --gpu 0 for MPS.")
        if not _mps_available():
            raise RuntimeError(
                "Apple MPS is not available on this macOS environment. Emap2lig "
                "requires MPS on macOS."
            )
        if not _macos_version_at_least(13, 2):
            raise RuntimeError(
                "MPS Conv3d inference requires macOS 13.2 or newer. Please "
                "upgrade macOS to run Emap2lig on MPS."
            )
        return torch.device("mps")

    raise RuntimeError(
        f"Unsupported platform {sys.platform!r}. Emap2lig inference supports "
        "Linux/CUDA and macOS/MPS only."
    )


def _require_accelerator(gpu: int) -> None:
    """Validate that a usable platform accelerator is available."""
    resolve_inference_device(gpu)


def _trainer_device_kwargs(device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        return {"accelerator": "gpu", "devices": [device.index or 0]}
    if device.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    raise RuntimeError(f"Unsupported inference device: {device}")


# ---------------------------------------------------------------------------
# Model weight helpers — download to ~/.emap2lig/models/
# ---------------------------------------------------------------------------


def get_model_path(filename: str) -> Path:
    """Return the expected local path for a model weight file.

    Parameters
    ----------
    filename : str
        Relative filename inside the HuggingFace repo
        (e.g. ``"emap2lig-find-v0.0.1.safetensors"``).

    Returns
    -------
    Path
        ``~/.emap2lig/models/<filename>``.
    """
    return MODEL_DIR / filename


def check_weights() -> dict[str, bool]:
    """Check which model weights are already downloaded.

    Returns
    -------
    dict[str, bool]
        Mapping of model key → whether the file exists locally.
    """
    return {key: get_model_path(fn).exists() for key, fn in MODEL_FILES.items()}


def download_weights(
    repo_id: str = REPO_ID,
    filename: str | None = None,
) -> Path:
    """Download a single weight file to ``~/.emap2lig/models/``.

    If *filename* is ``None``, all known model files are downloaded.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID.
    filename : str | None
        Relative path inside the repo. When ``None`` every file in
        :data:`MODEL_FILES` is downloaded.

    Returns
    -------
    Path
        Path to the downloaded file (or :data:`MODEL_DIR` when downloading all).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        for fn in MODEL_FILES.values():
            download_weights(repo_id=repo_id, filename=fn)
        return MODEL_DIR

    local_path = get_model_path(filename)
    if local_path.exists():
        logger.info(f"Weights already present at {local_path}")
        return local_path

    logger.info(f"Downloading {repo_id}/{filename} → {local_path}")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODEL_DIR))
    return local_path


def ensure_weights() -> None:
    """Download all model weights if not already present."""
    status = check_weights()
    if all(status.values()):
        logger.info("All model weights present.")
        return
    for key, present in status.items():
        if not present:
            fn = MODEL_FILES[key]
            logger.info(f"Missing {key} ({fn}), downloading...")
            download_weights(filename=fn)


def detect_ligand_objects(
    input_map: Path | str,
    output_dir: Path | str,
    cfg: object,
    emdb_id: str | None = None,
) -> tuple[int, Path | None]:
    """Detect ligand objects in cryo-EM density maps using Emap2lig.

    Args:
        input_map: Path to input cryo-EM map file.
        output_dir: Directory to save output files.
        cfg: Hydra configuration object.
        emdb_id: EMDB ID for the map (optional).

    Returns:
        Tuple of (status_code, blobs_dir).  *status_code* is 0 for
        success, 1 for failure.  *blobs_dir* is ``None`` on failure.
    """
    try:
        # Create paths
        input_path = Path(input_map)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        map_stem = str(input_path.stem).split(".")[0]
        logger.info(f"Map stem: {map_stem}")

        # Enforce platform accelerator inference for detection.
        device = resolve_inference_device(int(cfg.gpu))

        detection_spatial_size = int(cfg.detection_model.spatial_size)
        structure_spatial_size = int(cfg.spatial_size)

        # Output file paths/directories
        preprocess_dir = output_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        find_maps_dir = output_dir / "find_maps"
        find_maps_dir.mkdir(parents=True, exist_ok=True)
        blobs_dir = output_dir / "find_blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)

        unified_map_path = preprocess_dir / "unified.mrc"
        detection_labels = list(cfg.detection_labels)
        if not detection_labels:
            raise ValueError("cfg.detection_labels must define at least one label")
        if "ligand" not in detection_labels:
            raise ValueError("cfg.detection_labels must include 'ligand'")
        label_map_paths = {
            label: find_maps_dir / f"{label}.mrc" for label in detection_labels
        }
        label_mask_paths = {
            label: find_maps_dir / f"{label}_mask.mrc" for label in detection_labels
        }
        ligand_map_path = label_map_paths["ligand"]
        ligand_mask_path = label_mask_paths["ligand"]

        # Parse input map
        logger.info(f"Loading cryo-EM map from {input_path}")
        raw_obj = parse_mrc(input_path, emdb_id=emdb_id)
        unified_obj = get_unified_mrc(
            raw_obj,
            min_spatial_size=detection_spatial_size,
            extended_val=cfg.extended_val,
            contour_level=cfg.contour_level,
        )
        to_mrc(unified_obj, unified_map_path, verbose=False)

        missing_label_map = [
            label for label in detection_labels if not label_map_paths[label].exists()
        ]
        missing_label_mask = [
            label for label in detection_labels if not label_mask_paths[label].exists()
        ]

        if missing_label_map or missing_label_mask:
            # Convert map to tensor
            input_map_tensor = torch.from_numpy(unified_obj.grid_data).float()
            # Add batch and channel dimensions
            input_map_tensor = input_map_tensor.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, D, H, W]

            # Initialize model from config
            logger.info("Initializing Emap2lig model from configuration")
            detection_model = instantiate(cfg.detection_model)

            output_device = torch.device("cpu")

            # Run ligand detection
            logger.info(f"Running ligand detection on device: {device}")
            predicted_map = detection_model.sliding_window_inference(
                input_map=input_map_tensor,
                roi_size=detection_spatial_size,
                batch_size=cfg.detection_batch_size,
                device=device,
                output_device=output_device,
            )

            if predicted_map.shape[1] < len(detection_labels):
                raise ValueError(
                    "Detection output has fewer channels than configured labels: "
                    f"{predicted_map.shape[1]} < {len(detection_labels)}"
                )

            binary_map, channel_thresholds = detection_model._binarize_output(
                predicted_map
            )
            if "ligand" in channel_thresholds:
                logger.info(
                    f"Ligand detection threshold: {channel_thresholds['ligand']:.6f}"
                )

            ligand_mask_obj = None

            for channel_idx, label in enumerate(detection_labels):
                label_map = predicted_map[:, channel_idx : channel_idx + 1, ...]
                label_mask = binary_map[:, channel_idx : channel_idx + 1, ...]
                label_mask_np = label_mask.squeeze(0).squeeze(0).numpy()

                if np.count_nonzero(label_mask_np) == 0:
                    logger.info(
                        f"Skipping {label} map/mask save: binarized map is all zero"
                    )
                    if label == "ligand":
                        ligand_mask_obj = MapObject(
                            grid_data=label_mask_np,
                            voxel_size=unified_obj.voxel_size,
                            global_origin=unified_obj.global_origin,
                            emdb_id=unified_obj.emdb_id,
                        )
                    continue

                label_map_np = label_map.squeeze(0).squeeze(0).numpy()
                label_map_obj = MapObject(
                    grid_data=label_map_np,
                    voxel_size=unified_obj.voxel_size,
                    global_origin=unified_obj.global_origin,
                    emdb_id=unified_obj.emdb_id,
                )
                label_map_path = label_map_paths[label]
                logger.info(f"Saving {label} map to {label_map_path}")
                to_mrc(label_map_obj, label_map_path, verbose=False)

                label_mask_obj = MapObject(
                    grid_data=label_mask_np,
                    voxel_size=unified_obj.voxel_size,
                    global_origin=unified_obj.global_origin,
                    emdb_id=unified_obj.emdb_id,
                )
                label_mask_path = label_mask_paths[label]
                logger.info(f"Saving {label} mask to {label_mask_path}")
                to_mrc(label_mask_obj, label_mask_path, verbose=False)

                if label == "ligand":
                    ligand_mask_obj = label_mask_obj

            if ligand_mask_obj is None:
                raise RuntimeError("Failed to generate ligand mask object")

            logger.info("Ligand detection completed successfully")
        else:
            logger.info(
                f"All detection label maps/masks already exist under {find_maps_dir}"
            )
            # load the mask
            ligand_mask_obj = parse_mrc(ligand_mask_path)

            if not ligand_map_path.exists():
                logger.warning(f"Ligand map is missing at {ligand_map_path}")

        # Split the ligand mask into separate objects and save them
        logger.info("Splitting ligand mask into separate objects")

        # Label connected components in the ligand mask
        labeled_mask = measure.label(ligand_mask_obj.grid_data, connectivity=3)
        logger.info(f"Found {np.max(labeled_mask)} objects in the ligand mask")

        # Extract object information
        props = measure.regionprops(labeled_mask)

        i = 0
        # Process each object
        for prop in props:
            # Skip very small objects (likely noise)
            if prop.area < 32:
                continue

            i += 1

            # Create a binary mask for this object
            object_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
            object_mask[labeled_mask == prop.label] = 1

            # Create MapObject for this object
            mask_map = MapObject(
                grid_data=object_mask,
                voxel_size=ligand_mask_obj.voxel_size,
                global_origin=ligand_mask_obj.global_origin,
                emdb_id=ligand_mask_obj.emdb_id,
            )

            # Create a dictionary with the mask and density maps, plus the unified map
            mrc_objects = {"mask": mask_map, "unified": unified_obj}

            # Crop all maps together, excluding the unified map from cropping calculations
            logger.info(f"Cropping object {i} with unified map")
            cropped_mrcs, _ = crop_mrcs(
                mrc_objects=mrc_objects,
                excluded_keys=["unified"],
                min_spatial_size=structure_spatial_size,
            )

            # Save the cropped mask
            mask_path = blobs_dir / f"mask_{i}.mrc"

            # Create LigandDensityObject and save it
            ligand_density_object = DensityObject(
                object_id=i,
                density_grid=cropped_mrcs["unified"].grid_data,
                instance_grid=cropped_mrcs["mask"].grid_data,
                global_origin=cropped_mrcs["unified"].global_origin,
                voxel_size=cropped_mrcs["unified"].voxel_size,
            )

            # Save as NPZ file
            blob_path = blobs_dir / f"blob_{i}.npz"
            ligand_density_object.dump(blob_path)
            logger.info(f"Saved blob {i} to {blob_path}")

            logger.info(f"Saving mask {i} to {mask_path}")
            to_mrc(cropped_mrcs["mask"], mask_path, verbose=False)

        logger.info(f"Saved {i} blobs to {blobs_dir}")

        if i > 100:
            # exit the program
            logger.warning(
                f"The detection contains {i} (more than 100) objects. To avoid wasting time, exiting..."
            )
            sys.exit(1)

        # Check if any objects were detected
        if len(props) == 0:
            logger.warning("No ligand blobs detected")
            return 1, blobs_dir

        return 0, blobs_dir

    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Error during ligand object detection: {e}")
        return 1, None


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a 4-element integer code (PDB convention).

    Each character is mapped to ``ord(c) - 32``; shorter names are
    zero-padded to length 4.

    Args:
        name: Atom name string (max 4 characters after stripping).

    Returns:
        4-tuple of integer character codes.

    Raises:
        ValueError: If the name exceeds 4 characters after stripping.
    """
    name = name.strip()
    name_code = [ord(c) - 32 for c in name]
    if len(name_code) > 4:
        raise ValueError(
            f"Atom name {name!r} exceeds 4 characters (got {len(name_code)})"
        )
    name_code = name_code + [0] * (4 - len(name_code))
    return tuple(name_code)  # type: ignore[return-value]


def _extract_atom_features(
    mol: Chem.Mol,
    residue_id: int = 1,
    auto_name: bool = False,
) -> tuple[list[tuple], list[str]]:
    """Extract atom features and names from an RDKit molecule.

    Args:
        mol: RDKit molecule (expected to have a conformer and atom
            ``name`` properties when *auto_name* is False).
        residue_id: Residue identifier stored in each atom record.
        auto_name: When True, atoms without a ``name`` property are
            assigned ``"<SYMBOL><INDEX>"`` (e.g. ``"C1"``).  When
            False, the name defaults to an empty string.

    Returns:
        Tuple of (atom_tuples, atom_names) where each atom tuple
        conforms to the :data:`Atom` structured-array dtype.
    """
    atom_tuples: list[tuple] = []
    atom_names: list[str] = []

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        if atom.HasProp("name"):
            atom_name = atom.GetProp("name")
        elif auto_name:
            atom_name = f"{atom.GetSymbol()}{atom_idx + 1}"
        else:
            atom_name = ""

        atom_names.append(atom_name)

        name_code = convert_atom_name(atom_name)
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()

        chirality_type = atom.GetChiralTag().name
        chirality_type_id = chirality_type_ids.get(chirality_type, 0)
        chirality_one_hot = [False] * 7
        chirality_one_hot[chirality_type_id] = True

        in_ring = [
            atom.IsInRingSize(3),
            atom.IsInRingSize(4),
            atom.IsInRingSize(5),
            atom.IsInRingSize(6),
        ]

        try:
            conformer = mol.GetConformer()
            pos = conformer.GetAtomPosition(atom.GetIdx())
            ref_pos = (pos.x, pos.y, pos.z)
        except ValueError:
            ref_pos = (0.0, 0.0, 0.0)

        atom_tuples.append(
            (
                name_code,
                atomic_num,
                formal_charge,
                (0.0, 0.0, 0.0),
                ref_pos,
                False,
                tuple(chirality_one_hot),
                tuple(in_ring),
                residue_id,
            )
        )

    return atom_tuples, atom_names


def _extract_bond_features(
    mol: Chem.Mol,
    atom_offset: int = 0,
) -> list[tuple]:
    """Extract bond features from an RDKit molecule.

    Args:
        mol: RDKit molecule.
        atom_offset: Offset added to atom indices (used when combining
            residues from a branched ligand).

    Returns:
        List of bond tuples conforming to the :data:`Bond` dtype.
    """
    bond_tuples: list[tuple] = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx() + atom_offset
        end_idx = bond.GetEndAtomIdx() + atom_offset

        bond_type = bond.GetBondType().name
        bond_type_id = bond_type_ids.get(bond_type, 0)
        bond_type_one_hot = [False] * 5
        if bond_type_id < 5:
            bond_type_one_hot[bond_type_id] = True

        in_ring = [
            bond.IsInRingSize(3),
            bond.IsInRingSize(4),
            bond.IsInRingSize(5),
            bond.IsInRingSize(6),
        ]

        bond_tuples.append(
            (
                begin_idx,
                end_idx,
                tuple(bond_type_one_hot),
                tuple(in_ring),
            )
        )

    return bond_tuples


def process_branched_ligand(
    branched_config: dict,
    ligand_name: str,
    ligands_dir: Path,
    blobs: list[int] | None = None,
) -> None:
    """Process a branched ligand configuration to create a LigandObject.

    Args:
        branched_config: Dictionary with ``residues`` and ``bonds`` keys.
        ligand_name: Name for the ligand (e.g. ``"LIG1"``).
        ligands_dir: Directory to save the output NPZ.
        blobs: List of blob IDs associated with this ligand.
    """
    # Extract residue information
    residue_list = branched_config.get("residues", [])
    inter_bonds = branched_config.get("bonds", [])

    # Parse residue names from the list (format: "1. NAG")
    residue_names = []
    for res_entry in residue_list:
        parts = res_entry.strip().split(".")
        if len(parts) >= 2:
            res_name = parts[1].strip()
            residue_names.append(res_name)

    logger.info(
        f"Processing branched ligand {ligand_name} with residues: {residue_names}"
    )

    all_atoms: list[tuple] = []
    all_bonds: list[tuple] = []
    atom_names: list[str] = []
    atom_offset = 0
    residue_atom_counts: list[int] = []

    for residue_idx, res_name in enumerate(residue_names, start=1):
        try:
            ref_mol = get_ccd_mol(res_name)
        except CCDFetchError as exc:
            logger.warning(f"Residue {res_name}: {exc}")
            continue
        ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)

        residue_atoms, residue_atom_names = _extract_atom_features(
            ref_mol, residue_id=residue_idx, auto_name=True
        )
        all_atoms.extend(residue_atoms)
        atom_names.extend(residue_atom_names)
        residue_atom_counts.append(len(residue_atoms))

        residue_bonds = _extract_bond_features(ref_mol, atom_offset=atom_offset)
        all_bonds.extend(residue_bonds)

        atom_offset += len(residue_atoms)

    # Process inter-residue bonds
    for bond_info in inter_bonds:
        if len(bond_info) != 4:
            logger.warning(f"Invalid bond format: {bond_info}")
            continue

        res1_idx, atom1_name, res2_idx, atom2_name = bond_info

        # Convert to 0-based indices
        res1_idx -= 1
        res2_idx -= 1

        if res1_idx >= len(residue_atom_counts) or res2_idx >= len(residue_atom_counts):
            logger.warning(f"Invalid residue indices in bond: {bond_info}")
            continue

        # Find global atom indices
        # Calculate atom offset for each residue
        res1_offset = sum(residue_atom_counts[:res1_idx])
        res2_offset = sum(residue_atom_counts[:res2_idx])

        # Find atom indices within residues (simplified - assumes atom names are unique)
        atom1_idx = None
        atom2_idx = None

        # Search for atoms in the respective residues
        for i in range(residue_atom_counts[res1_idx]):
            if atom_names[res1_offset + i] == atom1_name:
                atom1_idx = res1_offset + i
                break

        for i in range(residue_atom_counts[res2_idx]):
            if atom_names[res2_offset + i] == atom2_name:
                atom2_idx = res2_offset + i
                break

        if atom1_idx is not None and atom2_idx is not None:
            # Inter-residue bonds are typically single bonds
            bond_type_one_hot = [True, False, False, False, False]  # SINGLE
            in_ring = [False, False, False, False]

            all_bonds.append(
                (
                    atom1_idx,
                    atom2_idx,
                    tuple(bond_type_one_hot),
                    tuple(in_ring),
                )
            )
        else:
            logger.warning(f"Could not find atoms for bond: {bond_info}")

    # Create structured arrays
    atoms = np.array(all_atoms, dtype=Atom)
    bonds = np.array(all_bonds, dtype=Bond)

    # Create LigandObject
    ref_mol_object = LigandObject(
        smiles="",
        atom_names=atom_names,
        atoms=atoms,
        bonds=bonds,
        name=ligand_name,  # Use ligand name (LIG1, LIG2, etc.)
        residue_names=residue_names,  # Actual residue names (NAG, NAG, etc.)
        symmetries=[],  # No symmetries for branched ligands
        blobs=blobs,
    )

    # Save to NPZ file
    output_path = ligands_dir / f"{ligand_name}.npz"
    ref_mol_object.dump(output_path)
    logger.info(f"Saved branched molecular object for {ligand_name} to {output_path}")


def parse_ligand_list(ligand_list_path: Path | str) -> list[LigandRecord]:
    """Parse a ligand list file into a list of LigandRecord objects.

    Supports YAML (list or single-item), and plain text with
    ``CCD:`` or ``SMILES:`` prefixes.

    Args:
        ligand_list_path: Path to the ligand list file.

    Returns:
        Parsed ligand records.  Empty list on unsupported format.
    """
    ligand_records: list[LigandRecord] = []

    with open(ligand_list_path) as f:
        content = f.read().strip()

    # Try to parse as YAML first
    try:
        yaml_data = yaml.safe_load(content)

        # Handle both single item and list cases
        if isinstance(yaml_data, list):
            items_to_process = yaml_data
        elif yaml_data is not None:
            # Single item case - wrap in a list
            items_to_process = [yaml_data]
        else:
            # Empty YAML or null
            items_to_process = []

        lig_counter = 0  # Unified counter for naming non-CCD ligands

        for item in items_to_process:
            if isinstance(item, dict):
                if "BRANCHED" in item:
                    # Handle BRANCHED dictionary
                    branched_config = item["BRANCHED"]
                    blob_ids = item.get("blob_id", item.get("blobs", None))
                    if blob_ids is not None and not isinstance(blob_ids, list):
                        blob_ids = [blob_ids]

                    # Extract residues and bonds from branched config
                    residue_list = branched_config.get("residues", [])
                    inter_bonds = branched_config.get("bonds", [])

                    # Parse residue names from the list (format: "1. NAG")
                    residues = {}
                    for i, res_entry in enumerate(residue_list, start=1):
                        parts = res_entry.strip().split(".")
                        if len(parts) >= 2:
                            res_name = parts[1].strip()
                            residues[i] = res_name

                    lig_counter += 1
                    ligand_records.append(
                        LigandRecord(
                            type="BRANCHED",
                            name=f"LIG{lig_counter}",
                            blobs=blob_ids,
                            residues=residues,
                            bonds=inter_bonds,
                        )
                    )
                elif "CCD" in item:
                    # Handle CCD dictionary format
                    ccd_code = item["CCD"]
                    blob_ids = item.get("blob_id", item.get("blobs", None))
                    if blob_ids is not None and not isinstance(blob_ids, list):
                        blob_ids = [blob_ids]

                    ligand_records.append(
                        LigandRecord(
                            type="CCD",
                            name=ccd_code,
                            blobs=blob_ids,
                        )
                    )
                elif "SMILES" in item:
                    # Handle SMILES dictionary format
                    smiles_string = item["SMILES"]
                    blob_ids = item.get("blob_id", item.get("blobs", None))
                    if blob_ids is not None and not isinstance(blob_ids, list):
                        blob_ids = [blob_ids]

                    lig_counter += 1
                    ligand_records.append(
                        LigandRecord(
                            type="SMILES",
                            name=f"LIG{lig_counter}",
                            blobs=blob_ids,
                            smiles=smiles_string,
                        )
                    )
                else:
                    logger.warning(f"Unrecognized dictionary format in list: {item}")
            elif isinstance(item, str):
                # Handle CCD: or SMILES: strings
                if item.startswith("CCD:"):
                    ccd_code = item.split(":", 1)[1].strip()
                    ligand_records.append(
                        LigandRecord(
                            type="CCD",
                            name=ccd_code,
                        )
                    )
                elif item.startswith("SMILES:"):
                    smiles_string = item.split(":", 1)[1].strip()
                    lig_counter += 1
                    ligand_records.append(
                        LigandRecord(
                            type="SMILES",
                            name=f"LIG{lig_counter}",
                            smiles=smiles_string,
                        )
                    )
                else:
                    logger.warning(f"Unrecognized string format in list: {item}")
            else:
                logger.warning(f"Unrecognized item format in list: {item}")

        return ligand_records

    except yaml.YAMLError:
        # If YAML parsing fails, try to parse as plain text
        logger.info(
            f"YAML parsing failed, trying plain text format for {ligand_list_path}"
        )

    # Handle plain text files with single ligand specification
    content_lines = [line.strip() for line in content.split("\n") if line.strip()]

    if len(content_lines) == 1:
        line = content_lines[0]
        if line.startswith("CCD:"):
            ccd_code = line.split(":", 1)[1].strip()
            ligand_records.append(
                LigandRecord(
                    type="CCD",
                    name=ccd_code,
                )
            )
            return ligand_records
        elif line.startswith("SMILES:"):
            smiles_string = line.split(":", 1)[1].strip()
            ligand_records.append(
                LigandRecord(
                    type="SMILES",
                    name="LIG1",
                    smiles=smiles_string,
                )
            )
            return ligand_records

    # If we get here, the format is not supported
    logger.error(
        f"Unsupported file format for {ligand_list_path}. "
        f"Supported formats: YAML files with lists/single items, or plain text with single CCD:/SMILES: entry."
    )
    return []


def prepare_CCD_data(ligand_record: LigandRecord, ligands_dir: Path) -> str | None:
    """
    Process a CCD ligand record to create a LigandObject.

    Args:
        ligand_record: CCD ligand record
        ligands_dir: Directory to save the output

    Returns:
        Ligand name if successful, None if failed
    """
    ligand_name = ligand_record.name
    logger.info(f"Processing CCD ligand {ligand_name}")

    try:
        ref_mol = get_ccd_mol(ligand_name)
    except CCDFetchError as exc:
        logger.warning(f"Ligand {ligand_name}: {exc}")
        return None

    try:
        # Generate SMILES
        smiles = Chem.MolToSmiles(ref_mol)

        # Remove hydrogens for processing
        ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)

        # Process the molecule
        process_molecule(ref_mol, ligand_name, smiles, ligands_dir, ligand_record.blobs)
        return ligand_name

    except Exception as e:
        logger.error(f"Error processing CCD ligand {ligand_name}: {e}")
        return None


def prepare_SMILES_data(ligand_record: LigandRecord, ligands_dir: Path) -> str | None:
    """
    Process a SMILES ligand record to create a LigandObject.

    Args:
        ligand_record: SMILES ligand record
        ligands_dir: Directory to save the output

    Returns:
        Ligand name if successful, None if failed
    """
    ligand_name = ligand_record.name
    smiles = ligand_record.smiles

    if not smiles:
        logger.error(f"No SMILES string provided for ligand {ligand_name}")
        return None

    logger.info(f"Processing SMILES ligand as {ligand_name}: {smiles}")

    try:
        # Get conformer from SMILES
        result, ref_mol = get_conformer_from_smiles(smiles)

        if result == "failed":
            logger.warning(f"Failed to generate conformer for SMILES: {smiles}")
            return None

        # Process the molecule
        process_molecule(ref_mol, ligand_name, smiles, ligands_dir, ligand_record.blobs)
        return ligand_name

    except Exception as e:
        logger.error(f"Error processing SMILES ligand {ligand_name}: {e}")
        return None


def prepare_BRANCHED_data(ligand_record: LigandRecord, ligands_dir: Path) -> str | None:
    """
    Process a branched ligand record to create a LigandObject.

    Args:
        ligand_record: BRANCHED ligand record
        ligands_dir: Directory to save the output

    Returns:
        Ligand name if successful, None if failed
    """
    ligand_name = ligand_record.name
    residues = ligand_record.residues or {}
    bonds = ligand_record.bonds or []

    logger.info(
        f"Processing branched ligand {ligand_name} with residues: {list(residues.values())}"
    )

    # Convert to the format expected by process_branched_ligand
    branched_config = {
        "residues": [f"{idx}. {res_name}" for idx, res_name in residues.items()],
        "bonds": bonds,
    }

    try:
        process_branched_ligand(
            branched_config, ligand_name, ligands_dir, ligand_record.blobs
        )
        return ligand_name
    except Exception as e:
        logger.error(f"Error processing branched ligand {ligand_name}: {e}")
        return None


def prepare_ligand_dataset(
    ligand_records: list[LigandRecord],
    output_dir: Path,
    ligands_dir: Path | None = None,
) -> tuple[int, Path]:
    """
    Prepare a dataset of ligands for structure modeling.

    Args:
        ligand_records: List of ligand records to process
        output_dir: Root output directory
        ligands_dir: If provided, write ligand NPZ files here instead of the
            default ``output_dir/preprocess/ligands``.  Useful for incremental
            runs that must not pick up ligands from earlier runs.

    Returns:
        Tuple of (status_code, ligands_dir)
    """
    if ligands_dir is None:
        ligands_dir = output_dir / "preprocess" / "ligands"
    ligands_dir.mkdir(parents=True, exist_ok=True)

    # Process each ligand
    processed_ligands = []

    for record in ligand_records:
        if record.type == "BRANCHED":
            result = prepare_BRANCHED_data(record, ligands_dir)
        elif record.type == "CCD":
            result = prepare_CCD_data(record, ligands_dir)
        elif record.type == "SMILES":
            result = prepare_SMILES_data(record, ligands_dir)
        else:
            logger.warning(f"Unrecognized ligand record type: {record.type}")
            continue

        if result is not None:
            processed_ligands.append(result)

    logger.info(f"Successfully processed {len(processed_ligands)} ligands")
    return 0, ligands_dir


def process_molecule(
    ref_mol: Chem.Mol,
    ligand_name: str,
    smiles: str,
    ligands_dir: Path,
    blobs: list[int] | None = None,
) -> None:
    """Process a molecule to create a LigandObject and save it.

    Args:
        ref_mol: RDKit molecule object.
        ligand_name: Name for the ligand.
        smiles: SMILES string for the ligand.
        ligands_dir: Directory to save the output NPZ.
        blobs: List of blob IDs associated with this ligand.
    """
    atoms_list, atom_names = _extract_atom_features(ref_mol, residue_id=1)
    bonds_list = _extract_bond_features(ref_mol)

    atoms = np.array(atoms_list, dtype=Atom)
    bonds = np.array(bonds_list, dtype=Bond)

    ref_mol_object = LigandObject(
        smiles=smiles,
        atom_names=atom_names,
        atoms=atoms,
        bonds=bonds,
        name=ligand_name,  # Use ligand name (LIG1, LIG2, etc. for SMILES/BRANCHED, CCD code for CCD)
        residue_names=[ligand_name],
        symmetries=[],
        blobs=blobs,
    )

    # Save to NPZ file
    output_path = ligands_dir / f"{ligand_name}.npz"
    ref_mol_object.dump(output_path)
    logger.info(f"Saved reference molecular object for {ligand_name} to {output_path}")


def create_blob_csv_tables(outputs_dir: Path) -> None:
    """Sort per-blob CSV files by score and update the ``best/`` directory.

    The writer already appends rows to each blob's CSV during inference.
    This function re-reads those CSVs, sorts them by ``consistency_iou``
    descending, writes them back, and copies the highest-scoring CIF into
    ``best/``.

    Args:
        outputs_dir: Directory containing blob subdirectories with CSV/CIF files.
    """
    logger.info("Creating CSV tables for blob results")

    best_dir = outputs_dir / "best"
    best_dir.mkdir(exist_ok=True)
    logger.info(f"Created best results directory: {best_dir}")

    blob_dirs = [
        d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("blob_")
    ]

    for blob_dir in blob_dirs:
        logger.info(f"Processing {blob_dir.name}")

        csv_file = blob_dir / f"{blob_dir.name}_results.csv"

        # Read existing CSV rows
        scores: dict[str, float] = {}
        if csv_file.exists():
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row["conformer_name"]
                        try:
                            scores[name] = float(row["consistency_iou"])
                        except (KeyError, ValueError):
                            pass
            except OSError as e:
                logger.warning(f"Failed to read CSV {csv_file}: {e}")

        # Safety net: include any CIF files that somehow lack a CSV entry
        for cif_file in blob_dir.glob("*.cif"):
            name = cif_file.stem
            if name not in scores:
                scores[name] = 0.0

        if not scores:
            logger.warning(f"No conformers found in {blob_dir}")
            continue

        # Re-write CSV sorted by score descending
        csv_data = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conformer_name", "consistency_iou"])
            writer.writerows(csv_data)
        logger.info(f"Created CSV table: {csv_file} with {len(csv_data)} entries")

        # Copy best CIF to the 'best' directory
        best_name, best_score = csv_data[0]
        best_cif = blob_dir / f"{best_name}.cif"
        if best_cif.exists():
            dest = best_dir / f"{blob_dir.name}_{best_name}.cif"
            try:
                shutil.copy2(best_cif, dest)
                logger.info(f"Copied best result: {dest} (score: {best_score:.4f})")
            except OSError as e:
                logger.warning(f"Failed to copy {best_cif} to best directory: {e}")
        else:
            logger.warning(f"Best .cif file not found: {best_cif}")


def load_config(
    gpu: int = 0,
    detection_batch_size: int | None = None,
    contour_level: float | None = None,
) -> object:
    """Load Emap2lig configuration using Hydra.

    Args:
        gpu: CUDA GPU device ID on Linux; ignored except 0/-1 on macOS MPS.
        detection_batch_size: Override detection batch size from config.
        contour_level: Override contour level from config.

    Returns:
        Hydra configuration object.
    """
    overrides = []
    if gpu is not None:
        overrides.append(f"gpu={gpu}")
    if detection_batch_size is not None:
        overrides.append(f"detection_batch_size={detection_batch_size}")
    if contour_level is not None:
        overrides.append(f"contour_level={contour_level}")
    config_dir = Path(__file__).resolve().parent
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="emap2lig", overrides=overrides)
    return cfg


def _multiplicity_chunks(
    multiplicity: int, max_parallel_multiplicity: int
) -> list[int]:
    """Split total multiplicity into bounded in-model prediction chunks."""
    if multiplicity < 1:
        raise ValueError("multiplicity must be at least 1")
    if max_parallel_multiplicity < 1:
        raise ValueError("max_parallel_multiplicity must be at least 1")

    chunks: list[int] = []
    remaining = multiplicity
    while remaining > 0:
        chunk = min(remaining, max_parallel_multiplicity)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def run_structure_modeling(
    blobs_dir: Path | str,
    output_dir: Path | str,
    ligand_records: list[LigandRecord],
    cfg,
    gpu: int = 0,
    multiplicity: int = 1,
    max_parallel_multiplicity: int = 8,
    blob_ids: list[int] | None = None,
) -> int:
    """Run structure modeling on detected blobs (Stage 2).

    Takes pre-detected blobs from Stage 1 and a list of ligand records,
    then runs the diffusion-based structure modeling pipeline.

    Args:
        blobs_dir: Directory containing detected blob NPZ files from Stage 1.
        output_dir: Root output directory (will contain ``build_struct/`` subdirectory).
        ligand_records: List of :class:`LigandRecord` objects specifying ligands.
        cfg: Hydra configuration object (from :func:`load_config`).
        gpu: CUDA GPU device ID on Linux; ignored except 0/-1 on macOS MPS.
        multiplicity: Number of conformers per blob per ligand.
        max_parallel_multiplicity: Maximum number of conformers generated in one
            model sub-batch.  Larger ``multiplicity`` values are split inside
            each prediction step to reduce peak accelerator memory use while
            keeping a single outer Lightning progress bar.
        blob_ids: If provided, only process blobs with these IDs.
            When ``None`` (default), all blobs in *blobs_dir* are processed.

    Returns:
        Status code: 0 for success, 1 for failure.
    """
    blobs_dir = Path(blobs_dir)
    output_dir = Path(output_dir)
    build_struct_dir = output_dir / "build_struct"
    build_struct_dir.mkdir(parents=True, exist_ok=True)

    try:
        _multiplicity_chunks(multiplicity, max_parallel_multiplicity)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    # Enforce platform accelerator inference.
    device = resolve_inference_device(gpu)

    logger.info("Starting ligand dataset preparation.")

    # Load blob paths, optionally filtered to specific blob IDs
    blob_paths = sorted(list(blobs_dir.glob("blob_*.npz")))
    if blob_ids is not None:
        allowed = set(blob_ids)
        blob_paths = [p for p in blob_paths if int(p.stem.split("_")[1]) in allowed]
        logger.info(f"Filtered to {len(blob_paths)} blobs (ids={blob_ids})")
    if not blob_paths:
        logger.error("No blob files found. Exiting.")
        return 1

    logger.info(f"Processing {len(ligand_records)} ligand records")

    # For incremental runs (specific blob_ids), use an isolated ligands
    # directory so the dataset only sees the newly requested ligands.
    incremental_ligands_dir: Path | None = None
    if blob_ids is not None:
        incremental_ligands_dir = (
            output_dir / "preprocess" / f"_ligands_{uuid.uuid4().hex[:8]}"
        )

    _, ligands_object_dir = prepare_ligand_dataset(
        ligand_records=ligand_records,
        output_dir=output_dir,
        ligands_dir=incremental_ligands_dir,
    )

    ligand_paths = sorted(list(Path(ligands_object_dir).glob("*.npz")))
    if not ligand_paths:
        logger.error("No ligand files found. Exiting.")
        return 1

    logger.info("Ligand dataset preparation completed.")

    logger.info("Starting inference with PyTorch Lightning.")

    # Use the already-filtered blob_paths as density inputs
    density_blob_paths = blob_paths
    if not density_blob_paths:
        logger.error("No density blob files found. Exiting.")
        return 1

    # Create dataset with total multiplicity so Lightning sees one batch per
    # blob-ligand pair.  The model processes those prompt points in smaller
    # internal chunks capped by max_parallel_multiplicity.
    dataset = LigandModelingDataset(
        density_object_list=density_blob_paths,
        ref_mol_dir=ligands_object_dir,
        multiplicity=multiplicity,
    )

    # Create dataloader — structure modeling always uses batch_size=1
    # (one blob-ligand pair at a time; multiplicity controls conformer count)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # Initialize model
    logger.info(f"Instantiating model {cfg.model._target_}")
    model = instantiate(cfg.model)

    # Set multiplicity from CLI/API arguments (no need for model to infer it).
    model.predict_args.multiplicity = multiplicity
    model.predict_args.max_parallel_multiplicity = max_parallel_multiplicity

    # Create writer callback
    writer = LigandWriter(
        output_dir=build_struct_dir,
        output_format="mmcif",
    )

    # Configure trainer
    trainer_kwargs = {
        **_trainer_device_kwargs(device),
        "callbacks": [writer],
        "logger": False,  # Disable default logger
    }

    # Create trainer
    logger.info("Initializing PyTorch Lightning Trainer")
    trainer = Trainer(**trainer_kwargs)

    # Run prediction using dataloader directly.  Large multiplicity values are
    # chunked inside model.forward(), not by repeating trainer.predict(), so the
    # user sees one Build progress pass over blob-ligand combinations.
    logger.info(
        f"Running inference on {len(dataset)} combinations with multiplicity "
        f"{multiplicity} (max parallel multiplicity {max_parallel_multiplicity})"
    )
    trainer.predict(model, dataloaders=dataloader)

    logger.info("Inference completed.")

    # Clean up temporary ligands directory used for incremental runs
    if incremental_ligands_dir is not None and incremental_ligands_dir.exists():
        shutil.rmtree(incremental_ligands_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary ligands dir: {incremental_ligands_dir}")

    # Create CSV tables for each blob
    create_blob_csv_tables(build_struct_dir)

    return 0


app = typer.Typer()


@app.command()
def main(
    input_map: str = typer.Option(
        ..., "--input-map", help="Path to input cryo-EM map file"
    ),
    output_dir: str = typer.Option(
        "./output", "--output-dir", help="Directory to save output files"
    ),
    gpu: int = typer.Option(0, "--gpu", help="Accelerator device ID"),
    detection_batch_size: int | None = typer.Option(
        None,
        "--detection-batch-size",
        help="Batch size for detection model sliding window inference (overrides config default of 16)",
    ),
    emdb_id: str | None = typer.Option(None, "--emdb-id", help="EMDB ID for the map"),
    contour_level: float | None = typer.Option(
        None, "--contour-level", help="Contour level for the map"
    ),
    ligand_list: str | None = typer.Option(
        None, "--ligand-list", help="Path to ligand list file"
    ),
    multiplicity: int = typer.Option(
        1,
        "--multiplicity",
        min=1,
        help="Number of conformers per blob per ligand",
    ),
    max_parallel_multiplicity: int = typer.Option(
        8,
        "--max-parallel-multiplicity",
        min=1,
        help="Maximum conformers generated in one forward pass",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Detect ligands in cryo-EM density maps using Emap2lig."""
    # Enforce platform accelerator inference before loading config.
    try:
        _require_accelerator(gpu)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--gpu") from exc

    # Set random seed
    seed_everything(seed)

    # Load configuration
    cfg = load_config(
        gpu=gpu, detection_batch_size=detection_batch_size, contour_level=contour_level
    )

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Detect ligand blobs
    status, blobs_dir = detect_ligand_objects(input_map, output_path, cfg, emdb_id)

    # If detection failed, exit
    if status != 0:
        logger.error("Ligand object detection failed. Exiting.")
        return status

    # Check if ligand list is provided
    if ligand_list is None:
        logger.error("Ligand list not provided. Exiting.")
        return 1

    # Parse ligand list (supports both simple and YAML formats)
    ligand_records = parse_ligand_list(Path(ligand_list))

    # Stage 2: Run structure modeling
    return run_structure_modeling(
        blobs_dir=blobs_dir,
        output_dir=output_path,
        ligand_records=ligand_records,
        cfg=cfg,
        gpu=gpu,
        multiplicity=multiplicity,
        max_parallel_multiplicity=max_parallel_multiplicity,
    )


if __name__ == "__main__":
    app()
