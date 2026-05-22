import torch
from loguru import logger
from pathlib import Path
from rdkit import Chem
from torch.utils.data import Dataset
import numpy as np
import gemmi

from emap2lig.data.types import DensityObject, LigandObject
from emap2lig.data.const import num_elements
from emap2lig.data.transforms import center_random_augmentation

# Set default pickle properties
pickle_option = Chem.PropertyPickleOptions.AllProps
Chem.SetDefaultPickleProperties(pickle_option)


def _get_instance_center(
    instance_mask: torch.Tensor,
    voxel_size: torch.Tensor,
    global_origin: torch.Tensor,
) -> torch.Tensor:
    """Get the center coordinates from instance mask.

    This is similar to tensor_to_point_cloud but returns only the center.

    Parameters
    ----------
    instance_mask : torch.Tensor
        Instance mask of shape [D, H, W]
    voxel_size : torch.Tensor
        Voxel size of shape [3]
    global_origin : torch.Tensor
        Global origin of shape [3]

    Returns
    -------
    torch.Tensor
        Center coordinates of shape [3]
    """
    D, H, W = instance_mask.shape

    # Create a grid of indices for the spatial dimensions
    z, y, x = torch.meshgrid(
        torch.arange(D, device=instance_mask.device),
        torch.arange(H, device=instance_mask.device),
        torch.arange(W, device=instance_mask.device),
        indexing="ij",
    )

    # Stack indices in the order (x, y, z), shape: [D, H, W, 3]
    indices = torch.stack([x, y, z], dim=-1).float()

    # Convert indices to coordinates, shape: [D, H, W, 3]
    coordinates = indices * voxel_size + global_origin

    # Flatten coordinates and instance mask for indexing
    flat_coordinates = coordinates.reshape(-1, 3)  # shape: [N, 3] where N = D*H*W
    flat_mask = instance_mask.flatten()  # shape: [N]

    # Get coordinates where instance mask is True
    valid_coords = flat_coordinates[
        flat_mask > 0
    ]  # shape: [M, 3] where M is number of valid points

    if len(valid_coords) == 0:
        # If no valid points, return the center of the volume
        center_idx = torch.tensor(
            [W // 2, H // 2, D // 2], dtype=torch.float32, device=instance_mask.device
        )
        return center_idx * voxel_size + global_origin

    # Calculate the center of valid coordinates
    center = valid_coords.mean(dim=0)  # shape: [3]
    return center


class LigandModelingDataset(Dataset):
    def __init__(
        self,
        density_object_list: list[Path],
        ref_mol_dir: Path,
        max_atoms: int = 200,
        pad_to_max: bool = True,
        multiplicity: int = 16,
    ):
        """
        Initialize the ligand modeling dataset.

        Args:
            density_object_list: List of paths to LigandDensityObject NPZ files
            ref_mol_dir: Directory containing RefMolecularObject NPZ files
            max_atoms: Maximum number of atoms to support
            pad_to_max: If True, pad arrays to max_atoms
            multiplicity: Number of prompt points to sample per instance mask
        """
        self.density_object_list = density_object_list
        self.max_atoms = max_atoms
        self.pad_to_max = pad_to_max
        self.multiplicity = multiplicity

        # Load all reference molecular objects
        self.ref_mol_paths = sorted(list(Path(ref_mol_dir).glob("*.npz")))
        self.ref_mol_names = [path.stem for path in self.ref_mol_paths]

        # Filter valid pairs based on blob associations and ratio criteria
        logger.info("Filtering valid density-ligand pairs with ratio between 5-25...")
        self._filter_valid_pairs()
        logger.info(
            f"Found {len(self.valid_pairs)} valid pairs out of {len(density_object_list) * len(self.ref_mol_paths)} possible combinations"
        )

    def _filter_valid_pairs(self):
        """Filter valid density-ligand pairs based on ratio of non-zero elements to atom count and blob associations."""
        self.valid_pairs = []

        for ref_mol_idx, ref_mol_path in enumerate(self.ref_mol_paths):
            ref_mol_object = LigandObject.load(ref_mol_path)
            n_atoms = len(ref_mol_object.atoms)

            # Handle numpy array case where None gets saved as a 0-d array
            blobs = ref_mol_object.blobs
            if blobs is None or (
                isinstance(blobs, np.ndarray)
                and blobs.ndim == 0
                and blobs.item() is None
            ):
                # No specific blob associations - traverse all available blobs
                for density_idx, density_path in enumerate(self.density_object_list):
                    density_object = DensityObject.load(density_path)
                    non_zero_count = np.count_nonzero(density_object.instance_grid)
                    ratio = non_zero_count / n_atoms

                    # Only include pairs with ratio between 5 and 25
                    if 5 <= ratio <= 25:
                        self.valid_pairs.append((density_idx, ref_mol_idx))
                        logger.debug(
                            f"Valid pair: {density_object.object_id}-{ref_mol_object.name}, ratio: {ratio:.2f}"
                        )
                    else:
                        logger.debug(
                            f"Invalid pair: {density_object.object_id}-{ref_mol_object.name}, ratio: {ratio:.2f}"
                        )
            else:
                # Specific blob associations - only consider blobs in the list
                # Handle both list and numpy array cases
                blob_list = blobs.tolist() if isinstance(blobs, np.ndarray) else blobs
                for blob_id in blob_list:
                    # Find density object with matching blob_id
                    matching_density_idx = None
                    for density_idx, density_path in enumerate(
                        self.density_object_list
                    ):
                        density_object = DensityObject.load(density_path)
                        if density_object.object_id == blob_id:
                            matching_density_idx = density_idx
                            break

                    if matching_density_idx is not None:
                        density_object = DensityObject.load(
                            self.density_object_list[matching_density_idx]
                        )
                        non_zero_count = np.count_nonzero(density_object.instance_grid)
                        ratio = non_zero_count / n_atoms

                        # Only include pairs with ratio between 5 and 25
                        if 5 <= ratio <= 25:
                            self.valid_pairs.append((matching_density_idx, ref_mol_idx))
                            logger.debug(
                                f"Valid pair: {density_object.object_id}-{ref_mol_object.name}, ratio: {ratio:.2f}"
                            )
                        else:
                            logger.debug(
                                f"Invalid pair: {density_object.object_id}-{ref_mol_object.name}, ratio: {ratio:.2f}"
                            )
                    else:
                        logger.warning(
                            f"Blob {blob_id} specified for ligand {ref_mol_object.name} not found in density objects"
                        )

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        # Get density and reference molecule indices from valid pairs
        density_idx, ref_mol_idx = self.valid_pairs[idx]

        # Load density object
        density_object_path = self.density_object_list[density_idx]
        density_object = DensityObject.load(density_object_path)

        # Load reference molecular object
        ref_mol_path = self.ref_mol_paths[ref_mol_idx]
        ref_mol_object = LigandObject.load(ref_mol_path)

        # Create a unique identifier
        ligand_name = ref_mol_object.name
        object_id = density_object.object_id
        identifier = f"{object_id}_{ligand_name}"

        # Create feature dictionary by combining density and reference information
        # Use keys matching the reference implementation and model expectations
        features = {
            # Structure info from reference molecular object
            "smiles_path": ref_mol_path,
            "atom_names": ref_mol_object.atom_names,
            "atoms": ref_mol_object.atoms,
            "bonds": ref_mol_object.bonds,
            "class_name": ref_mol_object.name,
            "identifier": identifier,
            "symmetries": ref_mol_object.symmetries,
            "smiles": ref_mol_object.smiles,
            # Density info from density object (using keys matching model expectations)
            "object_id": density_object.object_id,
            "input_density": torch.from_numpy(density_object.density_grid)
            .float()
            .unsqueeze(0),  # [1, D, H, W]
            "voxel_size": torch.from_numpy(density_object.voxel_size).float(),
            "global_origin": torch.from_numpy(density_object.global_origin).float(),
        }

        # Create instance mask for centering and prompt generation (not passed to model)
        instance_mask = (
            torch.from_numpy(density_object.instance_grid) > 0.0
        ).long()  # [D, H, W]

        # Add additional features for model input
        features = self._process_features(features)
        features = self._center_random_augment_ref_coords(features)
        features = self._center_groundtruth_coords(features, instance_mask)

        # Generate prompt points from instance mask
        features = self._generate_prompt_points(features, instance_mask)

        return features

    def _center_random_augment_ref_coords(
        self, features_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Center + (optionally) randomly augment reference coordinates per residue.

        The reference model applies `center_random_augmentation(..., augmentation=True)`
        per residue even during prediction, so we mirror that behavior here to match
        the training/prediction distribution.
        """
        if "ref_pos" not in features_dict:
            return features_dict

        ref_pos = features_dict["ref_pos"]  # [max_len, 3]
        atom_mask = features_dict["atom_mask"]  # [max_len]
        residue_id = features_dict["residue_id"]  # [max_len]

        # Get unique residue IDs (excluding padded atoms)
        valid_residue_ids = residue_id[atom_mask.bool()].unique()

        # Apply augmentation per residue
        augmented_ref_pos = ref_pos.clone()

        for res_id in valid_residue_ids:
            # Get atoms belonging to this residue
            residue_mask = (residue_id == res_id) & atom_mask.bool()
            residue_indices = torch.where(residue_mask)[0]

            if len(residue_indices) > 0:
                # Extract coordinates for this residue
                residue_coords = ref_pos[residue_indices]  # [n_atoms_in_residue, 3]
                residue_atom_mask = torch.ones(
                    (1, len(residue_indices)),
                    dtype=ref_pos.dtype,
                    device=ref_pos.device,
                )

                # Apply the same transform used in the reference training pipeline
                augmented_residue_coords = center_random_augmentation(
                    residue_coords[None],
                    residue_atom_mask,
                    centering=True,
                    augmentation=True,
                )[0]

                augmented_ref_pos[residue_indices] = augmented_residue_coords

        features_dict["ref_pos"] = augmented_ref_pos

        return features_dict

    def _center_groundtruth_coords(
        self,
        features_dict: dict[str, torch.Tensor],
        instance_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Center coordinates using the instance mask center (same as training).

        Args:
            features_dict: Dictionary containing feature tensors.
            instance_mask: Binary instance mask of shape [D, H, W].
        """
        voxel_size = features_dict["voxel_size"]  # [3]
        global_origin = features_dict["global_origin"]  # [3]

        center = _get_instance_center(instance_mask, voxel_size, global_origin)  # [3]

        # Center the coordinates using instance mask center
        features_dict["groundtruth_center"] = center
        features_dict["global_origin"] = features_dict["global_origin"] - center

        return features_dict

    def _generate_prompt_points(
        self,
        features_dict: dict[str, torch.Tensor],
        instance_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate prompt points by sampling from the instance mask.

        This samples multiple prompt points from valid voxels in the instance mask
        for use during inference. Each prompt point is a world coordinate within
        the detected blob.

        Uses the same coordinate conversion as _get_instance_center for consistency.

        Args:
            features_dict: Dictionary containing feature tensors.
            instance_mask: Binary instance mask of shape [D, H, W].

        Returns:
            Updated features_dict with prompt_points added.
        """
        voxel_size = features_dict["voxel_size"]  # [3]
        global_origin = features_dict["global_origin"]  # [3]

        D, H, W = instance_mask.shape

        # Create a grid of indices for the spatial dimensions
        z, y, x = torch.meshgrid(
            torch.arange(D, device=instance_mask.device),
            torch.arange(H, device=instance_mask.device),
            torch.arange(W, device=instance_mask.device),
            indexing="ij",
        )

        # Stack indices in the order (x, y, z), shape: [D, H, W, 3]
        indices = torch.stack([x, y, z], dim=-1).float()

        # Convert indices to world coordinates, shape: [D, H, W, 3]
        coordinates = indices * voxel_size + global_origin

        # Flatten coordinates and instance mask for indexing
        flat_coordinates = coordinates.reshape(-1, 3)  # shape: [N, 3] where N = D*H*W
        flat_mask = instance_mask.flatten()  # shape: [N]
        valid_indices = torch.where(flat_mask > 0)[0]  # [n_valid]

        if len(valid_indices) == 0:
            # If no valid voxels, use the center of the volume
            center_idx = torch.tensor(
                [W // 2, H // 2, D // 2],
                dtype=torch.float32,
                device=instance_mask.device,
            )
            center = center_idx * voxel_size + global_origin
            prompt_points = center.unsqueeze(0).repeat(self.multiplicity, 1)
        else:
            # Sample multiplicity prompt points from valid voxels
            prompt_points = []
            for _ in range(self.multiplicity):
                # Randomly select a valid voxel index
                random_idx = torch.randint(0, len(valid_indices), (1,)).item()
                selected_voxel_idx = valid_indices[random_idx]

                # Get the world coordinate from flattened coordinates
                prompt_point = flat_coordinates[selected_voxel_idx].clone()

                # Add small random noise (0 to 0.5) for diversity
                noise = 0.5 * torch.rand(
                    3, dtype=prompt_point.dtype, device=prompt_point.device
                )
                prompt_point = prompt_point + noise

                prompt_points.append(prompt_point)

            prompt_points = torch.stack(prompt_points, dim=0)  # [multiplicity, 3]

        features_dict["prompt_points"] = prompt_points

        return features_dict

    def _process_features(self, features):
        """
        Add additional features needed for model input.

        This creates atom features matching the reference implementation with 149 dimensions:
        - name (4) + element (128) + charge (1) + chirality (7) + in_ring (4) + residue_id (1)
        - element_is_metal (1) + covalent_r (1) + vdw_r (1) + atom_weight (1)

        Args:
            features: Dictionary of base features

        Returns:
            Dictionary with additional features
        """
        # Convert atom data to tensor features
        atoms = features["atoms"]
        n_atoms = len(atoms)

        # Check if number of atoms exceeds max_atoms
        if n_atoms > self.max_atoms:
            raise ValueError(f"Too many atoms ({n_atoms} > {self.max_atoms})")

        # Determine the actual length to use (max_atoms if padding, actual n_atoms otherwise)
        max_len = self.max_atoms if self.pad_to_max else n_atoms

        # Create atom feature tensor with 149 dimensions to match reference implementation
        # name (4) + element (128) + charge (1) + chirality (7) + in_ring (4) + residue_id (1)
        # + element_is_metal (1) + covalent_r (1) + vdw_r (1) + atom_weight (1) = 149
        atom_feature = np.zeros((max_len, 149), dtype=np.float32)
        ref_pos = np.zeros((max_len, 3), dtype=np.float32)
        atom_mask = np.zeros(max_len, dtype=bool)
        residue_id = np.zeros(max_len, dtype=np.int32)

        for i in range(n_atoms):
            atom = atoms[i]

            # One-hot encode element (atomic number)
            element_idx = atom["element"]
            element_one_hot = np.zeros(num_elements, dtype=np.float32)
            if element_idx < num_elements:
                element_one_hot[element_idx] = 1.0

            # Get gemmi element properties
            element = gemmi.Element(element_idx)
            element_is_metal = float(element.is_metal)
            covalent_r = float(element.covalent_r)
            vdw_r = float(element.vdw_r)
            atom_weight = float(element.weight)

            # Concatenate features (149 total)
            atom_feature[i] = np.concatenate(
                [
                    atom["name"],  # 4
                    element_one_hot,  # 128
                    [atom["charge"]],  # 1
                    atom["chirality"],  # 7
                    atom["in_ring"],  # 4
                    [atom["residue_id"]],  # 1
                    [element_is_metal],  # 1
                    [covalent_r],  # 1
                    [vdw_r],  # 1
                    [atom_weight],  # 1
                ]
            )

            # Reference position
            ref_pos[i] = atom["ref_pos"]

            # Set atom mask to True for actual atoms
            atom_mask[i] = True

            # Store residue_id
            residue_id[i] = atom["residue_id"]

        # Create bond feature tensor
        bond_feature = np.zeros(
            (max_len, max_len, 9), dtype=np.float32
        )  # bond types (5) + in_ring (4)

        for bond in features["bonds"]:
            i, j = bond["atom_1"], bond["atom_2"]
            if i < max_len and j < max_len:  # Safety check
                bond_feature[i, j, :5] = bond["type"]
                bond_feature[j, i, :5] = bond["type"]
                bond_feature[i, j, 5:] = bond["in_ring"]
                bond_feature[j, i, 5:] = bond["in_ring"]

        # Create pair mask
        pair_mask = np.outer(atom_mask, atom_mask).astype(bool)

        # Add features to dictionary
        features["atom_feature"] = torch.tensor(atom_feature, dtype=torch.float32)
        features["bond_feature"] = torch.tensor(bond_feature, dtype=torch.float32)
        features["ref_pos"] = torch.tensor(ref_pos, dtype=torch.float32)
        features["atom_mask"] = torch.tensor(atom_mask, dtype=torch.bool)
        features["pair_mask"] = torch.tensor(pair_mask, dtype=torch.bool)
        features["n_atoms"] = torch.tensor(n_atoms)
        features["residue_id"] = torch.tensor(residue_id, dtype=torch.int32)

        return features


def collate_fn(batch):
    """
    Collate function for DataLoader.

    This handles batching of the dataset items, including proper padding
    for variable-length sequences and stacking for fixed-size tensors.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Dictionary of batched tensors
    """
    # Get all keys
    keys = batch[0].keys()

    # Initialize result dictionary
    result = {}

    # Keys that should be kept as lists (not stacked)
    list_keys = {
        "atom_names",
        "identifier",
        "symmetries",
        "smiles",
        "smiles_path",
        "object_id",
        "class_name",
        "atoms",
        "bonds",
    }

    for key in keys:
        # Skip non-tensor items
        if key in list_keys:
            result[key] = [item[key] for item in batch]
            continue

        # Get all values for this key
        values = [item[key] for item in batch]

        # Check if values are tensors
        if isinstance(values[0], torch.Tensor):
            # Check if shapes are the same
            if all(v.shape == values[0].shape for v in values):
                # Stack tensors
                result[key] = torch.stack(values)
            else:
                # Pad tensors to maximum size
                max_size = [max(dim) for dim in zip(*[v.shape for v in values])]
                padded_values = []

                for v in values:
                    if len(v.shape) == 1:
                        # 1D tensor
                        padding = max_size[0] - v.shape[0]
                        padded_v = torch.nn.functional.pad(v, (0, padding))
                    elif len(v.shape) == 2:
                        # 2D tensor
                        padding = (
                            0,
                            max_size[1] - v.shape[1],
                            0,
                            max_size[0] - v.shape[0],
                        )
                        padded_v = torch.nn.functional.pad(v, padding)
                    elif len(v.shape) == 3:
                        # 3D tensor
                        padding = (
                            0,
                            max_size[2] - v.shape[2],
                            0,
                            max_size[1] - v.shape[1],
                            0,
                            max_size[0] - v.shape[0],
                        )
                        padded_v = torch.nn.functional.pad(v, padding)
                    else:
                        # Handle other dimensions if needed
                        raise ValueError(
                            f"Unsupported tensor dimension: {len(v.shape)}"
                        )

                    padded_values.append(padded_v)

                result[key] = torch.stack(padded_values)
        else:
            # For non-tensor values
            result[key] = values

    return result
