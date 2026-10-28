import numpy as np


def add_gaussians(
    grid: np.ndarray,
    grid_indices: np.ndarray,
    offsets: np.ndarray,
    voxel_size: np.ndarray,
    radius: float = 1.414,
) -> None:
    """
    Mark every voxel within ``radius`` Å of an atom as 1.0 (a binary sphere).

    Args:
        grid: The 3D grid to write into, modified in place.
        grid_indices: Coordinates of atoms in grid space (zyx), shape (N, 3).
        offsets: Offset coordinates to consider around each atom, shape (K, 3).
        voxel_size: Voxel size array (xyz).
    """
    if len(grid_indices) == 0:
        return

    # (N, K, 3): candidate voxel for every atom/offset pair.
    centers = np.floor(grid_indices[:, None, :] + offsets[None, :, :]).astype(np.int32)
    in_bounds = np.all((centers >= 0) & (centers < np.array(grid.shape)), axis=-1)
    shift = (centers - grid_indices[:, None, :]) * voxel_size[::-1]
    within = np.sum(shift**2, axis=-1) <= radius**2

    hit = centers[in_bounds & within]
    grid[hit[:, 0], hit[:, 1], hit[:, 2]] = 1.0


def get_offsets_from_radius(label_radius: float, voxel_size: np.ndarray):
    int_range_x = int(np.ceil(label_radius / voxel_size[0]))
    int_range_y = int(np.ceil(label_radius / voxel_size[1]))
    int_range_z = int(np.ceil(label_radius / voxel_size[2]))
    offsets = np.array(
        [
            (dz, dy, dx)
            for dx in range(-int_range_x, int_range_x + 1)
            for dy in range(-int_range_y, int_range_y + 1)
            for dz in range(-int_range_z, int_range_z + 1)
        ],
        dtype=np.float32,
    )
    offsets = np.array(
        offsets[
            np.linalg.norm(offsets * voxel_size[::-1], axis=1)
            <= label_radius + np.sqrt(3)
        ]
    )
    return offsets


def coords_to_gaussian_mask(
    coords: np.ndarray,
    grid_shape: tuple[int, int, int],
    voxel_size: np.ndarray,
    global_origin: np.ndarray,
) -> np.ndarray:
    """
    Convert atomic coordinates to a Gaussian-blurred density map using mass-weighted Gaussians.

    Args:
        coords: Atomic coordinates of shape (N, 3)
        grid_shape: Shape of the output grid (D, H, W)
        voxel_size: Voxel size (3,)
        global_origin: Global origin (3,)

    Returns:
        Density map of shape (D, H, W) after Gaussian blurring
    """
    # Initialize empty density grid
    density_grid = np.zeros(grid_shape, dtype=np.float32)

    if len(coords) == 0:
        return density_grid

    # Convert coordinates to grid indices
    grid_indices = (coords - global_origin) / voxel_size
    grid_indices = grid_indices[:, ::-1]  # xyz -> zyx for grid indexing

    # Get offsets for efficient Gaussian calculation
    offsets = get_offsets_from_radius(1.414, voxel_size)

    # Add mass-weighted Gaussians to grid
    add_gaussians(density_grid, grid_indices, offsets, voxel_size)

    return density_grid


def calculate_mask_iou(
    mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.01
) -> float:
    """
    Calculate true IoU (Jaccard) between two density maps or binary masks.

    Args:
        mask1: First density map or binary mask
        mask2: Second density map or binary mask
        threshold: Threshold for converting to binary masks

    Returns:
        Dice coefficient value between 0 and 1
    """
    # Ensure masks are binary
    binary_mask1 = (mask1 > threshold).astype(np.float32)
    binary_mask2 = (mask2 > threshold).astype(np.float32)

    # Calculate intersection and union
    intersection = float(np.sum(binary_mask1 * binary_mask2))
    union = float(np.sum(binary_mask1) + np.sum(binary_mask2) - intersection)

    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    # Calculate true IoU (Jaccard)
    iou = intersection / union

    return float(iou)


def gaussian_blur_and_iou(
    coords: np.ndarray,
    reference_mask: np.ndarray,
    grid_shape: tuple[int, int, int],
    voxel_size: np.ndarray,
    global_origin: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, float]:
    """
    Apply Gaussian blur to coordinates and calculate IoU with reference mask.

    Args:
        coords: Atomic coordinates of shape (N, 3)
        reference_mask: Reference binary mask or density map
        grid_shape: Shape of the output grid (D, H, W)
        voxel_size: Voxel size (3,)
        global_origin: Global origin (3,)
        ligand_object: Optional LigandObject for getting atom masses
        sigma_coeff: Coefficient for Gaussian sigma calculation
        cutoff: Cutoff distance for Gaussian blur
        threshold: Threshold for binary mask conversion

    Returns:
        Tuple of (blurred_density_map, iou_score)
    """
    # Generate Gaussian-blurred density map
    density_map = coords_to_gaussian_mask(
        coords=coords,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        global_origin=global_origin,
    )

    # Calculate IoU with reference mask
    iou_score = calculate_mask_iou(density_map, reference_mask, threshold)

    return density_map, iou_score
