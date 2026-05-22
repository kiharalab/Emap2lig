from typing import Literal

import numpy as np
import torch

from .download import get_contour_level
from .types import MapObject

from loguru import logger


def get_unified_mrc(
    mrc_obj: MapObject,
    emdb_id: str | None = None,
    apix: float = 1.0,
    contour_ratio: float = 0.5,
    min_spatial_size: int | None = None,
    extended_val: int | None = None,
    contour_level: float | None = None,
) -> MapObject:
    """Get a unified MRC object with standardized parameters.

    Args:
        mrc_obj: Input MRC object.
        emdb_id: EMDB ID for contour level lookup.
        apix: Target voxel size.
        contour_ratio: Ratio to scale the contour level.
        min_spatial_size: Minimum size for each dimension after cropping.
        extended_val: Number of voxels to extend the bounding box.
        contour_level: Optional explicit contour level.

    Returns:
        Processed MRC object with standardized parameters.
    """
    if contour_level is None and emdb_id is not None:
        contour_level = get_contour_level(emdb_id)

    # Resample to target voxel size
    mrc_obj = resample_mrc(mrc_obj, apix)

    # Normalize the map
    if contour_level is not None:
        mrc_obj = normalize_mrc(
            mrc_obj, contour_level=float(contour_level * contour_ratio)
        )
    else:
        mrc_obj = normalize_mrc(mrc_obj)

    # Crop to content
    if min_spatial_size is not None or extended_val is not None:
        mrc_obj = crop_mrc(
            mrc_obj,
            min_spatial_size=min_spatial_size if min_spatial_size is not None else 64,
            extended_val=extended_val,
        )

    return mrc_obj


@torch.no_grad()
def resample_mrc(
    mrc_object: MapObject,
    apix: float | tuple[float, float, float],
    use_gpu: bool = False,
) -> MapObject:
    """Resample an MapObject to a new voxel size.

    Args:
        mrc_object: Input MapObject.
        apix: Target voxel size.
        use_gpu: Whether to use GPU for resampling.

    Returns:
        Resampled MapObject.
    """
    original_voxel_size = mrc_object.voxel_size
    original_grid_size = mrc_object.grid_data.shape
    if isinstance(apix, float):
        target_voxel_size = np.array([apix, apix, apix], dtype=np.float32)
    else:
        target_voxel_size = np.array(apix, dtype=np.float32)

    target_grid_size = np.floor(
        mrc_object.grid_data.shape
        * original_voxel_size[::-1]
        / target_voxel_size[::-1],
    ).astype(np.int32)
    logger.debug(f"voxel size: {original_voxel_size} -> {target_voxel_size}")
    logger.debug(f"grid size: {original_grid_size} -> {target_grid_size}")

    with torch.autocast("cuda", enabled=use_gpu):
        z = (
            torch.arange(0, target_grid_size[0], dtype=torch.float32)
            / original_voxel_size[2]
            * target_voxel_size[2]
            / (original_grid_size[0] - 1)
            * 2
            - 1
        )
        y = (
            torch.arange(0, target_grid_size[1], dtype=torch.float32)
            / original_voxel_size[1]
            * target_voxel_size[1]
            / (original_grid_size[1] - 1)
            * 2
            - 1
        )
        x = (
            torch.arange(0, target_grid_size[2], dtype=torch.float32)
            / original_voxel_size[0]
            * target_voxel_size[0]
            / (original_grid_size[2] - 1)
            * 2
            - 1
        )

        new_grid = torch.stack(
            torch.meshgrid(x, y, z, indexing="ij"),
            dim=-1,
        ).unsqueeze(0)

        original_data = (
            torch.from_numpy(mrc_object.grid_data).unsqueeze(0).unsqueeze(0).float()
        )  # volumetric input
        if use_gpu:
            # Check if GPU is available
            if torch.cuda.is_available():
                logger.debug("CUDA is available. Using GPU for resampling.")
                device = torch.device("cuda")
                original_data = original_data.to(device)
            else:
                logger.warning("GPU is not available. Using CPU for resampling.")

        target_data = (
            torch.nn.functional.grid_sample(
                original_data, new_grid, mode="bilinear", align_corners=True
            )
            .cpu()
            .numpy()[0, 0]
            .transpose(2, 1, 0)
        )

        logger.debug(f"Resampled data shape: {target_data.shape}")

    return MapObject(
        grid_data=target_data,
        voxel_size=target_voxel_size,
        global_origin=mrc_object.global_origin,
    )


def normalize_mrc(
    mrc_object: MapObject,
    *,
    contour_level: float = 0.0,
    quantile_fraction: float = 0.98,
) -> MapObject:
    """Normalize an MapObject.

    Args:
        mrc_object: Input MRCObject.
        contour_level: Contour level for thresholding.
        quantile_fraction: Quantile fraction for upper bound.

    Returns:
        Normalized MapObject.
    """
    grid = mrc_object.grid_data
    grid[grid < 0] = 0

    # Thresholding using contour level
    grid[grid < contour_level] = 0

    if 1.0 > quantile_fraction > 0.0:
        quantile_val = np.quantile(grid[grid > 0], quantile_fraction)
        logger.info(f"Quantile-{quantile_fraction} value: {quantile_val}")
        grid = (grid - grid.min()) / (quantile_val - grid.min())
    else:
        logger.debug("No quantile value is calculated.")
        grid = (grid - grid.min()) / (grid.max() - grid.min())

    logger.debug(f"Normalized density range: min {grid.min()}, max {grid.max()}")

    return MapObject(
        grid_data=grid,
        voxel_size=mrc_object.voxel_size,
        global_origin=mrc_object.global_origin,
    )


def crop_mrc(
    mrc_object: MapObject,
    min_spatial_size: int = 64,
    pad_mode: Literal["symmetric", "end"] = "symmetric",
    extended_val: int | None = None,
) -> MapObject:
    """
    Crop an MRCObject to its non-zero content and optionally pad to minimum size.

    Args:
        mrc_object: Input MRCObject.
        min_spatial_size: Minimum size for each dimension after cropping.
        pad_mode: Padding mode.
            - "symmetric": Add padding symmetrically on both sides
            - "end": Add padding only at the end of each dimension
        extended_val: Number of voxels to extend the bounding box in each direction.

    Returns:
        Cropped MRCObject with adjusted origin and optional padding.
    """
    # Find non-zero content boundaries
    grid = mrc_object.grid_data
    grid_shape = grid.shape
    global_origin = mrc_object.global_origin

    indices = np.nonzero(grid)
    if extended_val is None:
        extended_val = 0

    # Calculate bounding box with extension
    min_indices = [max(0, np.min(dim) - extended_val) for dim in indices]
    max_indices = [
        min(grid.shape[i], np.max(dim) + 1 + extended_val)
        for i, dim in enumerate(indices)
    ]

    # Create slice objects for cropping
    bbox = tuple(slice(start, end) for start, end in zip(min_indices, max_indices))
    cropped_grid = grid[bbox]

    # Calculate new origin based on cropping
    bbox_xyz_start = np.array(min_indices[::-1], dtype=np.float32)
    new_origin = mrc_object.global_origin + np.multiply(
        bbox_xyz_start, mrc_object.voxel_size
    )

    # Handle padding if necessary
    needed_padding = [max(0, min_spatial_size - size) for size in cropped_grid.shape]

    if any(needed_padding):
        logger.warning(
            f"Padding required: shape {cropped_grid.shape} -> minimum {min_spatial_size}"
        )

        pad_width = []
        for pad_size in needed_padding:
            if pad_mode == "symmetric":
                # Distribute padding evenly on both sides
                half_pad = pad_size // 2
                pad_width.append((half_pad, pad_size - half_pad))
            else:  # "end" mode
                pad_width.append((0, pad_size))

        cropped_grid = np.pad(
            cropped_grid, pad_width, mode="constant", constant_values=0
        )

        # Adjust origin for symmetric padding
        if pad_mode == "symmetric":
            origin_offset = np.array([p[0] for p in pad_width][::-1], dtype=np.float32)
            new_origin -= np.multiply(origin_offset, mrc_object.voxel_size)

    logger.debug(f"Shape: {grid_shape} -> {cropped_grid.shape}")
    logger.debug(f"Global Origin: {global_origin} -> {new_origin}")

    return MapObject(
        grid_data=cropped_grid,
        voxel_size=mrc_object.voxel_size,
        global_origin=new_origin,
    )


def crop_mrcs(
    mrc_objects: dict[str, MapObject],
    array_list: list[np.ndarray] | None = None,
    excluded_keys: list[str] | None = None,
    min_spatial_size: int = 64,
    pad_mode: Literal["symmetric", "end"] = "symmetric",
    extended_val: int | None = None,
) -> tuple[dict[str, MapObject], list[np.ndarray]]:
    """Crop multiple MRC objects to a common bounding box and adjust arrays accordingly.

    Args:
        mrc_objects: Dictionary of MRCObject instances.
        array_list: List of arrays to crop alongside MRC objects. Assume the last three dimensions are spatial.
        excluded_keys: Keys to exclude from cropping.
        min_spatial_size: Minimum size for each dimension after cropping.
        pad_mode: Padding mode for reaching minimum size.
            - "symmetric": Add padding symmetrically on both sides
            - "end": Add padding only at the end of each dimension
        extended_val: Number of voxels to extend the bounding box in each direction.

    Returns:
        Tuple of cropped MRC objects (dict) and cropped arrays (list).

    Raises:
        ValueError: If input validation fails or no valid indices are found.
    """
    # Initialize defaults and validate input
    array_list = array_list or []
    excluded_keys = excluded_keys or ["map"]
    if extended_val is None:
        extended_val = 0

    # Validate origins
    origins = [mrc.global_origin for mrc in mrc_objects.values()]
    if not all(np.allclose(origin, origins[0]) for origin in origins):
        raise ValueError("All MRC objects must have the same global origin.")

    # Get the reference MRC object and shape
    reference_mrc = next(iter(mrc_objects.values()))
    voxel_shape = reference_mrc.grid_data.shape

    # Step 1: Find bounding box of foreground objects and decide padding values

    # Find bounding box across all non-excluded MRCs
    global_min_indices = None
    global_max_indices = None

    for key, mrc in mrc_objects.items():
        if key in excluded_keys or len(np.nonzero(mrc.grid_data)[0]) == 0:
            continue

        indices = np.nonzero(mrc.grid_data)
        min_indices = [int(np.min(dim)) for dim in indices]
        max_indices = [int(np.max(dim)) + 1 for dim in indices]

        if global_min_indices is None:
            global_min_indices = min_indices
            global_max_indices = max_indices
        else:
            global_min_indices = [
                min(g_min, m_min)
                for g_min, m_min in zip(global_min_indices, min_indices)
            ]
            global_max_indices = [
                max(g_max, m_max)
                for g_max, m_max in zip(global_max_indices, max_indices)
            ]

    # If no valid indices were found, raise an error
    if global_min_indices is None or global_max_indices is None:
        raise ValueError("No valid indices found in any MRC object")

    # Apply extension to bounding box
    global_min_indices = [max(0, idx - extended_val) for idx in global_min_indices]
    global_max_indices = [
        min(voxel_shape[i], idx + extended_val)
        for i, idx in enumerate(global_max_indices)
    ]

    # Check if the bounding box is valid
    if any(
        g_min >= g_max
        for g_min, g_max in zip(global_min_indices, global_max_indices, strict=False)
    ):
        raise ValueError(
            "Invalid bounding box: minimum indices greater than or equal to maximum indices"
        )

    # Calculate content size and determine needed padding
    content_size = [
        end - start
        for start, end in zip(global_min_indices, global_max_indices, strict=False)
    ]

    # Determine if we need padding to meet min_spatial_size
    needed_padding = [max(0, min_spatial_size - size) for size in content_size]

    # Determine padding values based on pad_mode
    pad_left = []
    pad_right = []

    for pad_size in needed_padding:
        if pad_mode == "symmetric":
            # Distribute padding evenly on both sides
            half_pad = pad_size // 2
            pad_left.append(half_pad)
            pad_right.append(pad_size - half_pad)
        else:  # "end" mode
            pad_left.append(0)
            pad_right.append(pad_size)

    # Calculate final start indices after padding
    final_start_indices = [
        max(0, min_idx - pad_l) for min_idx, pad_l in zip(global_min_indices, pad_left)
    ]

    # Calculate final size after padding
    final_size = [
        content_size[i] + pad_left[i] + pad_right[i] for i in range(len(content_size))
    ]

    # Calculate final end indices
    final_end_indices = [
        start + size for start, size in zip(final_start_indices, final_size)
    ]

    if any(needed_padding):
        logger.warning(
            f"Padding required: shape {content_size} -> minimum {min_spatial_size}",
        )

    # Calculate the origin adjustment based on final start indices
    bbox_xyz_start = np.array(final_start_indices[::-1], dtype=np.float32)
    final_origin = reference_mrc.global_origin + np.multiply(
        bbox_xyz_start,
        reference_mrc.voxel_size,
    )

    # Step 2: Create the final bounding box for cropping
    bbox = tuple(
        slice(start, end)
        for start, end in zip(final_start_indices, final_end_indices, strict=False)
    )

    # Step 3: Crop all objects based on the bounding box
    result_mrcs = {}
    for key, mrc in mrc_objects.items():
        # Create safe slices that respect the grid dimensions
        safe_bbox = tuple(
            slice(
                min(s.start, mrc.grid_data.shape[i]),
                min(s.stop, mrc.grid_data.shape[i]),
            )
            for i, s in enumerate(bbox)
        )

        # Extract the grid data using the safe bbox
        cropped_data = mrc.grid_data[safe_bbox]

        # If the cropped data is smaller than the target size, pad it
        cropped_shape = cropped_data.shape
        target_shape = tuple(
            end - start for start, end in zip(final_start_indices, final_end_indices)
        )

        if cropped_shape != target_shape:
            # Calculate padding needed for each dimension
            pad_width = []
            for i in range(len(cropped_shape)):
                pad_before = 0
                pad_after = target_shape[i] - cropped_shape[i]
                pad_width.append((pad_before, pad_after))

            # Pad the cropped data
            cropped_data = np.pad(
                cropped_data,
                tuple(pad_width),
                mode="constant",
                constant_values=0,
            )

        # Create new MapObject with the cropped and padded data
        result_mrcs[key] = MapObject(
            grid_data=cropped_data,
            voxel_size=mrc.voxel_size,
            global_origin=final_origin,
        )

    # Process array_list similarly
    result_array_list = []
    for array in array_list:
        # Determine the dimensions to crop
        leading_dims = array.ndim - 3

        # Create safe slices for the array
        array_bbox = tuple([...]) + tuple(
            slice(
                min(s.start, array.shape[leading_dims + i]),
                min(s.stop, array.shape[leading_dims + i]),
            )
            for i, s in enumerate(bbox)
        )

        # Crop the array
        cropped_array = array[array_bbox]

        # Check if padding is needed
        cropped_shape = cropped_array.shape[-3:]
        target_shape = tuple(
            end - start for start, end in zip(final_start_indices, final_end_indices)
        )

        if cropped_shape != target_shape:
            # Calculate padding needed for each spatial dimension
            pad_width = [(0, 0)] * leading_dims
            for i in range(3):
                pad_before = 0
                pad_after = target_shape[i] - cropped_shape[i]
                pad_width.append((pad_before, pad_after))

            # Pad the cropped array
            cropped_array = np.pad(
                cropped_array,
                tuple(pad_width),
                mode="constant",
                constant_values=0,
            )

        result_array_list.append(cropped_array)

    # Log results
    logger.debug(f"Origin: {reference_mrc.global_origin} -> {final_origin}")
    logger.debug(
        f"Spatial Size: {reference_mrc.grid_data.shape} -> {result_mrcs[next(iter(result_mrcs))].grid_data.shape}",
    )
    logger.debug(
        f"Bounding box: {list(zip(final_start_indices, final_end_indices, strict=False))}",
    )

    return result_mrcs, result_array_list
