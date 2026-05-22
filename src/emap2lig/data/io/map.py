from pathlib import Path

import mrcfile  # type: ignore
import numpy as np

from emap2lig.data.types import MapObject  # type: ignore

from loguru import logger


def _parse_mrc(
    path: Path, emdb_id: str | None = None, verbose: bool = False
) -> MapObject:
    with mrcfile.open(path, permissive=True, mode="r") as mrc:
        if verbose:
            mrc.print_header()

        # Get the data with correct dtype based on MODE
        mode = mrc.header.mode
        grid_data = np.array(mrc.data.copy())
        header = mrc.header

        # Convert to correct dtype based on MODE
        if mode == 0:  # int8
            grid_data = grid_data.astype(np.int8)
        elif mode == 1:  # int16
            grid_data = grid_data.astype(np.int16)
        elif mode == 6:  # uint16
            grid_data = grid_data.astype(np.uint16)
        else:
            grid_data = grid_data.astype(np.float32)

        voxel_size = np.array(mrc.voxel_size.tolist(), dtype=np.float32)
        origin = np.array(header.origin.tolist(), dtype=np.float32)

        n_crs_start = np.array(
            [header.nxstart, header.nystart, header.nzstart], dtype=np.float32
        )
        angle = np.asarray(
            [header.cellb.alpha, header.cellb.beta, header.cellb.gamma],
            dtype=np.float32,
        )

        # Check orthogonal
        if not np.allclose(angle, 90.0):
            raise ValueError("Map is not orthogonal")

        # Reorder
        map_crs = np.subtract([header.mapc, header.mapr, header.maps], 1)
        sort = np.array([0, 1, 2], dtype=np.int64)
        for i in range(3):
            sort[map_crs[i]] = i

        n_xyz_start = np.array([n_crs_start[i] for i in sort])
        grid_data = np.transpose(grid_data, axes=2 - sort[::-1])

        # MRC2000 compatibility
        if np.isclose(origin, 0.0).all():
            origin += np.multiply(n_xyz_start, voxel_size)
            logger.debug(
                f"Origin is zero. Calculating origin from n_xyz_start and voxel size. New origin is {origin}"
            )

        return MapObject(
            grid_data=grid_data,
            voxel_size=voxel_size,
            global_origin=origin,
            emdb_id=emdb_id,
        )


def parse_mrc(
    path: Path, emdb_id: str | None = None, verbose: bool = False
) -> MapObject:
    """Parse an MRC file, with support for gzipped files.

    Args:
        path: Path to the MRC file (can be .gz compressed)
        emdb_id: Optional EMDB ID
        verbose: Whether to print verbose information

    Returns:
        MapObject containing the parsed map data
    """
    return _parse_mrc(path, emdb_id, verbose)


def to_mrc(
    mrc_obj: MapObject, path: Path, verbose: bool = False, molstar_compat: bool = True
) -> None:
    """Save an Map to a file.

    Args:
        mrc_path: Path to save the MRC file.
        mrc_obj: MRCObject to save.
        verbose: Whether to print verbose messages.
        molstar_compat: If True, ensures saved format is compatible with mol* viewer.
    """
    # Determine optimal dtype based on dataset range
    optimal_dtype = determine_optimal_dtype(mrc_obj.grid_data, molstar_compat)

    if optimal_dtype != mrc_obj.grid_data.dtype:
        logger.debug(
            f"Converting data to optimal dtype: {optimal_dtype} and save to {path}",
        )

    # Convert dataset to optimal dtype
    grid_data = mrc_obj.grid_data.astype(optimal_dtype)

    with mrcfile.new(path, data=grid_data, overwrite=True) as mrc:
        mrc.header.origin = tuple(mrc_obj.global_origin)
        mrc.voxel_size = tuple(mrc_obj.voxel_size)
        if verbose:
            mrc.print_header()
            logger.debug(f"Saved to {path}")


def determine_optimal_dtype(
    grid_data: np.ndarray, molstar_compat: bool = True
) -> np.dtype:
    """Determine the optimal dtype for the voxel data according to MRC2014 spec.

    The MRC2014 format supports the following modes:
        MODE 0: 8-bit signed integer (range -128 to 127)
        MODE 1: 16-bit signed integer
        MODE 2: 32-bit signed real
        MODE 6: 16-bit unsigned integer
        MODE 12: 16-bit float (IEEE754) - only when molstar_compat=False

    Args:
        grid_data: Input voxel data.
        molstar_compat: If True, ensures dtype is compatible with mol* viewer
                       (avoids float16).

    Returns:
        Optimal numpy dtype for the data.
    """
    # Handle floating point
    if np.issubdtype(grid_data.dtype, np.floating):
        if not molstar_compat:
            max_val = np.max(np.abs(grid_data))
            if max_val < np.finfo(np.float16).max:
                return np.dtype(np.float16)  # MODE 12
        return np.dtype(np.float32)  # MODE 2

    # Handle integers
    if np.issubdtype(grid_data.dtype, np.integer):
        max_val = np.max(grid_data)
        min_val = np.min(grid_data)

        # Check if input is unsigned
        if np.issubdtype(grid_data.dtype, np.unsignedinteger):
            if max_val <= np.iinfo(np.uint8).max:
                # No unsigned 8-bit in MRC2014, use signed 8-bit if possible
                if max_val <= 127:
                    return np.dtype(np.int8)  # MODE 0
                return np.dtype(np.uint16)  # MODE 6
            elif max_val <= np.iinfo(np.uint16).max:
                return np.dtype(np.uint16)  # MODE 6
            else:
                return np.dtype(np.float32)  # MODE 2
        else:  # Signed integers
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                return np.dtype(np.int8)  # MODE 0
            elif (
                min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max
            ):
                return np.dtype(np.int16)  # MODE 1
            else:
                return np.dtype(np.float32)  # MODE 2

    # Default to float32 (MODE 2) for unknown types
    return np.dtype(np.float32)
