"""The vectorized sphere mask must match the original per-atom loop."""

import numpy as np
import pytest

from emap2lig.data.simulate import add_gaussians, get_offsets_from_radius


def _loop_reference(grid, grid_indices, offsets, voxel_size, radius=1.414):
    """The original numba kernel, run as plain Python."""
    grid_shape = np.array(grid.shape)
    for position in grid_indices:
        for offset in offsets:
            center = np.floor(position + offset).astype(np.int32)
            if all(0 <= center[d] < grid_shape[d] for d in range(3)):
                shift = (center - position) * voxel_size[::-1]
                if np.sum(shift**2) <= radius**2:
                    grid[center[0], center[1], center[2]] = 1.0


@pytest.mark.parametrize("seed", range(20))
def test_add_gaussians_matches_loop(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = tuple(int(x) for x in rng.integers(6, 24, 3))
    voxel_size = rng.uniform(0.6, 1.5, 3).astype(np.float32)
    # Include atoms slightly outside the grid to exercise the bounds check.
    grid_indices = rng.uniform(-2, np.array(shape) + 2, (int(rng.integers(0, 40)), 3))
    offsets = get_offsets_from_radius(1.414, voxel_size)

    expected = np.zeros(shape, dtype=np.float32)
    actual = np.zeros(shape, dtype=np.float32)
    _loop_reference(expected, grid_indices, offsets, voxel_size)
    add_gaussians(actual, grid_indices, offsets, voxel_size)

    assert np.array_equal(actual, expected)
