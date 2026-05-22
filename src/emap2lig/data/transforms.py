from __future__ import annotations

import torch
from torch.types import Device


def center_random_augmentation(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
) -> torch.Tensor:
    """Center and randomly augment atom coordinates.

    This is ported from the reference training pipeline. It is applied to the
    *reference conformer coordinates* (per-residue) to match the distribution the
    structure model was trained on.

    Parameters
    ----------
    atom_coords : torch.Tensor
        Coordinates of shape [B, N, 3] in (x, y, z) order.
    atom_mask : torch.Tensor
        Mask of shape [B, N] (0/1 or bool). Only masked-in atoms contribute to centering.
    s_trans : float
        Translation scale in angstroms, by default 1.0.
    augmentation : bool
        If True, apply random rotation and translation.
    centering : bool
        If True, subtract the masked mean coordinate.
    """
    if centering:
        # Weighted mean over valid atoms
        v = atom_mask.to(atom_coords.dtype)[..., None]  # [B, N, 1]
        denom = torch.sum(v, dim=1, keepdim=True).clamp_min(1e-6)  # [B, 1, 1]
        atom_mean = torch.sum(atom_coords * v, dim=1, keepdim=True) / denom  # [B, 1, 3]
        atom_coords = atom_coords - atom_mean

    if augmentation:
        atom_coords = randomly_rotate(atom_coords)
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * float(s_trans)
        atom_coords = atom_coords + random_trans

    return atom_coords


def randomly_rotate(coords: torch.Tensor) -> torch.Tensor:
    R = random_rotations(len(coords), dtype=coords.dtype, device=coords.device)
    return torch.einsum("bmd,bds->bms", coords, R)


def random_rotations(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """Generate random 3x3 rotation matrices."""
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def random_quaternions(
    n: int, dtype: torch.dtype | None = None, device: Device | None = None
) -> torch.Tensor:
    """Generate random unit quaternions (real part nonnegative)."""
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (real part first) to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape((*quaternions.shape[:-1], 3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise copysign for tensors."""
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)
