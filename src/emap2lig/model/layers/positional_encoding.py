import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    """
    Apply 3D sinusoidal positional encoding to point cloud data
    and return an output of the same shape as the input features.

    Parameters
    ----------
    feature_dim : int
        Dimensionality (M) of the per-point features. Must be even
        for standard sine/cosine pairing.
    base : float, optional
        Base value for the exponential decay of the frequency bands.
        Default is 50.0.
    """

    def __init__(self, feature_dim: int, base: float = 50.0):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding3D(feature_dim, base)

    def forward(
        self, point_coord: torch.Tensor, point_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for 3D positional encoding.

        Parameters
        ----------
        point_coord : torch.Tensor
            [B, N, 3] containing continuous x, y, z coordinates.
        point_feats : torch.Tensor
            [B, N, M] containing the initial per-point features.

        Returns
        -------
        conditioned_feats : torch.Tensor
            [B, N, M], the sum of the input features and the 3D positional encoding.
        """
        # Compute positional encoding without gradient tracking since it's deterministic
        with torch.no_grad():
            pos_enc_3d = self.pos_enc(point_coord)  # [B, N, M]
        conditioned_feats = (point_feats + pos_enc_3d).to(point_feats.dtype)
        return conditioned_feats


class SinusoidalPositionalEncoding3D(nn.Module):
    """
    Apply 3D sinusoidal positional encoding to point cloud data, supporting
    arbitrary output feature dimensions via padding.

    Maps 3D coordinates (x, y, z) to a feature vector of size `feature_dim`.
    If `feature_dim` is not a multiple of 6, the output is padded with zeros.

    Parameters
    ----------
    feature_dim : int
        Target dimensionality (M) of the output positional encoding.
    base : float, optional
        Base value for the exponential decay of the frequency bands.
        Default is 50.0.
    """

    def __init__(self, feature_dim: int, base: float = 50.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.base = base
        self.n_dim = 3  # Fixed for 3D

        # Calculate the largest number of features per axis usable for encoding,
        # ensuring it's an even number. This is the number of features
        # actually used per axis for sin/cos pairs.
        self.d_per_axis = (feature_dim // self.n_dim // 2) * 2

        # Add a check to ensure feature_dim is large enough
        if self.d_per_axis == 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) is too small for 3D encoding. "
                f"Need at least {2 * self.n_dim} features to encode 3 dimensions."
            )

        self.half_dim = self.d_per_axis // 2

        # Calculate the total dimension actually used by sin/cos pairs across 3 axes
        self.actual_encoded_dim = self.d_per_axis * self.n_dim

        # Calculate padding needed to reach the target feature_dim
        self.padding_dim = feature_dim - self.actual_encoded_dim

        # Precompute the frequency scaling factors (div_term)
        # Create on CPU initially, it will be moved by register_buffer
        freq = torch.arange(self.half_dim, dtype=torch.float32)
        div_term = torch.exp(
            -math.log(base) * (2 * freq / self.d_per_axis)
        )  # Shape: [half_dim]

        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [batch_size, num_points, 3]
                containing the x, y, z coordinates of points

        Returns:
            pos_encoding: Tensor of shape [batch_size, num_points, feature_dim]
                containing the positional encodings
        """
        assert coords.shape[-1] == self.n_dim, (
            f"Input coordinate tensor's last dimension ({coords.shape[-1]}) must be {self.n_dim}"
        )

        batch_size, num_points, _ = coords.shape

        # Expand coords to shape [B, N, 3, 1] for broadcasting
        coords_expanded = coords.unsqueeze(-1)

        # self.div_term is already on the correct device because it's a buffer.
        # Reshape div_term for broadcasting: [half_dim] -> [1, half_dim]
        div_term_expanded = self.div_term.unsqueeze(0)

        # Compute phase for each coordinate:
        # Broadcasting: [B, N, 3, 1] * [1, half_dim] -> [B, N, 3, half_dim]
        phase = coords_expanded * div_term_expanded

        # Calculate sin and cos parts: shape [B, N, 3, half_dim]
        pos_sin = torch.sin(phase)
        pos_cos = torch.cos(phase)

        # Concatenate sin and cos parts along the last dimension for each axis:
        # resulting shape: [B, N, 3, d_per_axis]
        pos_enc_per_axis = torch.cat([pos_sin, pos_cos], dim=-1)

        # Flatten the last two dimensions to combine axis encodings:
        # Shape: [B, N, 3 * d_per_axis] = [B, N, actual_encoded_dim]
        # Replace rearrange with view
        pos_emb = pos_enc_per_axis.view(batch_size, num_points, self.actual_encoded_dim)

        # Pad with zeros if necessary to reach the target feature_dim
        if self.padding_dim > 0:
            # Use F.pad for padding the last dimension on the right
            pos_emb = F.pad(pos_emb, (0, self.padding_dim))

        return pos_emb
