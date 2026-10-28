import math

import torch
import torch.nn.init as init
from torch import Tensor, nn

from .primitives import LayerNorm, LinearNoBias


class TriangleMultiplicationOutgoing(nn.Module):
    """Triangle multiplication update in the outgoing direction.

    Implements the triangle multiplicative update from AlphaFold2, which updates
    pair representations by considering triangular relationships. The 'outgoing'
    direction updates z_ij using the pattern: z_ij <- sum_k(a_ik * b_jk).
    """

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = LayerNorm(dim, eps=1e-5)
        self.p_in = LinearNoBias(dim, 2 * dim)
        self.g_in = LinearNoBias(dim, 2 * dim)

        self.norm_out = LayerNorm(dim)
        self.p_out = LinearNoBias(dim, dim)
        self.g_out = LinearNoBias(dim, dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.ones_(self.norm_in.weight)
        init.zeros_(self.norm_in.bias)

        # lecun_normal_init_ equivalent: trunc_normal with scale=1.0
        init.trunc_normal_(
            self.p_in.weight, std=math.sqrt(1.0 / max(1, self.p_in.weight.shape[1]))
        )
        init.zeros_(self.g_in.weight)

        init.ones_(self.norm_out.weight)
        init.zeros_(self.norm_out.bias)

        init.zeros_(self.p_out.weight)
        init.zeros_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        # Input gating and normalization
        x = self.norm_in(x)  # [B, N, N, D]
        x_in = x  # Store for skip connection
        x = self.p_in(x) * self.g_in(x).sigmoid()  # [B, N, N, 2D] gated

        # Apply mask to prevent invalid updates
        v_mask = mask.unsqueeze(-1)  # [B, N, N, 1]
        x = x * v_mask  # [B, N, N, 2D]

        # Split into two factors for multiplication
        a, b = torch.chunk(x.float(), 2, dim=-1)  # [B, N, N, D] each

        # Triangle multiplication: outgoing direction
        # For each (i,j), sum over k: a[i,k] * b[j,k]
        x = torch.einsum("bikd,bjkd->bijd", a, b)  # [B, N, N, D]

        # Output gating with residual connection
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()  # [B, N, N, D]

        # Zero out output for padded positions
        x = x * v_mask

        return x  # [B, N, N, D]


class TriangleMultiplicationIncoming(nn.Module):
    """Triangle multiplication update in the incoming direction.

    Implements the triangle multiplicative update from AlphaFold2, which updates
    pair representations by considering triangular relationships. The 'incoming'
    direction updates z_ij using the pattern: z_ij <- sum_k(a_ki * b_kj).
    """

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = LayerNorm(dim, eps=1e-5)
        self.p_in = LinearNoBias(dim, 2 * dim)
        self.g_in = LinearNoBias(dim, 2 * dim)

        self.norm_out = LayerNorm(dim)
        self.p_out = LinearNoBias(dim, dim)
        self.g_out = LinearNoBias(dim, dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.ones_(self.norm_in.weight)
        init.zeros_(self.norm_in.bias)

        # lecun_normal_init_ equivalent: trunc_normal with scale=1.0
        init.trunc_normal_(
            self.p_in.weight, std=math.sqrt(1.0 / max(1, self.p_in.weight.shape[1]))
        )
        init.zeros_(self.g_in.weight)

        init.ones_(self.norm_out.weight)
        init.zeros_(self.norm_out.bias)

        init.zeros_(self.p_out.weight)
        init.zeros_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        # Input gating and normalization
        x = self.norm_in(x)  # [B, N, N, D]
        x_in = x  # Store for skip connection
        x = self.p_in(x) * self.g_in(x).sigmoid()  # [B, N, N, 2D] gated

        # Apply mask to prevent invalid updates
        v_mask = mask.unsqueeze(-1)  # [B, N, N, 1]
        x = x * v_mask  # [B, N, N, 2D]

        # Split into two factors for multiplication
        a, b = torch.chunk(x.float(), 2, dim=-1)  # [B, N, N, D] each

        # Triangle multiplication: incoming direction
        # For each (i,j), sum over k: a[k,i] * b[k,j]
        x = torch.einsum("bkid,bkjd->bijd", a, b)  # [B, N, N, D]

        # Output gating with residual connection
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()  # [B, N, N, D]

        # Zero out output for padded positions
        x = x * v_mask

        return x  # [B, N, N, D]
