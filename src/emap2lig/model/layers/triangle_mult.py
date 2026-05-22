import math

import torch
import torch.nn.init as init
from torch import Tensor, nn

from .primitives import LayerNorm, LinearNoBias

try:
    from cuequivariance_torch.primitives.triangle import (
        triangle_multiplicative_update,  # type: ignore
    )

    HAS_CUEQUIVARIANCE = True
except ImportError:
    HAS_CUEQUIVARIANCE = False

from loguru import logger


@torch.compiler.disable
def kernel_triangular_mult(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
):
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class TriangleMultiplicationOutgoing(nn.Module):
    """Triangle multiplication update in the outgoing direction.

    Implements the triangle multiplicative update from AlphaFold2, which updates
    pair representations by considering triangular relationships. The 'outgoing'
    direction updates z_ij using the pattern: z_ij <- sum_k(a_ik * b_jk).
    """

    def __init__(self, dim: int = 128, use_cuequiv: bool = False) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        use_cuequiv: bool
            Whether to use cuequivariance kernels, default False

        """
        super().__init__()
        self.use_cuequiv = use_cuequiv

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
        # Use optimized cuequivariance kernel if available
        if self.use_cuequiv:
            if HAS_CUEQUIVARIANCE:
                return kernel_triangular_mult(
                    x,
                    direction="outgoing",
                    mask=mask,
                    norm_in_weight=self.norm_in.weight,
                    norm_in_bias=self.norm_in.bias,
                    p_in_weight=self.p_in.weight,
                    g_in_weight=self.g_in.weight,
                    norm_out_weight=self.norm_out.weight,
                    norm_out_bias=self.norm_out.bias,
                    p_out_weight=self.p_out.weight,
                    g_out_weight=self.g_out.weight,
                    eps=1e-5,
                )
            else:
                logger.warning(
                    "Cuequivariance not installed, using fallback implementation"
                )

        # Fallback PyTorch implementation
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

    def __init__(self, dim: int = 128, use_cuequiv: bool = False) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        use_cuequiv: bool
            Whether to use cuequivariance kernels, default False

        """
        super().__init__()
        self.use_cuequiv = use_cuequiv

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
        # Use optimized cuequivariance kernel if available
        if self.use_cuequiv:
            if HAS_CUEQUIVARIANCE:
                return kernel_triangular_mult(
                    x,
                    direction="incoming",
                    mask=mask,
                    norm_in_weight=self.norm_in.weight,
                    norm_in_bias=self.norm_in.bias,
                    p_in_weight=self.p_in.weight,
                    g_in_weight=self.g_in.weight,
                    norm_out_weight=self.norm_out.weight,
                    norm_out_bias=self.norm_out.bias,
                    p_out_weight=self.p_out.weight,
                    g_out_weight=self.g_out.weight,
                    eps=1e-5,
                )
            else:
                logger.warning(
                    "Cuequivariance not installed, using fallback implementation"
                )

        # Fallback PyTorch implementation
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
