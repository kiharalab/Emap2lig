"""Primitive neural network building blocks and utilities.

This module provides commonly used neural network components including
custom layer normalization, adaptive normalization, activation functions,
and multi-layer perceptrons optimized for the emap2ligand architecture.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Convenient aliases for common layer types
LinearNoBias = partial(nn.Linear, bias=False)  # Linear layer without bias
Linear = nn.Linear  # Standard linear layer


def disable_autocast_for(device: torch.device | str) -> Iterator[None]:
    """Disable autocast for the active device when PyTorch supports it."""
    device_type = torch.device(device).type
    is_available = getattr(torch.amp, "is_autocast_available", None)
    if is_available is not None and is_available(device_type):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation with BF16 optimization.

    Provides layer normalization with automatic float32 casting for improved
    numerical stability when using mixed precision training.
    """

    def __init__(self, c_in: int, eps: float = 1e-5):
        """Initialize LayerNorm.

        Parameters
        ----------
        c_in : int
            Number of input channels/features
        eps : float, optional
            Small value for numerical stability, by default 1e-5
        """
        super().__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x: Tensor) -> Tensor:  # [*, C] -> [*, C]
        """Apply layer normalization.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape [*, C] where * is any number of dimensions

        Returns
        -------
        Tensor
            Normalized tensor, same shape as input [*, C]
        """
        with disable_autocast_for(x.device):
            dtype = x.dtype
            out = nn.functional.layer_norm(
                x.float() if x.is_floating_point() else x,
                self.c_in,
                self.weight.float() if self.weight.is_floating_point() else self.weight,
                self.bias.float() if self.bias.is_floating_point() else self.bias,
                self.eps,
            )
        if x.is_floating_point():
            out = out.to(dtype)
        return out


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation function.

    Applies SwiGLU activation: SiLU(x1) * x2 where x1, x2 are chunks of input.
    Expects input to have even last dimension for splitting.
    """

    def forward(
        self,
        x: Tensor,  # [*, 2*D]
    ) -> Tensor:  # [*, D]
        """Apply SwiGLU activation.

        Parameters
        ----------
        x : Tensor
            Input tensor with even last dimension, shape [*, 2*D]

        Returns
        -------
        Tensor
            Activated tensor, shape [*, D]
        """
        with disable_autocast_for(x.device):
            dtype = x.dtype
            x, gates = x.float().chunk(2, dim=-1)  # [*, D] each
            return (F.silu(gates) * x).to(dtype)  # [*, D]


class MLP(nn.Module):
    """Multi-layer perceptron with SwiGLU activation.

    Implements a two-layer MLP with layer normalization, SwiGLU activation,
    and configurable input/output dimensions.
    """

    def __init__(self, dim: int, hidden: int, out_dim: int) -> None:
        """Initialize MLP.

        Parameters
        ----------
        dim : int
            Input dimension
        hidden : int
            Hidden dimension
        out_dim : int
            Output dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            Linear(dim, hidden),
            SwiGLU(),
            Linear(hidden // 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # [*, dim] -> [*, out_dim]
        """Forward pass through MLP.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape [*, dim]

        Returns
        -------
        Tensor
            Output tensor, shape [*, out_dim]
        """
        return self.net(x)  # [*, out_dim]
