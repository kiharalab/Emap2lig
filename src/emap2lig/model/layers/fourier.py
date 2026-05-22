from math import pi

import torch
from einops import rearrange
from torch import nn
from torch.nn import Module

from .primitives import Linear


class FourierEmbedding(Module):
    """Fourier embedding layer for continuous time values.

    Applies random Fourier features to embed continuous time values into
    a higher-dimensional space. Uses fixed random projections followed by
    cosine activation to create smooth, periodic embeddings.
    """

    def __init__(self, dim: int):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the output embeddings
        """
        super().__init__()
        # Fixed random projection layer (frozen weights)
        self.proj = Linear(1, dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.proj.weight, mean=0, std=1)
        nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)  # Freeze the random projection

    def forward(
        self,
        times: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, dim]
        """Apply Fourier embedding to time values.

        Parameters
        ----------
        times : torch.Tensor
            Time values, shape [B]

        Returns
        -------
        torch.Tensor
            Fourier embeddings, shape [B, dim]
        """
        times = rearrange(times, "b -> b 1")  # [B, 1]
        rand_proj = self.proj(times)  # [B, dim]
        return torch.cos(2 * pi * rand_proj)  # [B, dim]
