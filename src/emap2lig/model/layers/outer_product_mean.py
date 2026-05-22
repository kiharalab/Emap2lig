import torch
import torch.nn.init as init
from einops import rearrange
from torch import Tensor, nn

from .primitives import LayerNorm


class OuterProductMean(nn.Module):
    """Outer product mean layer for computing pairwise interactions.

    This layer computes the outer product of two projections of the input tensor,
    then applies a mean operation across the sequence dimension to produce a
    pairwise interaction tensor.
    """

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        """Initialize the outer product mean layer.

        Parameters
        ----------
        c_in : int
            The input dimension.
        c_hidden : int
            The hidden dimension.
        c_out : int
            The output dimension.

        """
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.zeros_(self.proj_o.weight)
        init.zeros_(self.proj_o.bias)

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """Forward pass computing outer product mean.

        Parameters
        ----------
        m : torch.Tensor
            The sequence tensor with shape (B, S, N, c_in) where:
            - B: batch size
            - S: sequence length
            - N: number of residues/positions
            - c_in: input feature dimension
        mask : torch.Tensor
            The mask tensor with shape (B, S, N) indicating valid positions.

        Returns
        -------
        torch.Tensor
            The output tensor with shape (B, N, N, c_out) representing
            pairwise interactions between all residue pairs.

        """
        # Expand mask to match feature dimension: (B, S, N) -> (B, S, N, 1)
        mask = mask.unsqueeze(-1).to(m)

        # Apply layer normalization and compute projections
        m = self.norm(m)  # (B, S, N, c_in)
        a = self.proj_a(m) * mask  # (B, S, N, c_hidden)
        b = self.proj_b(m) * mask  # (B, S, N, c_hidden)

        # Create pairwise mask: (B, S, N, 1) * (B, S, 1, N) -> (B, S, N, N)
        mask = mask[:, :, None, :] * mask[:, :, :, None]
        # Sum across sequence dimension and clamp to avoid division by zero
        num_mask = mask.sum(1).clamp(min=1)  # (B, N, N)

        # Compute outer product: (B, S, N, c_hidden) ⊗ (B, S, N, c_hidden) -> (B, N, N, c_hidden, c_hidden)
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())

        # Flatten last two dimensions: (B, N, N, c_hidden*c_hidden)
        z = rearrange(z, "b i j c d -> b i j (c d)")

        # Apply mean operation across sequence dimension
        z = z / num_mask  # (B, N, N, c_hidden*c_hidden)

        # Project to final output dimension: (B, N, N, c_out)
        z = self.proj_o(z.to(m))
        return z
