import torch
import torch.nn.init as init
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .primitives import LayerNorm


class AttentionPairBias(nn.Module):
    """Attention layer with pairwise bias for molecular structure modeling.

    This layer implements multi-head attention with additional pairwise bias terms
    that capture structural relationships between atoms or residues.

    The attention mechanism computes:
    Attention(Q, K, V) = softmax((QK^T + Z) / sqrt(d_k) + mask) V

    where Z is the pairwise bias matrix projected from input pairwise features.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence/atom feature dimension.
        c_z : int
            The input pairwise feature dimension.
        num_heads : int
            The number of attention heads. Must divide c_s evenly.
        inf : float, optional
            Large value for masking invalid positions, by default 1e6
        initial_norm : bool, optional
            Whether to apply layer normalization to input features, by default True

        """
        super().__init__()

        assert c_s % num_heads == 0, (
            f"c_s ({c_s}) must be divisible by num_heads ({num_heads})"
        )

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = LayerNorm(c_s)  # Normalize input sequence features

        # Attention projection layers
        self.proj_q = nn.Linear(
            c_s, c_s
        )  # Query projection: [B, S, c_s] -> [B, S, c_s]
        self.proj_k = nn.Linear(
            c_s, c_s, bias=False
        )  # Key projection: [B, S, c_s] -> [B, S, c_s]
        self.proj_v = nn.Linear(
            c_s, c_s, bias=False
        )  # Value projection: [B, S, c_s] -> [B, S, c_s]
        self.proj_g = nn.Linear(
            c_s, c_s, bias=False
        )  # Gating projection: [B, S, c_s] -> [B, S, c_s]

        # Pairwise bias projection: [B, S, S, c_z] -> [B, H, S, S]
        self.proj_z = nn.Sequential(
            LayerNorm(c_z),  # Normalize pairwise features
            nn.Linear(c_z, num_heads, bias=False),  # Project to num_heads channels
            Rearrange("b ... h -> b h ..."),  # Rearrange to [B, H, S, S]
        )

        # Output projection: [B, S, c_s] -> [B, S, c_s]
        self.proj_o = nn.Linear(c_s, c_s, bias=False)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor | None = None,
        multiplicity: int = 1,
    ) -> Tensor:
        """Forward pass through attention layer with pairwise bias.

        Parameters
        ----------
        s : torch.Tensor
            Input sequence/atom features of shape [B, S, c_s]
            where B=batch_size, S=sequence_length, c_s=feature_dim
        z : torch.Tensor
            Input pairwise features of shape [B, S, S, c_z]
            where c_z=pairwise_feature_dim
        mask : torch.Tensor | None, optional
            Attention mask of shape [B, S] indicating valid positions.
            True means valid, False means masked. If None, all positions are valid.
        multiplicity : int, optional
            Diffusion multiplicity factor for batch expansion, by default 1
            When > 1, sequence batch becomes [B*multiplicity, S, c_s] while
            pairwise features remain [B, S, S, c_z]

        Returns
        -------
        torch.Tensor
            Output sequence features of shape [B*multiplicity, S, c_s]

        """
        B, S = s.shape[:2]

        # Create default mask if none provided: [B, S]
        if mask is None:
            mask = torch.ones(B, S, device=s.device, dtype=torch.bool)

        # Create mask for zeroing outputs: [B, S, 1]
        v_mask = mask.float().unsqueeze(-1)

        # Layer normalization: [B, S, c_s] -> [B, S, c_s]
        if self.initial_norm:
            s = self.norm_s(s)

        # Compute attention projections and zero out padded positions
        # [B, S, c_s] -> [B, S, H, D]
        q = self.proj_q(s).view(B, S, self.num_heads, self.head_dim)  # [B, S, H, D]
        k = self.proj_k(s).view(B, S, self.num_heads, self.head_dim)  # [B, S, H, D]
        v = self.proj_v(s).view(B, S, self.num_heads, self.head_dim)  # [B, S, H, D]

        # Zero out Q/K/V for padded positions to prevent information leakage
        qkv_mask = v_mask.unsqueeze(-1)  # [B, S, 1, 1]
        q = q * qkv_mask
        k = k * qkv_mask
        v = v * qkv_mask

        # Expand pairwise features for multiplicity: [B, S, S, c_z] -> [B*m, S, S, c_z]
        z = z.repeat_interleave(multiplicity, 0)
        z = self.proj_z(z)  # [B*m, H, S, S] - pairwise bias for each head

        # Gating values: [B, S, c_s] -> [B, S, c_s]
        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Compute attention scores: [B, S, H, D] x [B, S, H, D] -> [B, H, S, S]
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())

            # Scale and add pairwise bias: [B, H, S, S] + [B*m, H, S, S] -> [B, H, S, S]
            attn = attn / (self.head_dim**0.5) + z.float()

            # Apply attention mask: [B, S] -> [B, 1, 1, S] -> [B, H, S, S]
            # Mask out invalid positions with large negative values
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf

            # Softmax to get attention weights: [B, H, S, S]
            attn = attn.softmax(dim=-1)

            # Apply attention to values: [B, H, S, S] x [B, S, H, D] -> [B, S, H, D]
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)

        # Reshape and apply gating: [B, S, H, D] -> [B, S, c_s]
        o = o.reshape(B, S, self.c_s)
        o = self.proj_o(g * o)  # [B, S, c_s] -> [B, S, c_s]

        # Zero out output for padded positions
        o = o * v_mask

        return o

    def _init_weights(self):
        """Initialize weights."""
        init.zeros_(self.proj_o.weight)  # Initialize with small weights
