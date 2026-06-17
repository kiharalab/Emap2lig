import torch
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .positional_encoding import PositionalEncoder
from .primitives import LayerNorm, LinearNoBias, disable_autocast_for


class SelectedCrossAttention(nn.Module):
    """Cross attention between atoms and pre-selected point features.

    Performs cross-attention between atom queries and pre-selected point features.
    Point selection (top-k based on instance segmentation) is done once in InstanceSegModule
    and reused across all conditioning blocks.
    """

    def __init__(
        self,
        atom_dim: int,
        point_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
    ):
        """Initialize SelectedCrossAttention.

        Parameters
        ----------
        atom_dim : int
            Dimension of atom features
        point_dim : int
            Dimension of point features
        num_heads : int, optional
            Number of attention heads, by default 8
        head_dim : int, optional
            Dimension per attention head, by default 32
        """
        super().__init__()

        self.n_heads = num_heads
        self.head_dim = head_dim
        hidden_dim = head_dim * num_heads

        # Layer norms and projections
        self.atom_norm = LayerNorm(atom_dim)
        self.point_norm = LayerNorm(point_dim)

        # Point->Atom attention projections
        self.atom_q_proj = LinearNoBias(atom_dim, hidden_dim)
        self.point_kv_proj = LinearNoBias(point_dim, hidden_dim * 2)

        self.q_pos_enc = PositionalEncoder(hidden_dim)
        self.k_pos_enc = PositionalEncoder(hidden_dim)
        self.v_pos_enc = PositionalEncoder(hidden_dim)

        # Output projection
        self.proj_o = LinearNoBias(hidden_dim, atom_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.xavier_uniform_(self.point_kv_proj.weight, gain=1)
        init.xavier_uniform_(self.atom_q_proj.weight, gain=1)
        init.zeros_(self.proj_o.weight)

    def forward(
        self,
        atom_feats: torch.Tensor,  # [B, N_a, C_a]
        atom_coords: torch.Tensor,  # [B, N_a, 3]
        selected_point_feats: torch.Tensor,  # [B, num_points, C_v]
        selected_point_coords: torch.Tensor,  # [B, num_points, 3]
        atom_mask: torch.Tensor | None = None,  # [B, N_a]
    ) -> torch.Tensor:
        """Forward: run cross-attention between atoms and pre-selected points.

        Parameters
        ----------
        atom_feats : torch.Tensor
            Atom features of shape [B, N_a, C_a]
        atom_coords : torch.Tensor
            Atom coordinates of shape [B, N_a, 3]
        selected_point_feats : torch.Tensor
            Pre-selected point features of shape [B, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates of shape [B, num_points, 3]
        atom_mask : torch.Tensor | None
            Atom mask of shape [B, N_a], True = valid atom. If None, all atoms are valid.

        Returns
        -------
        torch.Tensor
            Updated atom features of shape [B, N_a, C_a]
        """
        B, N_a, _C_a = atom_feats.shape

        # Create default mask if none provided
        if atom_mask is None:
            atom_mask = torch.ones(B, N_a, device=atom_feats.device, dtype=torch.bool)

        # Create mask for zeroing outputs: [B, N_a, 1]
        v_mask = atom_mask.float().unsqueeze(-1)

        # Normalize atom queries and build Q with positional encoding
        atom_feats_norm = self.atom_norm(atom_feats)  # [B, N_a, C_a]
        q = self.atom_q_proj(atom_feats_norm)  # [B, N_a, H*D]
        with disable_autocast_for(q.device):
            q = self.q_pos_enc(atom_coords, q)

        # Zero out Q for padded positions
        q = q * v_mask

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads).contiguous()

        # Normalize point features
        selected_feats_norm = self.point_norm(selected_point_feats)

        # Project K/V and add positional encodings
        kv = self.point_kv_proj(selected_feats_norm)
        k, v = kv.chunk(2, dim=-1)

        with disable_autocast_for(k.device):
            k = self.k_pos_enc(selected_point_coords, k)
            v = self.v_pos_enc(selected_point_coords, v)

        # Reshape for attention
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads).contiguous()
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads).contiguous()

        if q.device.type == "cuda":
            with (
                sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)

        # Merge heads and project
        o = rearrange(o, "b h n d -> b n (h d)")
        o = self.proj_o(o.float())

        # Zero out output for padded positions
        o = o * v_mask

        return atom_feats + o
