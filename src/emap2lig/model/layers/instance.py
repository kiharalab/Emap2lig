import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..seg.munet import DownSample3d, UpSample3d
from .primitives import LayerNorm, LinearNoBias


class CrossAttention(nn.Module):
    """
    Cross-attention module using scaled dot-product attention with optional key padding mask.

    Queries come from one feature set (e.g., downsampled volume features) and
    keys/values come from another feature set (e.g., atom features). Uses
    torch.nn.functional.scaled_dot_product_attention for efficient attention
    with automatic backend selection (FlashAttention, Memory-Efficient, or Math).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        """
        Initialize CrossAttention.

        Args:
            query_dim (int): Dimension of query features.
            context_dim (int): Dimension of context features (keys/values).
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension per attention head.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads

        # Layer norms
        self.query_norm = LayerNorm(query_dim)
        self.context_norm = LayerNorm(context_dim)

        # Projection layers
        self.query_proj = LinearNoBias(query_dim, self.hidden_dim)
        self.key_proj = LinearNoBias(context_dim, self.hidden_dim)
        self.value_proj = LinearNoBias(context_dim, self.hidden_dim)

        # Output projection
        self.output_proj = LinearNoBias(self.hidden_dim, query_dim)

    def forward(
        self,
        query: torch.Tensor,  # [B, N_q, query_dim]
        context: torch.Tensor,  # [B, N_c, context_dim]
        attn_mask: torch.Tensor | None = None,  # [B, 1, 1, N_c] boolean mask
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention.

        Args:
            query (torch.Tensor): Query features of shape [B, N_q, query_dim].
            context (torch.Tensor): Context features of shape [B, N_c, context_dim].
            attn_mask (torch.Tensor | None): Boolean attention mask broadcastable to
                [B, H, N_q, N_c]. True = valid position to attend, False = masked.

        Returns:
            torch.Tensor: Attended query features of shape [B, N_q, query_dim].
        """
        # Store original query for residual connection
        residual = query

        # Apply layer norms
        query = self.query_norm(query)  # [B, N_q, query_dim]
        context = self.context_norm(context)  # [B, N_c, context_dim]

        # Project to Q, K, V
        q = self.query_proj(query)  # [B, N_q, hidden_dim]
        k = self.key_proj(context)  # [B, N_c, hidden_dim]
        v = self.value_proj(context)  # [B, N_c, hidden_dim]

        # Reshape for multi-head attention
        # [B, N, hidden_dim] -> [B, num_heads, N, head_dim]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Apply scaled dot-product attention
        # SDPA automatically selects optimal backend (FlashAttention, Memory-Efficient, etc.)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )  # [B, H, N_q, D]

        # Reshape back to [B, N_q, hidden_dim]
        attended = rearrange(attended, "b h n d -> b n (h d)")

        # Final output projection
        output = self.output_proj(attended)  # [B, N_q, query_dim]

        # Residual connection
        output = output + residual  # [B, N_q, query_dim]

        return output


class Instance3dBlock(nn.Module):
    """
    A single instance segmentation block that processes 3D volume features.

    This module performs the core instance segmentation logic by combining:
    - 3D convolutional layers for spatial feature processing
    - Cross-attention between volume features and molecule features
    - Relative position encoding for spatial awareness
    """

    def __init__(
        self,
        volume_channels: int,
        atom_dim: int,
        num_attention_heads: int = 8,
    ):
        """
        Initialize Instance3dBlock.

        Args:
            volume_channels (int): Number of channels in the 3D volume features.
            atom_dim (int): Dimension of atom features for cross-attention.
            num_attention_heads (int): Number of attention heads for cross-attention.
        """
        super().__init__()

        # 3D convolutional layers for spatial processing
        self.conv3d_1 = nn.Conv3d(
            volume_channels, volume_channels, kernel_size=3, padding=1
        )
        self.conv3d_2 = nn.Conv3d(
            volume_channels, volume_channels, kernel_size=3, padding=1
        )
        self.norm1 = nn.GroupNorm(8, volume_channels)
        self.norm2 = nn.GroupNorm(8, volume_channels)

        # Cross-attention between volume and atom features
        self.cross_attention = CrossAttention(
            query_dim=volume_channels,
            context_dim=atom_dim,
            num_heads=num_attention_heads,
        )

        # Activation function
        self.activation = nn.SiLU(inplace=True)

        # Downsample/Upsample modules (factor 4x: 2x down twice, 4x up once)
        self.down = nn.Sequential(
            DownSample3d(volume_channels),
            DownSample3d(volume_channels),
        )
        self.up = UpSample3d(volume_channels, volume_channels, scale_factor=4)

    def forward(
        self,
        volume_features: torch.Tensor,
        atom_features: torch.Tensor,
        rel_pos_encoded_flat: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass through Instance3dBlock.

        Args:
            volume_features (torch.Tensor): 3D volume features of shape [B, C, D, H, W].
            atom_features (torch.Tensor): Atom features of shape [B, N_a, atom_dim].
            rel_pos_encoded_flat (torch.Tensor): Relative positions of shape [B, N_q, C].
            attn_mask (torch.Tensor | None): Boolean attention mask of shape [B, 1, 1, N_a].

        Returns:
            torch.Tensor: Processed volume features of shape [B, C, D, H, W].
        """
        # Residual input
        residual = volume_features

        # Convolutional body (no residual add here)
        x = self.conv3d_1(volume_features)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv3d_2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Downsample volume to reduce attention complexity (4x via two 2x stages)
        x_ds = self.down(x)  # [B, C, D', H', W']
        _, _, Dd, Hd, Wd = x_ds.shape

        # Convert downsampled volume to point format [B, N_q, C]
        point_features = rearrange(x_ds, "b c d h w -> b (d h w) c")

        # Add precomputed relative positional encoding (already flattened)
        rel_pos_encoded = rel_pos_encoded_flat  # [B, N_q, C]

        # Add relative position encoding to point features
        point_features = point_features + rel_pos_encoded

        # Apply cross-attention with boolean mask
        attended_features = self.cross_attention(
            query=point_features,  # [B, N_q, C]
            context=atom_features,  # [B, N_a, C_atom]
            attn_mask=attn_mask,  # [B, 1, 1, N_a]
        )  # [B, N_q, C]

        # Convert back to downsampled volume and upsample to original resolution
        output_ds = rearrange(
            attended_features, "b (d h w) c -> b c d h w", d=Dd, h=Hd, w=Wd
        )
        attn_up = self.up(output_ds)  # [B, C, D, H, W]

        return residual + x + attn_up


class InstanceSeg(nn.Module):
    """
    Instance segmentation module consisting of several Instance3dBlock modules.

    This module stacks multiple Instance3dBlock modules to perform hierarchical
    instance-aware feature processing with cross-attention to molecule context.
    """

    def __init__(
        self,
        channels: int,
        atom_dim: int,
        num_blocks: int = 4,
        num_attention_heads: int = 8,
    ):
        """
        Initialize InstanceSeg module.

        Args:
            channels (int): Number of channels in the 3D volume features.
            atom_dim (int): Dimension of atom features for cross-attention.
            num_blocks (int): Number of Instance3dBlock modules to stack.
            num_attention_heads (int): Number of attention heads for cross-attention.
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.rel_pos_proj = LinearNoBias(3, channels)

        # Stack of Instance3dBlock modules
        self.blocks = nn.ModuleList(
            [
                Instance3dBlock(
                    volume_channels=channels,
                    atom_dim=atom_dim,
                    num_attention_heads=num_attention_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final projection head
        self.output_proj = nn.Conv3d(channels, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(
        self,
        volume_features: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        rel_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through InstanceSeg module.

        Args:
            volume_features (torch.Tensor): 3D volume features of shape [B, C, D, H, W].
            atom_features (torch.Tensor): Atom features of shape [B, N_a, atom_dim].
            atom_mask (torch.Tensor): Atom mask of shape [B, N_a], True = valid atom.
            rel_positions (torch.Tensor): Relative positions of shape [B, 3, D, H, W].

        Returns:
            torch.Tensor: Instance-aware volume features of shape [B, C, D, H, W].
        """
        x = volume_features

        # Precompute relative positional encoding at downsampled resolution (4x)
        scale = 4
        rel_ds = F.avg_pool3d(rel_positions, kernel_size=scale, stride=scale)
        rel_flat = rearrange(rel_ds, "b c d h w -> b (d h w) c")  # [B, N_q, 3]

        # Projection from 3 -> channels, then reuse across blocks
        rel_pos_encoded_flat = self.rel_pos_proj(rel_flat)  # [B, N_q, C]

        # Create boolean attention mask for SDPA
        # Shape [B, N_a] -> [B, 1, 1, N_a] for broadcasting to [B, H, N_q, N_a]
        # True = valid position to attend (per SDPA docs)
        attn_mask = atom_mask.bool().unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N_a]

        # Process through each Instance3dBlock
        for block in self.blocks:
            x = block(
                x,
                atom_features,
                rel_pos_encoded_flat,
                attn_mask,
            )

        # Final output projection
        x = self.output_proj(x)

        return self.act(x)
