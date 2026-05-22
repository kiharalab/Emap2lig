import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from loguru import logger


class SelfAttention(nn.Module):
    """
    ### Self Attention Layer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        """
        super().__init__()

        self.n_heads = num_heads
        self.d_head = dim_head
        d_attn = dim_head * num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, d_model)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, depth * height * width, d_model]`
        """

        # Get query, key and value vectors
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_len, 3*d_attn)
        q, k, v = qkv.chunk(3, dim=-1)  # Shape: (batch_size, seq_len, d_attn)

        q = rearrange(
            q, "b s (h d) -> b h s d", h=self.n_heads
        )  # Shape: (batch_size, n_heads, seq_len, d_head)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        output = nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # Shape: (batch_size, n_heads, seq_len, d_head)
        output = rearrange(
            output, "b h s d -> b s (h d)"
        )  # Shape: (batch_size, seq_len, d_attn)

        return self.o_proj(output)  # Shape: (batch_size, seq_len, d_model)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class BasicTransformerBlock(nn.Module):
    """Basic Transformer Layer"""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = SelfAttention(d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, depth * height * width, d_model]`
        """
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Feed-forward network
        x = self.ff(self.norm2(x)) + x

        return x


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (100 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)  # type: torch.Tensor | None

    @torch.no_grad()
    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        batch_size, x, y, z, orig_ch = tensor.shape
        spatial_shape = (x, y, z, orig_ch)

        if self.cached_penc is not None and self.cached_penc.shape[1:] == spatial_shape:
            return self.cached_penc.repeat(batch_size, 1, 1, 1, 1)

        if self.cached_penc is not None and self.cached_penc.shape[1:] != spatial_shape:
            logger.warning(
                "Positional encoding cache was reset because the spatial dimensions changed: "
                f"from {self.cached_penc.shape[1:]} to {spatial_shape}. Please make sure to apply this module to tensors "
                "with consistent spatial dimensions at every forward call."
            )

        self.cached_penc = None
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch]
        return self.cached_penc.repeat(batch_size, 1, 1, 1, 1)


class SpatialTransformerBlock3d(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(
        self, channels: int, n_heads: int, n_layers: int, num_groups: int = 32
    ):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        """
        super().__init__()
        # Initial group normalization
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
        )
        # Initial $1 \times 1$ convolution
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.positional_encoding = PositionalEncoding3D(channels)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(channels, n_heads, channels // n_heads)
                for _ in range(n_layers)
            ]
        )

        # Final $1 \times 1$ convolution
        self.proj_out = nn.Conv3d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, depth, height, width]`
        """
        # Get shape `[batch_size, channels, height, width, depth]`
        _, _, h, w, d = x.shape
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x.to(torch.float32)).to(x.dtype)
        # Initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width, depth]`
        # to `[batch_size, height * width * depth, channels]`
        x = x.permute(0, 2, 3, 4, 1)

        pos_emb = self.positional_encoding(x)
        x += pos_emb

        x = rearrange(x, "b h w d c -> b (h w d) c").contiguous()

        # Apply the transformer layers
        for block in self.transformer_blocks:
            x = block(x)
        # Reshape and transpose from `[batch_size, height * width * depth, channels]`
        # to `[batch_size, channels, height, width, depth]`
        x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d).contiguous()
        # Final $1 \times 1$ convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.0),
            nn.Linear(d_model * d_mult, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * self.gelu(gate)
