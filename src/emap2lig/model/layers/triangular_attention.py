# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2025 Shu Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
from torch import Tensor

from .primitives import LayerNorm, LinearNoBias, disable_autocast_for


class TriangleAttention(nn.Module):
    """
    Triangle attention mechanism for pair representations.

    Implements Algorithm 13 (starting node) and Algorithm 14 (ending node)
    from AlphaFold2. Applies attention along one axis of the pair representation
    while using information from the perpendicular axis as bias.

    The attention mechanism operates on triangular patterns:
    - Starting node: attention flows from i->k with j fixed
    - Ending node: attention flows from j->k with i fixed

    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ) -> None:
        """
        Initialize triangle attention module.

        Args:
            c_in: Input channel dimension for pair representation
            c_hidden: Hidden dimension per attention head
            no_heads: Number of attention heads
            starting: If True, implements starting node attention (Algorithm 13),
                     otherwise ending node attention (Algorithm 14)
            inf: Large value for masking invalid positions
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        # Layer normalization for input
        self.layer_norm = LayerNorm(c_in)

        # Linear projection for triangle bias
        # Projects pair features to per-head bias: [*, I, J, c_in] -> [*, no_heads, I, J]
        self.linear_bias = LinearNoBias(c_in, no_heads)

        # Attention projection layers
        self.linear_q = LinearNoBias(c_in, no_heads * c_hidden)
        self.linear_k = LinearNoBias(c_in, no_heads * c_hidden)
        self.linear_v = LinearNoBias(c_in, no_heads * c_hidden)
        self.linear_g = LinearNoBias(c_in, no_heads * c_hidden)
        self.linear_o = LinearNoBias(no_heads * c_hidden, c_in)

        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.kaiming_normal_(self.linear_bias.weight, nonlinearity="linear")
        init.xavier_uniform_(self.linear_q.weight, gain=1)
        init.xavier_uniform_(self.linear_k.weight, gain=1)
        init.xavier_uniform_(self.linear_v.weight, gain=1)
        init.zeros_(self.linear_g.weight)
        init.zeros_(self.linear_o.weight)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for triangle attention.

        Args:
            x: Input pair representation [*, I, J, c_in]
                For starting=True: attention along j-axis with i fixed
                For starting=False: attention along i-axis with j fixed
            mask: Mask tensor [*, I, J] indicating valid positions (True=valid, False=masked)

        Returns:
            Updated pair representation [*, I, J, c_in]
        """
        return self._forward_normal(x, mask)

    def _forward_normal(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Standard attention implementation using einsum operations.

        Args:
            x: Input pair representation [*, I, J, c_in]
            mask: Mask tensor [*, I, J] indicating valid positions.

        Returns:
            Output pair representation [*, I, J, c_in]
        """
        # Get input dimensions
        *_batch_dims, n_i, n_j, _ = x.shape

        # For ending node attention, swap spatial dimensions using einops
        if not self.starting:
            # Swap I and J dimensions: [*, I, J, c_in] -> [*, J, I, c_in]
            x = rearrange(x, "... i j c -> ... j i c")
            mask = rearrange(mask, "... i j -> ... j i")
            n_i, n_j = n_j, n_i

        # Apply layer normalization
        # [*, I, J, c_in] -> [*, I, J, c_in]
        x = self.layer_norm(x)

        # Compute triangle bias from pair features.
        # Boltz layout: [*, 1, no_heads, I, J].
        tri_bias = self.linear_bias(x)
        tri_bias = rearrange(tri_bias, "... i j h -> ... 1 h i j")

        # Compute Q, K, V projections
        # [*, I, J, c_in] -> [*, I, J, no_heads * c_hidden]
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Reshape for multi-head attention using einops
        # [*, I, J, no_heads * c_hidden] -> [*, I, no_heads, J, c_hidden]
        q = rearrange(q, "... i j (h d) -> ... i h j d", h=self.no_heads)
        k = rearrange(k, "... i j (h d) -> ... i h j d", h=self.no_heads)
        v = rearrange(v, "... i j (h d) -> ... i h j d", h=self.no_heads)

        # Mask attention keys, not the Q/K/V projections (Boltz semantics).

        with disable_autocast_for(x.device):
            # Compute attention scores in fp32 for numerical stability.
            # [*, I, no_heads, J, J]
            k_transposed = rearrange(k, "... i h j d -> ... i h d j")
            attn_scores = torch.matmul(q.float(), k_transposed.float())
            attn_scores = attn_scores / (self.c_hidden**0.5)

            # Add triangle bias and key mask.
            # attn_scores: [*, I, no_heads, J, J]
            # tri_bias:    [*, 1, no_heads, I, J]
            attn_scores = attn_scores + tri_bias.float()
            mask_bias = self.inf * (mask[..., :, None, None, :].float() - 1)
            attn_scores = attn_scores + mask_bias.float()

            # Apply softmax
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Apply attention to values: [*, I, no_heads, J, c_hidden]
            o = torch.matmul(attn_weights, v.float()).to(v.dtype)

        # Reshape back to pair format using einops
        # [*, I, no_heads, J, c_hidden] -> [*, I, J, no_heads * c_hidden]
        o = rearrange(o, "... i h j d -> ... i j (h d)")

        # Apply gating
        g = self.sigmoid(self.linear_g(x))
        o = o * g

        # Final projection
        o = self.linear_o(o)

        # Keep Boltz semantics: masked pair outputs are not zeroed here.

        # For ending node, swap back to original orientation using einops
        if not self.starting:
            # [*, J, I, c_in] -> [*, I, J, c_in]
            o = rearrange(o, "... j i c -> ... i j c")

        return o


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Triangle attention starting node (Algorithm 13).

    Applies attention along the j-axis (columns) with i fixed (rows).
    Each row attends to all positions in that row.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        """
        Initialize starting node triangle attention.

        Args:
            c_in: Input channel dimension
            c_hidden: Hidden dimension per head
            no_heads: Number of attention heads
            inf: Large value for masking
        """
        super().__init__(
            c_in,
            c_hidden,
            no_heads,
            starting=True,
            inf=inf,
        )


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Triangle attention ending node (Algorithm 14).

    Applies attention along the i-axis (rows) with j fixed (columns).
    Each column attends to all positions in that column.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        """
        Initialize ending node triangle attention.

        Args:
            c_in: Input channel dimension
            c_hidden: Hidden dimension per head
            no_heads: Number of attention heads
            inf: Large value for masking
        """
        super().__init__(
            c_in,
            c_hidden,
            no_heads,
            starting=False,
            inf=inf,
        )
