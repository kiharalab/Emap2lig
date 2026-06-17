import math

import pytest
import torch
from einops import rearrange

from emap2lig.model.layers.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)


def _reference_non_cueq_attention(
    layer, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not layer.starting:
        x = rearrange(x, "... i j c -> ... j i c")
        mask = rearrange(mask, "... i j -> ... j i")

    x = layer.layer_norm(x)
    q = rearrange(layer.linear_q(x), "... i j (h d) -> ... i h j d", h=layer.no_heads)
    k = rearrange(layer.linear_k(x), "... i j (h d) -> ... i h j d", h=layer.no_heads)
    v = rearrange(layer.linear_v(x), "... i j (h d) -> ... i h j d", h=layer.no_heads)

    scores = torch.einsum("...ihqd,...ihkd->...ihqk", q.float(), k.float())
    scores = scores / math.sqrt(layer.c_hidden)

    # AF2/Boltz triangle attention uses pair bias b[q, k], broadcast over row i.
    pair_bias = rearrange(layer.linear_bias(x), "... q k h -> ... 1 h q k")
    key_mask_bias = layer.inf * (mask[..., :, None, None, :].float() - 1)
    weights = torch.softmax(scores + pair_bias.float() + key_mask_bias, dim=-1)

    out = torch.matmul(weights, v.float()).to(v.dtype)
    out = rearrange(out, "... i h j d -> ... i j (h d)")
    out = out * torch.sigmoid(layer.linear_g(x))
    out = layer.linear_o(out)

    if not layer.starting:
        out = rearrange(out, "... j i c -> ... i j c")
    return out


@pytest.mark.parametrize(
    "layer_type",
    [TriangleAttentionStartingNode, TriangleAttentionEndingNode],
)
def test_non_cueq_triangle_attention_matches_pair_bias_reference(layer_type) -> None:
    torch.manual_seed(1234)
    layer = layer_type(c_in=4, c_hidden=3, no_heads=2, inf=1e5, use_cuequiv=False)

    # Constructor zeroes gate/output projections. Make the residual path observable.
    with torch.no_grad():
        layer.linear_g.weight.fill_(0.25)
        layer.linear_o.weight.copy_(
            torch.linspace(-0.3, 0.3, layer.linear_o.weight.numel()).view_as(
                layer.linear_o.weight
            )
        )

    x = torch.linspace(-1.5, 1.5, steps=2 * 4 * 4 * 4).view(2, 4, 4, 4)
    atom_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
    )
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]

    with torch.inference_mode():
        actual = layer(x, pair_mask)
        expected = _reference_non_cueq_attention(layer, x, pair_mask)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
