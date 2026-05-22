import torch
from torch import Tensor


def get_dropout_mask(
    dropout: float,
    z: Tensor,  # [B, N, N, D]
    training: bool,
    columnwise: bool = False,
) -> Tensor:  # [B, 1, N, 1] or [B, N, 1, 1]
    """Generate dropout mask for pair representations.

    Creates a dropout mask that can be applied either columnwise or rowwise
    to pair representations. The mask is scaled to maintain expected values.

    Parameters
    ----------
    dropout : float
        The dropout rate (probability of setting elements to zero)
    z : torch.Tensor
        The tensor to apply dropout to, shape [B, N, N, D]
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        If True, applies dropout columnwise [B, 1, N, 1],
        otherwise rowwise [B, N, 1, 1], by default False

    Returns
    -------
    torch.Tensor
        The dropout mask, shape [B, 1, N, 1] if columnwise else [B, N, 1, 1]
        Values are either 0 or 1/(1-dropout) to maintain expected values
    """
    dropout = dropout * training  # Only apply dropout during training
    # Select slice for mask generation based on orientation
    v = (
        z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    )  # [B, 1, N, 1] or [B, N, 1, 1]
    # Generate random mask and apply threshold
    d = torch.rand_like(v) > dropout  # [B, 1, N, 1] or [B, N, 1, 1]
    # Scale to maintain expected values
    d = d * 1.0 / (1.0 - dropout)  # [B, 1, N, 1] or [B, N, 1, 1]
    return d
