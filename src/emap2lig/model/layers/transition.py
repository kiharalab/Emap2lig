import math

import torch.nn.init as init
from torch import Tensor, nn

from .primitives import LayerNorm


class Transition(nn.Module):
    """Two-layer MLP with SiLU gating and residual-style design.

    Implements a transition layer with layer normalization, dual linear projections
    for gating, SiLU activation, and a final output projection. Commonly used in
    transformer architectures for feature transformation.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: int | None = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        init.ones_(self.norm.weight)
        init.zeros_(self.norm.bias)

        # lecun_normal_init_ equivalent: trunc_normal with scale=1.0
        init.trunc_normal_(
            self.fc1.weight, std=math.sqrt(1.0 / max(1, self.fc1.weight.shape[1]))
        )
        init.trunc_normal_(
            self.fc2.weight, std=math.sqrt(1.0 / max(1, self.fc2.weight.shape[1]))
        )
        init.zeros_(self.fc3.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transition layer.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., input_dim)

        Returns
        -------
        torch.Tensor
            The output data of shape (..., out_dim)
        """
        x = self.norm(x)  # (..., input_dim)
        # Gated activation: SiLU(fc1(x)) * fc2(x)
        x = self.silu(self.fc1(x)) * self.fc2(x)  # (..., hidden_dim)
        x = self.fc3(x)  # (..., out_dim)
        return x
