import torch
from torch import nn

from .primitives import LayerNorm, LinearNoBias


class AtomDecoder(nn.Module):
    """Atom coordinate decoder.

    Decodes atom features into 3D coordinates using layer normalization
    followed by a linear projection.
    """

    def __init__(
        self,
        atom_dim: int,
    ) -> None:
        """Initialize AtomDecoder.

        Parameters
        ----------
        atom_dim : int
            Dimension of input atom features
        """
        super().__init__()

        self.layer_norm = LayerNorm(atom_dim)
        self.decode_coords = LinearNoBias(atom_dim, 3)
        self.__init_weights__()

    def __init_weights__(self):
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(
        self,
        atom_feats: torch.Tensor,  # [B, N_a, C_a]
    ) -> torch.Tensor:  # [B, N_a, 3]
        """Decode atom features to coordinates.

        Parameters
        ----------
        atom_feats : Tensor
            Atom features, shape [B, N_a, C_a]

        Returns
        -------
        Tensor
            Predicted coordinates, shape [B, N_a, 3]
        """
        atom_feats = self.layer_norm(atom_feats)  # [B, N_a, C_a]
        return self.decode_coords(atom_feats)  # [B, N_a, 3]
