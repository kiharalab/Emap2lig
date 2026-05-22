import torch
from torch import nn

from ..layers import LinearNoBias


class ConformerEmbedder(nn.Module):
    """Conformer Embedder: translates a reference ligand conformer into single-atom
    and pairwise feature representations.

    Ingests per-atom features, pairwise features, and 3D reference coordinates.
    Computes relative position vectors and normalized interatomic distances for all
    atom pairs, fusing geometric and chemical information into the pairwise
    representation. Single-atom embeddings are projected into pair space so that
    each pair representation is conditioned on both constituent atoms.
    """

    def __init__(
        self,
        atom_dim_in: int,
        pair_dim_in: int,
        atom_dim: int,
        pair_dim: int,
    ):
        super().__init__()

        self.embed_atom_features = LinearNoBias(atom_dim_in, atom_dim)
        self.embed_pair_features = LinearNoBias(pair_dim_in, pair_dim)
        self.embed_atom_pair_ref_pos = LinearNoBias(3, pair_dim)
        self.embed_atom_pair_ref_dist = LinearNoBias(1, pair_dim)
        self.embed_atom_pair_mask = LinearNoBias(1, pair_dim)
        self.proj_atom_to_pair = LinearNoBias(atom_dim, pair_dim)

    def forward(
        self,
        ref_pos: torch.Tensor,
        atom_feature: torch.Tensor,
        pair_feature: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to embed input features.

        Args:
            ref_pos: Reference positions [B, N_a, 3]
            atom_feature: Atom features [B, N_a, atom_dim_in]
            pair_feature: Pair features [B, N_a, N_a, pair_dim_in]
            atom_mask: Atom mask [B, N_a]
            pair_mask: Pair mask [B, N_a, N_a]

        Returns:
            tuple containing:
                - atom_feats: Embedded atom features [B, N_a, atom_dim]
                - pair_feats: Embedded pair features [B, N_a, N_a, pair_dim]
        """
        atom_feats = self.embed_atom_features(atom_feature) * atom_mask.unsqueeze(
            -1
        )  # [B, N_a, d_atom]

        d = ref_pos[:, :, None, :] - ref_pos[:, None, :, :]  # [B, N_a, N_a, 3]
        d_norm = torch.sum(torch.square(d), dim=-1, keepdim=True)  # [B, N_a, N_a, 1]
        d_norm = 1 / (1 + d_norm)  # [B, N_a, N_a, 1]

        v_pair = pair_mask.float().unsqueeze(-1)  # [B, N_a, N_a, 1]

        pair_feats = (
            self.embed_pair_features(pair_feature) * v_pair
        )  # [B, N_a, N_a, d_pair]
        pair_feats = (
            pair_feats + self.embed_atom_pair_ref_pos(d) * v_pair
        )  # [B, N_a, N_a, d_pair]
        pair_feats = (
            pair_feats + self.embed_atom_pair_ref_dist(d_norm) * v_pair
        )  # [B, N_a, N_a, d_pair]
        pair_feats = (
            pair_feats + self.embed_atom_pair_mask(v_pair) * v_pair
        )  # [B, N_a, N_a, d_pair]

        v_atom = atom_mask.float().unsqueeze(-1)  # [B, N_a, 1]

        # Add the combined single condition to the pair features
        atom_to_pair = self.proj_atom_to_pair(atom_feats) * v_atom  # [B, N_a, d_pair]
        pair_feats = (
            pair_feats + atom_to_pair[:, :, None, :] + atom_to_pair[:, None, :, :]
        )  # [B, N_a, N_a, d_pair]

        # Final masking to ensure padded positions are zero
        pair_feats = pair_feats * v_pair

        return atom_feats, pair_feats
