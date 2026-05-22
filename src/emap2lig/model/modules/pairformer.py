import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn

from ..layers import (
    MLP,
    AttentionPairBias,
    OuterProductMean,
    Transition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairFormerBlock(nn.Module):
    """Single PairFormer transformer block.

    Implements the core PairFormer operations:
    1. Pair-to-atom attention with pairwise bias
    2. Atom-to-pair outer product mean
    3. Triangle multiplication (outgoing/incoming)
    4. Triangle attention (starting/ending nodes)
    5. Transition layers for atoms and pairs
    """

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        n_heads: int = 8,
        head_dim: int = 32,
        transition_expansion_factor: int = 2,
        outer_product_mean_hidden_dim: int = 16,
        tri_attn_use_kernel: bool = False,
        tri_mul_use_kernel: bool = False,
    ) -> None:
        """Initialize PairFormerBlock.

        Parameters
        ----------
        atom_dim : int
            Atom feature dimension
        pair_dim : int
            Pair feature dimension
        n_heads : int, optional
            Number of attention heads, by default 8
        head_dim : int, optional
            Dimension per attention head, by default 32
        transition_expansion_factor : int, optional
            Expansion factor for transition layers, by default 2
        outer_product_mean_hidden_dim : int, optional
            Hidden dimension for outer product mean, by default 16
        tri_attn_use_kernel : bool, optional
            Use kernel implementation for triangle attention, by default False
        tri_mul_use_kernel : bool, optional
            Use kernel implementation for triangle multiplication, by default False
        """
        super().__init__()
        self.atom_dim = atom_dim
        self.pair_dim = pair_dim
        self.outer_product_mean_hidden_dim = outer_product_mean_hidden_dim
        self.tri_mul_use_kernel = tri_mul_use_kernel

        # Pair-to-atom attention with pairwise bias
        self.pair2atom = AttentionPairBias(
            c_s=atom_dim,
            c_z=pair_dim,
            num_heads=n_heads,
        )

        # Atom-to-pair outer product mean
        self.atom2pair = OuterProductMean(
            c_in=atom_dim,
            c_hidden=outer_product_mean_hidden_dim,
            c_out=pair_dim,
        )

        # Triangle multiplication updates
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pair_dim, use_cuequiv=tri_mul_use_kernel
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pair_dim, use_cuequiv=tri_mul_use_kernel
        )

        # Triangle attention layers
        self.tri_att_start = TriangleAttentionStartingNode(
            c_in=pair_dim,
            c_hidden=head_dim,
            no_heads=n_heads,
            use_cuequiv=tri_attn_use_kernel,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_in=pair_dim,
            c_hidden=head_dim,
            no_heads=n_heads,
            use_cuequiv=tri_attn_use_kernel,
        )

        # Transition layers
        self.transition_atom = Transition(
            atom_dim, atom_dim * transition_expansion_factor
        )
        self.transition_pair = Transition(
            pair_dim, pair_dim * transition_expansion_factor
        )

    def forward(
        self,
        atom_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through PairFormerBlock.

        Parameters
        ----------
        atom_feats : torch.Tensor
            Atom features of shape [B, N_atoms, atom_dim]
        pair_feats : torch.Tensor
            Pair features of shape [B, N_atoms, N_atoms, pair_dim]
        atom_mask : torch.Tensor
            Atom mask of shape [B, N_atoms]
        pair_mask : torch.Tensor
            Pair mask of shape [B, N_atoms, N_atoms]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated atom and pair features
        """
        # Create masks for zeroing outputs
        v_atom = atom_mask.float().unsqueeze(-1)  # [B, N, 1]
        v_pair = pair_mask.float().unsqueeze(-1)  # [B, N, N, 1]

        # Pair-to-atom attention with pairwise bias
        atom_feats = atom_feats + self.pair2atom(
            s=atom_feats,
            z=pair_feats,
            mask=atom_mask,
        )

        # Atom-to-pair outer product mean
        pair_feats = pair_feats + self.atom2pair(
            m=atom_feats.unsqueeze(1),
            mask=atom_mask.unsqueeze(1),
        )

        # Triangle multiplication updates
        pair_feats = pair_feats + self.tri_mul_out(pair_feats, mask=pair_mask)
        pair_feats = pair_feats + self.tri_mul_in(pair_feats, mask=pair_mask)

        # Triangle attention layers
        pair_feats = pair_feats + self.tri_att_start(
            pair_feats,
            mask=pair_mask,
        )
        pair_feats = pair_feats + self.tri_att_end(
            pair_feats,
            mask=pair_mask,
        )

        # Transition layers with output masking
        atom_feats = atom_feats + self.transition_atom(atom_feats) * v_atom
        pair_feats = pair_feats + self.transition_pair(pair_feats) * v_pair

        return atom_feats, pair_feats


class PairFormer(nn.Module):
    """PairFormer: central module for chemical feature propagation.

    Refines single-atom and pairwise representations through a stack of
    PairFormerBlock layers following the AlphaFold3 design (reduced to four
    blocks). Each block couples pair-to-atom attention, atom-to-pair outer
    product, triangle multiplication/attention, and transition layers to
    enforce geometric consistency across three-body relationships.
    """

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        num_blocks: int,
        num_heads: int = 16,
        head_dim: int = 32,
        transition_expansion_factor: int = 4,
        tri_attn_use_kernel: bool = False,
        tri_mul_use_kernel: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        """Initialize PairFormer.

        Parameters
        ----------
        atom_dim : int
            Atom feature dimension
        pair_dim : int
            Pair feature dimension
        num_blocks : int
            Number of PairFormerBlock layers
        num_heads : int, optional
            Number of attention heads, by default 16
        head_dim : int, optional
            Dimension per attention head, by default 32
        transition_expansion_factor : int, optional
            Expansion factor for transition layers, by default 4
        tri_attn_use_kernel : bool, optional
            Use kernel implementation for triangle attention, by default False
        tri_mul_use_kernel : bool, optional
            Use kernel implementation for triangle multiplication, by default False
        """
        super().__init__()

        self.atom_dim = atom_dim
        self.pair_dim = pair_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.transition_expansion_factor = transition_expansion_factor
        self.tri_attn_use_kernel = tri_attn_use_kernel
        self.tri_mul_use_kernel = tri_mul_use_kernel
        self.activation_checkpointing = activation_checkpointing

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                PairFormerBlock(
                    atom_dim=self.atom_dim,
                    pair_dim=self.pair_dim,
                    n_heads=self.num_heads,
                    head_dim=self.head_dim,
                    transition_expansion_factor=self.transition_expansion_factor,
                    tri_attn_use_kernel=self.tri_attn_use_kernel,
                    tri_mul_use_kernel=self.tri_mul_use_kernel,
                )
            )

        # Wrap blocks with activation checkpointing at construction time
        if self.activation_checkpointing:
            self.blocks = nn.ModuleList([checkpoint_wrapper(b) for b in self.blocks])

    def forward(
        self,
        atom_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through PairFormer.

        Parameters
        ----------
        atom_feats : torch.Tensor
            Atom features of shape [B, N_atoms, atom_dim]
        pair_feats : torch.Tensor
            Pair features of shape [B, N_atoms, N_atoms, pair_dim]
        atom_mask : torch.Tensor
            Atom mask of shape [B, N_atoms]
        pair_mask : torch.Tensor
            Pair mask of shape [B, N_atoms, N_atoms]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated atom and pair features
        """
        # Process through all PairFormerBlock layers (wrapped if checkpointing enabled)
        for block in self.blocks:
            atom_feats, pair_feats = block(
                atom_feats=atom_feats,
                pair_feats=pair_feats,
                atom_mask=atom_mask,
                pair_mask=pair_mask,
            )

        return atom_feats, pair_feats


class AuxiliaryModule(nn.Module):
    """Auxiliary prediction module.

    Predicts pairwise distance distributions and multiple classification tasks:
    - Distance distributions (distogram)
    - Atom element classification
    - Atom chirality classification
    - Atom ring size classification
    - Bond type classification
    - Bond ring size classification
    - Bond existence classification
    """

    def __init__(
        self,
        pair_dim: int,
        atom_dim: int,
        num_bins: int = 20,
        num_elements: int = 6,
        num_chirality_types: int = 7,
        num_bond_types: int = 5,
        num_ring_sizes: int = 4,
    ) -> None:
        """Initialize AuxiliaryModule.

        Parameters
        ----------
        pair_dim : int
            Pair feature dimension
        atom_dim : int
            Atom feature dimension
        num_bins : int, optional
            Number of distance bins, by default 8
        num_elements : int, optional
            Number of simplified element classes, by default 6 (C,N,O,P,S,Metal)
        num_chirality_types : int, optional
            Number of chirality types, by default 7
        num_bond_types : int, optional
            Number of bond types, by default 5
        num_ring_sizes : int, optional
            Number of ring size classes, by default 4
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_elements = num_elements
        self.num_chirality_types = num_chirality_types
        self.num_bond_types = num_bond_types
        self.num_ring_sizes = num_ring_sizes

        # Distance prediction head
        self.pair_disto_head = MLP(
            dim=pair_dim,
            hidden=pair_dim * 2,
            out_dim=num_bins,
        )

        # Atom element classification head
        self.atom_element_head = MLP(
            dim=atom_dim,
            hidden=atom_dim * 2,
            out_dim=num_elements,
        )

        # Atom chirality classification head
        self.atom_chirality_head = MLP(
            dim=atom_dim,
            hidden=atom_dim * 2,
            out_dim=num_chirality_types,
        )

        # Atom ring size classification head
        self.atom_ring_size_head = MLP(
            dim=atom_dim,
            hidden=atom_dim * 2,
            out_dim=num_ring_sizes,
        )

        # Bond type classification head
        self.bond_type_head = MLP(
            dim=pair_dim,
            hidden=pair_dim * 2,
            out_dim=num_bond_types,
        )

        # Bond ring size classification head
        self.bond_ring_size_head = MLP(
            dim=pair_dim,
            hidden=pair_dim * 2,
            out_dim=num_ring_sizes,
        )

        # Bond existence classification head (binary)
        self.bond_exists_head = MLP(
            dim=pair_dim,
            hidden=pair_dim * 2,
            out_dim=1,  # Binary classification: bond exists or not
        )

    def forward(
        self,
        pair_feats: torch.Tensor,
        pair_mask: torch.Tensor,
        atom_feats: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for auxiliary predictions including distogram and classification tasks.

        Parameters
        ----------
        pair_feats : torch.Tensor
            Pair features of shape [B, N_atoms, N_atoms, pair_dim]
        pair_mask : torch.Tensor
            Pair mask of shape [B, N_atoms, N_atoms]
        atom_feats : torch.Tensor
            Atom features of shape [B, N_atoms, atom_dim]
        atom_mask : torch.Tensor
            Atom mask of shape [B, N_atoms]

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing all predictions
        """
        outputs = {}

        # Distance prediction
        v_pair = pair_mask.float().unsqueeze(-1)
        pair_dist_logits = self.pair_disto_head(pair_feats) * v_pair
        outputs["pair_dist_logits"] = pair_dist_logits

        # Bond type prediction
        bond_type_logits = self.bond_type_head(pair_feats) * v_pair
        outputs["bond_type_logits"] = bond_type_logits

        # Bond ring size prediction
        bond_ring_logits = self.bond_ring_size_head(pair_feats) * v_pair
        outputs["bond_ring_logits"] = bond_ring_logits

        # Bond existence prediction
        bond_exists_logits = self.bond_exists_head(pair_feats) * v_pair
        outputs["bond_exists_logits"] = bond_exists_logits

        # Atom-level predictions
        v_atom = atom_mask.float().unsqueeze(-1)

        # Element prediction
        atom_element_logits = self.atom_element_head(atom_feats) * v_atom
        outputs["atom_element_logits"] = atom_element_logits

        # Chirality prediction
        atom_chirality_logits = self.atom_chirality_head(atom_feats) * v_atom
        outputs["atom_chirality_logits"] = atom_chirality_logits

        # Ring size prediction
        atom_ring_logits = self.atom_ring_size_head(atom_feats) * v_atom
        outputs["atom_ring_logits"] = atom_ring_logits

        return outputs
