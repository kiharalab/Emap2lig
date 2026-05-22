import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn
from torch.nn import Module, ModuleList

from ..layers import (
    AttentionPairBias,
    FourierEmbedding,
    LayerNorm,
    Linear,
    LinearNoBias,
    PositionalEncoder,
    SelectedCrossAttention,
    Transition,
)


class AtomConditioner(Module):
    """Atom conditioning with temporal and positional encoding.

    Combines current and initial atom features, adds positional encoding from
    coordinates and a time-dependent Fourier embedding, then refines with
    lightweight transition(s).
    """

    def __init__(
        self,
        atom_dim: int,
        fourier_dim: int,
        num_transitions: int = 1,
        transition_expansion_factor: int = 2,
        activation_checkpointing: bool = False,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        atom_dim : int
            The atom representation dimension.
        fourier_dim : int
            The fourier embeddings dimension.
        num_transitions : int
            The number of transitions layers.
        transition_expansion_factor : int
            The transition expansion factor.
        """
        super().__init__()

        # Input projection (concat current + initial atom features)
        input_dim = atom_dim * 2
        self.norm_single = LayerNorm(input_dim)
        self.single_embed = Linear(input_dim, atom_dim)
        self.pos_enc = PositionalEncoder(atom_dim)

        # Fourier embedding for time
        self.fourier_embed = FourierEmbedding(fourier_dim)
        self.norm_fourier = LayerNorm(fourier_dim)
        self.fourier_to_atom = LinearNoBias(fourier_dim, atom_dim)

        # Transition layers
        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = nn.Sequential(
                LayerNorm(atom_dim),
                Linear(atom_dim, transition_expansion_factor * atom_dim),
                nn.SiLU(),
                Linear(transition_expansion_factor * atom_dim, atom_dim),
            )
            transitions.append(transition)
        self.transitions = transitions
        self.activation_checkpointing = activation_checkpointing

    def forward(
        self,
        atom_feats: torch.Tensor,  # [B, N_a, C_a]
        atom_init_feats: torch.Tensor,  # [B, N_a, C_a]
        atom_coords: torch.Tensor,  # [B, N_a, 3]
        atom_mask: torch.Tensor,  # [B, N_a]
        times: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, N_a, C_a]
        """Forward pass for atom conditioning.

        Parameters
        ----------
        atom_feats : torch.Tensor
            Current atom features, shape [B, N_a, C_a]
        atom_init_feats : torch.Tensor
            Initial atom features, shape [B, N_a, C_a]
        atom_coords : torch.Tensor
            Atom coordinates, shape [B, N_a, 3]
        atom_mask : torch.Tensor
            Atom mask, shape [B, N_a]
        times : torch.Tensor
            Time steps, shape [B]

        Returns
        -------
        torch.Tensor
            Conditioned atom features, shape [B, N_a, C_a]
        """
        # Concatenate atom features and inputs
        v_atom = atom_mask.float().unsqueeze(-1)  # [B, N_a, 1]
        x = torch.cat([atom_feats, atom_init_feats], dim=-1)
        x = self.norm_single(x)
        a = self.single_embed(x) * v_atom
        a = self.pos_enc(atom_coords, a) * v_atom  # Mask after positional encoding
        # Apply Fourier embedding to time steps
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        # Project fourier embeddings to atom dimension
        fourier_proj = self.fourier_to_atom(normed_fourier)
        # Add fourier projection to atom features
        a = a + fourier_proj.unsqueeze(1) * v_atom
        # Apply transition layers
        for transition in self.transitions:
            a = a + transition(a) * v_atom
        return a


class PointConditioner(nn.Module):
    """Point cloud conditioning.

    Adds relative position features from a `prompt_point`, then applies a
    sequence of `PointConditionerBlock` modules that perform cross attention
    against pre-selected point features and pair-biased updates.
    """

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        point_dim: int,
        aggregation_dim: int,
        num_blocks: int = 4,
        num_heads: int = 3,
        head_dim: int = 16,
        num_transitions: int = 1,
        transition_expansion_factor: int = 2,
        length_scale: float = 1.0,
        use_global_feats: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.length_scale = length_scale
        self.use_global_feats = use_global_feats
        self.activation_checkpointing = activation_checkpointing
        # Relative position encoding projection for atoms
        self.atom_rel_pos_proj = nn.Linear(3, atom_dim)

        self.blocks = nn.ModuleList(
            [
                PointConditionerBlock(
                    atom_dim=atom_dim,
                    pair_dim=pair_dim,
                    point_dim=point_dim,
                    aggregation_dim=aggregation_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_transitions=num_transitions,
                    transition_expansion_factor=transition_expansion_factor,
                    use_global_feats=use_global_feats,
                )
                for _ in range(num_blocks)
            ]
        )
        if self.activation_checkpointing:
            self.blocks = nn.ModuleList([checkpoint_wrapper(b) for b in self.blocks])

    def forward(
        self,
        atom_feats: torch.Tensor,  # [B, N_a, C_a]
        pair_feats: torch.Tensor,  # [B, N_a, N_a, C_p]
        selected_point_feats: torch.Tensor,  # [B, num_points, C_v]
        selected_point_coords: torch.Tensor,  # [B, num_points, 3]
        atom_coords: torch.Tensor,  # [B, N_a, 3]
        atom_mask: torch.Tensor,  # [B, N_a]
        prompt_point: torch.Tensor,  # [B, 3]
        global_features: torch.Tensor,  # [B, aggregation_dim]
    ) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        atom_feats : torch.Tensor
            The atom embeddings, shape [B, N_a, C_a]
        pair_feats : torch.Tensor
            The pair features, shape [B, N_a, N_a, C_p]
        selected_point_feats : torch.Tensor
            Pre-selected point features from EM embedder, shape [B, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates from EM embedder, shape [B, num_points, 3]
        atom_coords : torch.Tensor
            The atom coordinates, shape [B, N_a, 3]
        atom_mask : torch.Tensor
            The atom mask, shape [B, N_a]
        prompt_point : torch.Tensor
            Prompt point of shape [B, 3] for relative position encoding
        global_features : torch.Tensor | None
            The global point features, shape [B, aggregation_dim]

        Returns
        -------
        torch.Tensor
            The updated atom embeddings, shape [B, N_a, C_a]
        """
        atom_mask = atom_mask.bool()

        # Apply relative position encoding with prompt point and mask, then project
        rel_pos_feats = self.compute_rel_pos(
            atom_coords=atom_coords,
            prompt_point=prompt_point,
            atom_mask=atom_mask,
        )
        atom_feats = atom_feats + rel_pos_feats

        # Apply blocks with pre-selected points (selection done once in EM embedder)
        for block in self.blocks:
            atom_feats = block(
                atom_feats=atom_feats,
                pair_feats=pair_feats,
                selected_point_feats=selected_point_feats,
                selected_point_coords=selected_point_coords,
                atom_coords=atom_coords,
                atom_mask=atom_mask,
                global_point_feats=global_features,
            )

        return atom_feats

    def compute_rel_pos(
        self,
        atom_coords: torch.Tensor,
        prompt_point: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked relative positions and apply projection.

        Args:
            atom_coords: Atom coordinates. Shape: [B, N_a, 3]
            prompt_point: Prompt point per batch. Shape: [B, 3]
            atom_mask: Boolean mask for valid atoms. Shape: [B, N_a]
        Returns:
            Projected, mask-applied relative position features.
            Shape: [B, N_a, C_a]
        """
        _, N_a, _ = atom_coords.shape

        # Expand prompt_point to [B, N_a, 3]
        prompt_expanded = prompt_point.unsqueeze(1).expand(-1, N_a, -1)

        # Compute relative positions and apply mask
        rel_pos = atom_coords - prompt_expanded  # [B, N_a, 3]
        v_atom = atom_mask.float().unsqueeze(-1)  # [B, N_a, 1]
        rel_pos = rel_pos * v_atom

        # Apply linear projection to match atom feature dimension
        rel_pos_feats = self.atom_rel_pos_proj(rel_pos)  # [B, N_a, C_a]

        return rel_pos_feats


class PointConditionerBlock(nn.Module):
    """A single point conditioning block.

    1) Point-to-atom cross-attention using pre-selected points
    2) Optional fusion of global point features
    3) Pair-biased attention update
    """

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        point_dim: int,
        aggregation_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        num_transitions: int = 1,
        transition_expansion_factor: int = 2,
        use_global_feats: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_global_feats = use_global_feats

        self.cross_attn = SelectedCrossAttention(
            atom_dim=atom_dim,
            point_dim=point_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.attention_pair_bias = AttentionPairBias(
            c_s=atom_dim,
            c_z=pair_dim,
            num_heads=num_heads,
        )

        # Optional transition for global point features
        self.point_transition: Transition | None = None
        if use_global_feats:
            self.point_transition = Transition(
                dim=aggregation_dim,
                hidden=aggregation_dim,
                out_dim=atom_dim,
            )

        # No per-block gating; handled inside attention

    def forward(
        self,
        atom_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        selected_point_feats: torch.Tensor,
        selected_point_coords: torch.Tensor,
        atom_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        global_point_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for a single conditioning block.

        Parameters
        ----------
        atom_feats : torch.Tensor
            Atom features, shape [B, N_a, C_a]
        pair_feats : torch.Tensor
            Pair features, shape [B, N_a, N_a, C_p]
        selected_point_feats : torch.Tensor
            Pre-selected point features, shape [B, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates, shape [B, num_points, 3]
        atom_coords : torch.Tensor
            Atom coordinates, shape [B, N_a, 3]
        atom_mask : torch.Tensor
            Atom mask, shape [B, N_a]
        global_point_feats : torch.Tensor
            Global point features, shape [B, aggregation_dim]

        Returns
        -------
        torch.Tensor
            Updated atom features, shape [B, N_a, C_a]
        """
        # Create mask for zeroing outputs: [B, N_a, 1]
        v_mask = atom_mask.float().unsqueeze(-1)

        # Apply pair attention bias
        atom_feats = atom_feats + self.attention_pair_bias(
            s=atom_feats,
            z=pair_feats,
            mask=atom_mask,
        )

        atom_feats = self.cross_attn(
            atom_feats=atom_feats,
            atom_coords=atom_coords,
            selected_point_feats=selected_point_feats,
            selected_point_coords=selected_point_coords,
            atom_mask=atom_mask,
        )

        # Process and add global point features (if available), masked
        if (
            isinstance(self.point_transition, Transition)
            and global_point_feats is not None
        ):
            processed_global_feats = self.point_transition(global_point_feats)
            atom_feats = atom_feats + processed_global_feats * v_mask

        return atom_feats
