import torch
from einops import rearrange
from torch import nn

from ..layers import InstanceSeg
from ..seg.model import SegHead
from ..seg.munet import MUNetBackbone


class InstanceSegModule(nn.Module):
    """Instance segmentation module for cryo-EM density maps.

    Predicts a ligand-specific 3D probability mask to isolate the target instance
    within the cryo-EM map. Integrates dense voxel features from a MUNet backbone
    with instance-level atomic context via cross-attention, then extracts global
    and per-point features for downstream diffusion conditioning.
    """

    def __init__(
        self,
        channels: int = 64,
        n_res_blocks: int = 1,
        attention_levels: tuple[int, ...] = (2,),
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        n_heads: int = 8,
        tf_layers: int = 1,
        kernel_size: int = 5,
        num_groups: int = 32,
        atom_dim: int = 128,
        instance_num_blocks: int = 4,
        instance_num_attention_heads: int = 8,
        voxel_global_dim: int = 384,
        voxel_num_groups: int = 8,
        augment_num_output_channels: int = 15,
        num_selected_points: int = 8192,
    ) -> None:
        super().__init__()

        # Architecture components
        self.backbone = MUNetBackbone(
            in_channels=1,
            channels=channels,
            n_res_blocks=n_res_blocks,
            attention_levels=list(attention_levels),
            channel_multipliers=list(channel_multipliers),
            n_heads=n_heads,
            tf_layers=tf_layers,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )

        self.seg_head = SegHead(
            in_channels=channels, out_channels=augment_num_output_channels
        )
        self.seg_act = torch.sigmoid

        # Instance segmentation module
        self.instance_seg = InstanceSeg(
            channels=channels,
            atom_dim=atom_dim,
            num_blocks=instance_num_blocks,
            num_attention_heads=instance_num_attention_heads,
        )

        # Conv3d projector for voxel features (backbone + augment + instance -> 64)
        self.voxel_projector = nn.Sequential(
            nn.Conv3d(
                augment_num_output_channels + channels + 1,
                channels,
                kernel_size=1,
            ),
            nn.GroupNorm(voxel_num_groups, channels),
            nn.SiLU(inplace=True),
        )

        self.labels = [
            "ligand",
            "backbone",
            "sidechain",
            "sugar",
            "phosphate",
            "base",
            "C",
            "N",
            "O",
            "P",
            "S",
            "Metal",
            "Ring4",
            "Ring5",
            "Ring6",
        ]

        # Global feature extraction: fuse coords with voxel feats then pool
        self.global_fuse = nn.Sequential(
            nn.Conv3d(channels + 3, voxel_global_dim, kernel_size=1),
            nn.GroupNorm(voxel_num_groups, voxel_global_dim),
            nn.SiLU(inplace=True),
        )
        self.instance_gate = nn.Sequential(
            nn.Conv3d(1, voxel_global_dim, kernel_size=1),
            nn.GroupNorm(voxel_num_groups, voxel_global_dim),
            nn.SiLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.num_selected_points = num_selected_points

    def forward_embedding(
        self,
        input_map: torch.Tensor,
        global_origin: torch.Tensor,
        voxel_size: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        prompt_point: torch.Tensor,
        multiplicity: int = 1,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass to compute instance segmentation and 64-channel voxel features.

        Args:
            input_map (torch.Tensor): Input tensor of shape [B, 1, 48, 48, 48].
            global_origin (torch.Tensor): Global origin tensor of shape [B, 3].
            voxel_size (torch.Tensor): Voxel size tensor of shape [B, 3].
            atom_features (torch.Tensor): Atom features from PairFormer trunk of shape [B, N_atoms, D_atom].
            atom_mask (torch.Tensor): Atom mask from PairFormer trunk of shape [B, N_atoms].
            prompt_point (torch.Tensor): Prompt points of shape [B*multiplicity, 3] for relative position computation.
            multiplicity (int): Multiplicity factor for sampling, by default 1.

        Returns:
            Tuple of (augment_output, instance_output, voxel_features, global_features,
            selected_point_feats, selected_point_coords).
            - augment_output: [B, 15, 48, 48, 48]
            - instance_output: [B*multiplicity, 1, 48, 48, 48]
            - voxel_features: [B*multiplicity, 64, 48, 48, 48]
            - global_features: [B*multiplicity, 1, 256]
            - selected_point_feats: [B*multiplicity, num_selected_points, 64]
            - selected_point_coords: [B*multiplicity, num_selected_points, 3]
        """
        # get backbone features and augment output from pretrained AugmentRegSeg model
        features = self.backbone(input_map)  # [B, 64, 48, 48, 48]
        augment_output = self.seg_act(self.seg_head(features))  # [B, 15, 48, 48, 48]

        # Assert all required inputs are provided for instance segmentation
        assert atom_features is not None, (
            "atom_features must be provided for instance segmentation"
        )
        assert atom_mask is not None, (
            "atom_mask must be provided for instance segmentation"
        )
        assert prompt_point is not None, (
            "prompt_point must be provided for instance segmentation"
        )

        # Repeat volume-level features to match multiplicity for processing each prompt point
        features_expanded = features.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, 64, 48, 48, 48]
        atom_features_expanded = atom_features.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, N_atoms, atom_dim]
        atom_mask_expanded = atom_mask.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, N_atoms]
        global_origin_expanded = global_origin.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, 3]
        voxel_size_expanded = voxel_size.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, 3]

        # Compute relative positions for each prompt point
        rel_positions = self.compute_rel_pos_3d(
            global_origin=global_origin_expanded,
            voxel_size=voxel_size_expanded,
            prompt_point=prompt_point,  # [B*multiplicity, 3] - use all prompt points
            volume_shape=(48, 48, 48),  # Shape of the backbone features
        )  # [B*multiplicity, 3, 48, 48, 48]

        instance_output = self.instance_seg(
            volume_features=features_expanded,  # [B*multiplicity, 64, 48, 48, 48]
            atom_features=atom_features_expanded,  # [B*multiplicity, N_atoms, atom_dim]
            atom_mask=atom_mask_expanded,  # [B*multiplicity, N_atoms]
            rel_positions=rel_positions,  # [B*multiplicity, 3, 48, 48, 48]
        )  # [B*multiplicity, 1, 48, 48, 48] - instance mask

        # Expand augment_output to match multiplicity for feature concatenation
        augment_output_expanded = augment_output.repeat_interleave(
            multiplicity, 0
        )  # [B*multiplicity, 15, 48, 48, 48]

        # Use predicted instance mask from MUNet detection
        instance_mask_feats = instance_output

        # Concatenate backbone, augment, and instance features then project to 64ch
        concatenated_features = torch.cat(
            [
                features_expanded,  # [B*multiplicity, 64, 48, 48, 48]
                augment_output_expanded,  # [B*multiplicity, 15, 48, 48, 48]
                instance_mask_feats,  # [B*multiplicity, 1, 48, 48, 48]
            ],
            dim=1,
        )  # [B*multiplicity, 80, 48, 48, 48]

        voxel_features = self.voxel_projector(
            concatenated_features
        )  # [B*multiplicity, 64, 48, 48, 48]

        # Compute absolute world coordinates grid [B, 3, D, H, W]
        world_coords = self.compute_world_coords_3d(
            global_origin=global_origin_expanded,
            voxel_size=voxel_size_expanded,
            volume_shape=(48, 48, 48),
        )  # [B, 3, 48, 48, 48]

        # Concatenate coords with voxel features along channel dim and apply gate
        fused = torch.cat([voxel_features, world_coords], dim=1)  # [B, 67, D, H, W]
        fused_proj = self.global_fuse(fused)  # [B, 256, D, H, W]
        fuse_gate = self.instance_gate(instance_mask_feats)  # [B, 256, D, H, W]
        global_features = (
            self.global_pool(fused_proj * fuse_gate).flatten(1).unsqueeze(1)
        )  # [B, 1, 256]

        # Select top-k points based on instance probabilities (done once, reused across blocks)
        selected_point_feats, selected_point_coords = self.select_top_k_points(
            voxel_features=voxel_features,
            instance_probs=instance_mask_feats,
            world_coords=world_coords,
        )

        return (
            augment_output,
            instance_output,
            voxel_features,
            global_features,
            selected_point_feats,
            selected_point_coords,
        )

    def select_top_k_points(
        self,
        voxel_features: torch.Tensor,
        instance_probs: torch.Tensor,
        world_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k points based on instance segmentation probabilities.

        This centralizes the point selection that was previously done per-block
        in SelectedCrossAttention. Since instance_probs doesn't change between
        conditioning blocks, we select once and reuse.

        Args:
            voxel_features (torch.Tensor): Voxel features of shape [B, C_v, D, H, W]
            instance_probs (torch.Tensor): Instance probabilities of shape [B, 1, D, H, W]
            world_coords (torch.Tensor): World coordinates of shape [B, 3, D, H, W]

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - selected_feats: Selected point features of shape [B, num_points, C_v]
                - selected_coords: Selected point coordinates of shape [B, num_points, 3]
        """
        _, C_v, _, _, _ = voxel_features.shape

        # Flatten voxel features, coords, and probabilities
        voxel_flat = rearrange(
            voxel_features, "b c d h w -> b (d h w) c"
        )  # [B, N, C_v]
        coords_flat = rearrange(world_coords, "b c d h w -> b (d h w) c")  # [B, N, 3]
        prob_flat = rearrange(instance_probs, "b 1 d h w -> b (d h w)")  # [B, N]

        # Select top-k points based on instance probabilities
        topk_idx = torch.topk(
            prob_flat, k=self.num_selected_points, dim=1, largest=True, sorted=False
        ).indices  # [B, num_points]

        # Gather features and coordinates for selected points
        selected_feats = torch.gather(
            voxel_flat,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, C_v),
        ).contiguous()  # [B, num_points, C_v]

        selected_coords = torch.gather(
            coords_flat,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, 3),
        ).contiguous()  # [B, num_points, 3]

        return selected_feats, selected_coords

    @staticmethod
    def _voxel_to_world(
        global_origin: torch.Tensor,
        voxel_size: torch.Tensor,
        volume_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Convert voxel indices to world coordinates.

        Returns:
            World coordinates of shape [B, 3, D, H, W].
        """
        B = global_origin.shape[0]
        D, H, W = volume_shape
        device = global_origin.device

        z, y, x = torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        voxel_indices = torch.stack([x, y, z], dim=-1).float()  # [D, H, W, 3]
        world_coords = voxel_indices.unsqueeze(0) * voxel_size.view(
            B, 1, 1, 1, 3
        ) + global_origin.view(B, 1, 1, 1, 3)  # [B, D, H, W, 3]
        return world_coords.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]

    def compute_rel_pos_3d(
        self,
        global_origin: torch.Tensor,
        voxel_size: torch.Tensor,
        prompt_point: torch.Tensor,
        volume_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute 3D relative positions between a prompt point and all voxels.

        Returns:
            Relative position volume of shape [B, 3, D, H, W].
        """
        B = global_origin.shape[0]
        world_coords = self._voxel_to_world(
            global_origin, voxel_size, volume_shape
        )  # [B, 3, D, H, W]
        # world_coords is [B, 3, D, H, W]; subtract prompt_point [B, 3, 1, 1, 1]
        return world_coords - prompt_point.view(B, 3, 1, 1, 1)

    def compute_world_coords_3d(
        self,
        global_origin: torch.Tensor,
        voxel_size: torch.Tensor,
        volume_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute absolute 3D world coordinates for each voxel.

        Returns:
            World coordinates of shape [B, 3, D, H, W].
        """
        return self._voxel_to_world(global_origin, voxel_size, volume_shape)
