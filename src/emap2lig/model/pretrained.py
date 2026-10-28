"""Pretrained Emap2lig models with the architecture of the released checkpoints."""

from ..config import DETECTION_SPATIAL_SIZE
from ..utils.weights import (
    DETECTION_WEIGHTS,
    FRAGMENT_WEIGHTS,
    STRUCTURE_WEIGHTS,
    WEIGHTS_REPO_ID,
)
from .model import (
    AuxiliaryArgs,
    ConformerEmbedderArgs,
    DiffusionArgs,
    DiffusionPredictArgs,
    Emap2lig,
    InstanceSegArgs,
    PairformerArgs,
)
from .modules.diffusion import DiffusionModuleArgs
from .seg import FragmentRegSeg, MUNetRegSeg


def _seg_kwargs() -> dict:
    return dict(
        in_channels=1,
        mid_channels=32,
        reg_channels=6,
        seg_channels=6,
        spatial_size=DETECTION_SPATIAL_SIZE,
        n_res_blocks=1,
        attention_levels=[2],
        channel_multipliers=[1, 2, 4],
        n_heads=4,
        tf_layers=1,
        kernel_size=5,
        num_groups=32,
        load_pretrained=True,
        repo_id=WEIGHTS_REPO_ID,
    )


def build_detection_model() -> MUNetRegSeg:
    """Stage 1 ligand/macromolecule segmentation model with pretrained weights."""
    return MUNetRegSeg(**_seg_kwargs(), filename=DETECTION_WEIGHTS)


def build_fragment_detection_model() -> FragmentRegSeg:
    """Fragment segmentation model with pretrained weights."""
    return FragmentRegSeg(**_seg_kwargs(), filename=FRAGMENT_WEIGHTS)


def build_structure_model(num_sampling_steps: int = 100) -> Emap2lig:
    """Stage 2 diffusion structure model with pretrained weights."""
    d_atom_in, d_pair_in, d_point_in = 149, 9, 76
    atom_dim, pair_dim, point_dim = 128, 64, 64
    return Emap2lig(
        d_atom_in=d_atom_in,
        d_pair_in=d_pair_in,
        d_point_in=d_point_in,
        d_atom_hidden=atom_dim,
        d_pair_hidden=pair_dim,
        d_point_hidden=point_dim,
        load_pretrained=True,
        repo_id=WEIGHTS_REPO_ID,
        filename=STRUCTURE_WEIGHTS,
        is_conf_embedder_compiled=True,
        is_pairformer_compiled=False,
        is_instance_seg_compiled=True,
        is_auxiliary_module_compiled=True,
        is_diffusion_module_compiled=True,
        predict_args=DiffusionPredictArgs(
            multiplicity=16,
            num_sampling_steps=num_sampling_steps,
            output_format="mmcif",
        ),
        conf_embedder_args=ConformerEmbedderArgs(
            atom_dim_in=d_atom_in,
            pair_dim_in=d_pair_in,
            atom_dim=atom_dim,
            pair_dim=pair_dim,
        ),
        pairformer_args=PairformerArgs(
            atom_dim=atom_dim,
            pair_dim=pair_dim,
            num_blocks=4,
            num_heads=8,
            head_dim=16,
            transition_expansion_factor=4,
            activation_checkpointing=False,
        ),
        instance_seg_args=InstanceSegArgs(num_selected_points=8192),
        auxiliary_args=AuxiliaryArgs(
            pair_dim=pair_dim,
            atom_dim=atom_dim,
            num_bins=20,
            num_elements=6,
            num_chirality_types=7,
            num_bond_types=5,
            num_ring_sizes=4,
        ),
        diffusion_args=DiffusionArgs(
            sigma_min=0.0004,
            sigma_max=48.0,
            sigma_data=4.0,
            rho=7.0,
            P_mean=-1.2,
            P_std=1.2,
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.0,
            step_scale=1.0,
            diffusion_model_args=DiffusionModuleArgs(
                atom_dim=atom_dim,
                pair_dim=pair_dim,
                point_dim=point_dim,
                aggregation_dim=384,
                fourier_dim=256,
                conditioning_transition_layers=2,
                num_point_blocks=8,
                num_heads=8,
                head_dim=16,
                activation_checkpointing=False,
            ),
        ),
    )
