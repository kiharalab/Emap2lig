from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor

from .modules.conf_embedder import ConformerEmbedder
from .modules.diffusion import AtomDiffusion, DiffusionModuleArgs
from .modules.instance_seg import InstanceSegModule
from .modules.pairformer import AuxiliaryModule, PairFormer

from loguru import logger

_MODEL_DIR = str(Path.home() / ".emap2lig" / "models")


@dataclass
class DiffusionPredictArgs:
    """Prediction arguments for DiffusionStructureModel."""

    multiplicity: int = 4
    num_sampling_steps: int = 20
    output_format: str = "mmcif"


@dataclass
class ConformerEmbedderArgs:
    """Arguments for the ConformerEmbedder module.

    Parameters
    ----------
    atom_dim_in : int
        Input dimension for atom features.
    pair_dim_in : int
        Input dimension for pair features.
    atom_dim : int
        Output dimension for atom embeddings.
    pair_dim : int
        Output dimension for pair embeddings.
    """

    atom_dim_in: int
    pair_dim_in: int
    atom_dim: int
    pair_dim: int


@dataclass
class PairformerArgs:
    """Arguments for the PairFormer module.

    Parameters
    ----------
    atom_dim : int
        Atom feature dimension.
    pair_dim : int
        Pair feature dimension.
    num_blocks : int
        Number of PairFormerBlock layers.
    num_heads : int, optional
        Number of attention heads, by default 16.
    head_dim : int, optional
        Dimension per attention head, by default 32.
    transition_expansion_factor : int, optional
        Expansion factor for transition layers, by default 4.
    tri_attn_use_kernel : bool, optional
        Use kernel implementation for triangle attention, by default False.
    tri_mul_use_kernel : bool, optional
        Use kernel implementation for triangle multiplication, by default False.
    """

    atom_dim: int
    pair_dim: int
    num_blocks: int
    num_heads: int = 16
    head_dim: int = 32
    transition_expansion_factor: int = 4
    tri_attn_use_kernel: bool = False
    tri_mul_use_kernel: bool = False
    activation_checkpointing: bool = False


@dataclass
class InstanceSegArgs:
    """Arguments for the InstanceSegModule.

    Parameters
    ----------
    channels : int, optional
        Number of channels, by default 64.
    n_res_blocks : int, optional
        Number of residual blocks, by default 1.
    attention_levels : list[int], optional
        Attention levels, by default [2].
    channel_multipliers : list[int], optional
        Channel multipliers, by default [1, 2, 4].
    n_heads : int, optional
        Number of attention heads, by default 8.
    tf_layers : int, optional
        Number of transformer layers, by default 1.
    kernel_size : int, optional
        Kernel size, by default 5.
    num_groups : int, optional
        Number of groups for group normalization, by default 32.
    num_selected_points : int, optional
        Number of points to select based on instance probabilities, by default 8192.
    """

    channels: int = 64
    n_res_blocks: int = 1
    attention_levels: tuple[int, ...] = (2,)
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    n_heads: int = 8
    tf_layers: int = 1
    kernel_size: int = 5
    num_groups: int = 32
    num_selected_points: int = 8192


@dataclass
class DiffusionArgs:
    """Arguments for the AtomDiffusion module.

    Parameters
    ----------
    diffusion_model_args : DiffusionModelArgs
        Arguments for the diffusion model.
    sigma_min : float, optional
        Minimum noise level, by default 0.0004.
    sigma_max : float, optional
        Maximum noise level, by default 160.0.
    sigma_data : float, optional
        Standard deviation of the data distribution, by default 16.0.
    rho : float, optional
        Controls the shape of the noise schedule, by default 7.0.
    P_mean : float, optional
        Mean for log-normal noise distribution, by default -1.2.
    P_std : float, optional
        Std for log-normal noise distribution, by default 1.2.
    """

    diffusion_model_args: DiffusionModuleArgs
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.0
    step_scale: float = 1.0


@dataclass
class AuxiliaryArgs:
    """Arguments for the AuxiliaryModule.

    Parameters
    ----------
    pair_dim : int
        Pair feature dimension.
    atom_dim : int
        Atom feature dimension.
    num_bins : int, optional
        Number of distance bins, by default 20.
    num_elements : int, optional
        Number of simplified element classes, by default 6.
    num_chirality_types : int, optional
        Number of chirality types, by default 7.
    num_bond_types : int, optional
        Number of bond types, by default 5.
    num_ring_sizes : int, optional
        Number of ring size classes, by default 4.
    """

    pair_dim: int
    atom_dim: int
    num_bins: int = 20
    num_elements: int = 6
    num_chirality_types: int = 7
    num_bond_types: int = 5
    num_ring_sizes: int = 4


class Emap2lig(LightningModule):
    def __init__(
        self,
        d_atom_in: int,
        d_pair_in: int,
        d_point_in: int,
        d_atom_hidden: int,
        d_pair_hidden: int,
        d_point_hidden: int,
        conf_embedder_args: ConformerEmbedderArgs | dict[str, Any],
        pairformer_args: PairformerArgs | dict[str, Any],
        instance_seg_args: InstanceSegArgs | dict[str, Any],
        diffusion_args: DiffusionArgs | dict[str, Any],
        auxiliary_args: AuxiliaryArgs | dict[str, Any],
        is_conf_embedder_compiled: bool = False,
        is_instance_seg_compiled: bool = False,
        is_pairformer_compiled: bool = False,
        is_auxiliary_module_compiled: bool = False,
        is_diffusion_module_compiled: bool = False,
        predict_args: DiffusionPredictArgs | dict[str, Any] | None = None,
        load_pretrained: bool = False,
        repo_id: str = "KiharaLab/Emap2lig",
        filename: str = "",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Convert dict/DictConfig args to dataclass if needed
        def to_dict(obj: Any) -> dict[str, Any]:
            """Convert DictConfig or dict to regular dict."""
            if isinstance(obj, DictConfig):
                return OmegaConf.to_container(obj, resolve=True)  # type: ignore
            return obj

        if isinstance(conf_embedder_args, (dict, DictConfig)):
            conf_embedder_args = ConformerEmbedderArgs(**to_dict(conf_embedder_args))
        if isinstance(pairformer_args, (dict, DictConfig)):
            pairformer_args = PairformerArgs(**to_dict(pairformer_args))
        if isinstance(instance_seg_args, (dict, DictConfig)):
            instance_seg_args = InstanceSegArgs(**to_dict(instance_seg_args))
        if isinstance(auxiliary_args, (dict, DictConfig)):
            auxiliary_args = AuxiliaryArgs(**to_dict(auxiliary_args))

        # Handle diffusion_args specially since it has nested dataclass
        if isinstance(diffusion_args, (dict, DictConfig)):
            diffusion_args = to_dict(diffusion_args)
            if "diffusion_model_args" in diffusion_args:
                dm_args = diffusion_args["diffusion_model_args"]
                if isinstance(dm_args, (dict, DictConfig)):
                    diffusion_args["diffusion_model_args"] = DiffusionModuleArgs(
                        **to_dict(dm_args)
                    )
            diffusion_args = DiffusionArgs(**diffusion_args)

        # Instance segmentation module (MUNet backbone + instance seg + feature extraction)
        self.instance_seg = InstanceSegModule(
            channels=instance_seg_args.channels,
            n_res_blocks=instance_seg_args.n_res_blocks,
            attention_levels=instance_seg_args.attention_levels,
            channel_multipliers=instance_seg_args.channel_multipliers,
            n_heads=instance_seg_args.n_heads,
            tf_layers=instance_seg_args.tf_layers,
            kernel_size=instance_seg_args.kernel_size,
            num_groups=instance_seg_args.num_groups,
            num_selected_points=instance_seg_args.num_selected_points,
        )
        self.is_instance_seg_compiled = is_instance_seg_compiled

        # Conformer embedder
        self.conf_embedder_args = conf_embedder_args
        self.conf_embedder = ConformerEmbedder(
            atom_dim_in=conf_embedder_args.atom_dim_in,
            pair_dim_in=conf_embedder_args.pair_dim_in,
            atom_dim=conf_embedder_args.atom_dim,
            pair_dim=conf_embedder_args.pair_dim,
        )
        self.is_conf_embedder_compiled = is_conf_embedder_compiled

        # Pairformer
        self.pairformer_args = pairformer_args
        self.pairformer = PairFormer(
            atom_dim=pairformer_args.atom_dim,
            pair_dim=pairformer_args.pair_dim,
            num_blocks=pairformer_args.num_blocks,
            num_heads=pairformer_args.num_heads,
            head_dim=pairformer_args.head_dim,
            transition_expansion_factor=pairformer_args.transition_expansion_factor,
            tri_attn_use_kernel=pairformer_args.tri_attn_use_kernel,
            tri_mul_use_kernel=pairformer_args.tri_mul_use_kernel,
            activation_checkpointing=pairformer_args.activation_checkpointing,
        )
        self.is_pairformer_compiled = is_pairformer_compiled

        # Auxiliary module
        self.auxiliary_args = auxiliary_args
        self.auxiliary_module = AuxiliaryModule(
            pair_dim=auxiliary_args.pair_dim,
            atom_dim=auxiliary_args.atom_dim,
            num_bins=auxiliary_args.num_bins,
            num_elements=auxiliary_args.num_elements,
            num_chirality_types=auxiliary_args.num_chirality_types,
            num_bond_types=auxiliary_args.num_bond_types,
            num_ring_sizes=auxiliary_args.num_ring_sizes,
        )
        self.is_auxiliary_module_compiled = is_auxiliary_module_compiled

        # Diffusion module
        self.diffusion_args = diffusion_args
        self.diffusion_module = AtomDiffusion(
            diffusion_model_args=diffusion_args.diffusion_model_args,
            sigma_min=diffusion_args.sigma_min,
            sigma_max=diffusion_args.sigma_max,
            sigma_data=diffusion_args.sigma_data,
            rho=diffusion_args.rho,
            P_mean=diffusion_args.P_mean,
            P_std=diffusion_args.P_std,
            gamma_0=diffusion_args.gamma_0,
            gamma_min=diffusion_args.gamma_min,
            noise_scale=diffusion_args.noise_scale,
            step_scale=diffusion_args.step_scale,
        )
        self.is_diffusion_module_compiled = is_diffusion_module_compiled

        # Handle predict args
        if predict_args is None:
            self.predict_args = DiffusionPredictArgs()
        elif isinstance(predict_args, (dict, DictConfig)):
            self.predict_args = DiffusionPredictArgs(**to_dict(predict_args))
        else:
            self.predict_args = predict_args

        # Load pretrained weights from HuggingFace Hub (or local cache)
        if load_pretrained and filename:
            local_path = Path(_MODEL_DIR) / filename
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=_MODEL_DIR
                )
            self.load_state_dict(load_file(model_path), strict=False)

        self.compile_model()

    def compile_model(self):
        if sys.platform == "darwin":
            logger.info("Disabling torch.compile on macOS/MPS inference")
            return

        if self.is_conf_embedder_compiled:
            self.conf_embedder.compile()
        if self.is_pairformer_compiled:
            self.pairformer.compile()
        if self.is_auxiliary_module_compiled:
            self.auxiliary_module.compile()
        if self.is_instance_seg_compiled:
            self.instance_seg.backbone.compile()
            self.instance_seg.seg_head.compile()

        if self.is_diffusion_module_compiled:
            self.diffusion_module.diffusion_model.compile()

    def forward(
        self,
        feats: dict[str, Tensor],
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        out_dict = {}

        # Conformer embedding: reference ligand → atom + pair representations
        (
            atom_init_feats,
            pair_init_feats,
        ) = self.conf_embedder(
            feats["ref_pos"],
            feats["atom_feature"],
            feats["bond_feature"],
            feats["atom_mask"],
            feats["pair_mask"],
        )

        atom_feats = atom_init_feats
        pair_feats = pair_init_feats

        atom_feats, pair_feats = self.pairformer(
            atom_feats,
            pair_feats,
            feats["atom_mask"],
            feats["pair_mask"],
        )

        auxiliary_outputs = self.auxiliary_module.forward(
            pair_feats=pair_feats,
            pair_mask=feats["pair_mask"],
            atom_feats=atom_feats,
            atom_mask=feats["atom_mask"],
        )
        # Update output dictionary with all auxiliary module outputs
        out_dict.update(auxiliary_outputs)

        # Reshape prompt_points from [B, multiplicity, 3] to [B*multiplicity, 3]
        prompt_points = feats["prompt_points"]
        if prompt_points.dim() == 3:
            # [B, multiplicity, 3] -> [B*multiplicity, 3]
            prompt_points = prompt_points.reshape(-1, 3)

        # Instance segmentation + voxel feature extraction
        (
            augment_output,
            instance_output,
            voxel_features,
            global_features,
            selected_point_feats,
            selected_point_coords,
        ) = self.instance_seg.forward_embedding(
            feats["input_density"],
            feats["global_origin"],
            feats["voxel_size"],
            atom_features=atom_feats,
            atom_mask=feats["atom_mask"],
            prompt_point=prompt_points,
            multiplicity=multiplicity,
        )
        out_dict["augment_output"] = augment_output
        out_dict["instance_mask_output"] = instance_output
        out_dict["voxel_features"] = voxel_features
        out_dict["global_features"] = global_features

        # Run diffusion module (sampling)
        sampled_atom_coords = self.diffusion_module.sample(
            atom_inputs=atom_init_feats,
            atom_feats=atom_feats,
            pair_feats=pair_feats,
            selected_point_feats=selected_point_feats,
            selected_point_coords=selected_point_coords,
            global_features=global_features,
            atom_mask=feats["atom_mask"],
            ref_pos=feats["ref_pos"],
            prompt_points=prompt_points,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity,
        )
        out_dict["sampled_atom_coords"] = sampled_atom_coords

        return out_dict

    def predict_step(self, batch, batch_idx):
        # identifier / class_name are lists (one per item in batch)
        identifiers = batch.get("identifier", batch.get("class_name", ["unknown"]))
        label = identifiers[0] if isinstance(identifiers, list) else identifiers

        # Show progress like "Predicting 3/59 batches — ligand ATP"
        total = "?"
        try:
            if self.trainer and self.trainer.num_predict_batches:
                total = self.trainer.num_predict_batches[0]
        except Exception:
            pass
        logger.info(f"Predicting {batch_idx + 1}/{total} batches — ligand {label}")

        try:
            # Use multiplicity from predict_args (set by main.py from CLI)
            out_dict = self.forward(
                batch,
                num_sampling_steps=self.predict_args.num_sampling_steps,
                multiplicity=self.predict_args.multiplicity,
            )
        except Exception as e:
            import traceback

            logger.error(f"Failed to predict batch {batch_idx + 1}/{total}: {e}")
            logger.error(traceback.format_exc())
            return None

        return out_dict
