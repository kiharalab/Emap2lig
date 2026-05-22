import math
from dataclasses import dataclass

import torch
from torch.nn import Module, functional as F

from ..layers import AtomDecoder
from .conditioning import AtomConditioner, PointConditioner

from loguru import logger


@dataclass
class DiffusionModuleArgs:
    """Arguments for the diffusion module (DiffusionModule).

    Parameters
    ----------
    atom_dim : int
        The atom single representation dimension.
    pair_dim : int
        The atom pair representation dimension.
    point_dim : int
        The sampled point feature dimension from voxel projector.
    aggregation_dim : int, optional
        The aggregation dimension for point features, by default 96.
    fourier_dim : int, optional
        The dimension of the fourier embedding, by default 256.
    conditioning_transition_layers : int, optional
        The number of transition layers for conditioning, by default 2.
    num_point_blocks : int, optional
        The number of conditioning blocks, by default 4.
    num_heads : int, optional
        The number of heads in attention, by default 3.
    head_dim : int, optional
        The dimension of each head, by default 16.
    """

    atom_dim: int
    pair_dim: int
    point_dim: int
    aggregation_dim: int = 96
    fourier_dim: int = 256
    conditioning_transition_layers: int = 2
    num_point_blocks: int = 4
    num_heads: int = 3
    head_dim: int = 16
    activation_checkpointing: bool = False


class DiffusionModule(Module):
    """EDM-style network predicting coordinate updates.

    Combines atom conditioning, point-conditioning (sparse point-to-atom attention
    against pre-selected voxel features), and a coordinate decoder.
    """

    def __init__(
        self,
        atom_dim: int,
        pair_dim: int,
        point_dim: int,
        aggregation_dim: int = 96,
        fourier_dim: int = 256,
        conditioning_transition_layers: int = 2,
        num_point_blocks: int = 4,
        num_heads: int = 3,
        head_dim: int = 16,
        activation_checkpointing: bool = False,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        atom_dim : int
            The atom single representation dimension.
        pair_dim : int
            The atom pair representation dimension.
        point_dim : int
            The point cloud representation dimension.
        aggregation_dim : int, optional
            The aggregation dimension for point features, by default 96.
        fourier_dim : int, optional
            The dimension of the fourier embedding, by default 256.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        num_point_blocks : int, optional
            The number of conditioning blocks, by default 4.
        num_heads : int, optional
            The number of heads in attention, by default 3.
        head_dim : int, optional
            The dimension of each head, by default 16.
        """
        super().__init__()

        self.atom_conditioner = AtomConditioner(
            atom_dim=atom_dim,
            fourier_dim=fourier_dim,
            num_transitions=conditioning_transition_layers,
            activation_checkpointing=activation_checkpointing,
        )

        # Unified point conditioner operating on pre-selected voxel points
        self.point_conditioner = PointConditioner(
            atom_dim=atom_dim,
            pair_dim=pair_dim,
            point_dim=point_dim,
            aggregation_dim=aggregation_dim,
            num_blocks=num_point_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            num_transitions=conditioning_transition_layers,
            use_global_feats=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.decoder = AtomDecoder(atom_dim)

    def forward(
        self,
        atom_init_feats: torch.Tensor,  # [B, N_a, C_a]
        atom_feats: torch.Tensor,  # [B, N_a, C_a]
        atom_mask: torch.Tensor,  # [B, N_a]
        pair_feats: torch.Tensor,  # [B, N_a, N_a, C_p]
        selected_point_feats: torch.Tensor,  # [B, num_points, C_v]
        selected_point_coords: torch.Tensor,  # [B, num_points, 3]
        global_features: torch.Tensor | None,  # [B, aggregation_dim]
        r_noisy: torch.Tensor,  # [B, N_a, 3]
        sigma: torch.Tensor,  # [B] or scalar
        prompt_point: torch.Tensor,  # [B, 3]
    ) -> torch.Tensor:
        """Forward pass for the diffusion module.

        Parameters
        ----------
        atom_init_feats : torch.Tensor
            The atom inputs, shape [B, N_a, C_a]
        atom_feats : torch.Tensor
            The atom features, shape [B, N_a, C_a]
        atom_mask : torch.Tensor
            The atom mask, shape [B, N_a]
        pair_feats : torch.Tensor
            The pair features, shape [B, N_a, N_a, C_p]
        selected_point_feats : torch.Tensor
            Pre-selected point features from EM embedder, shape [B, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates from EM embedder, shape [B, num_points, 3]
        global_features : torch.Tensor | None
            The global features, shape [B, aggregation_dim]
        r_noisy : torch.Tensor
            The noisy atom coordinates, shape [B, N_a, 3]
        sigma : torch.Tensor
            The noise level, shape [B] or scalar
        prompt_point : torch.Tensor
            The prompt point for relative position encoding, shape [B, 3]

        Returns
        -------
        torch.Tensor
            The coordinate update (denoised coordinates), shape [B, N_a, 3]
        """
        atom_mask = atom_mask.bool()

        # Convert sigma to proper format for conditioning
        if sigma.dim() == 0:  # scalar
            batch_size = atom_feats.shape[0]
            sigma = sigma.expand(batch_size)
        elif sigma.dim() == 1 and sigma.shape[0] == 1:
            batch_size = atom_feats.shape[0]
            sigma = sigma.expand(batch_size)

        # Create time conditioning from sigma (EDM approach)
        times = torch.log(sigma) / 4.0  # Scale sigma to reasonable range

        atom_feats_conditioned = self.atom_conditioner(
            times=times,
            atom_feats=atom_feats,
            atom_init_feats=atom_init_feats,
            atom_coords=r_noisy,
            atom_mask=atom_mask,
        )

        # Apply unified point conditioning with pre-selected points
        atom_feats_conditioned = self.point_conditioner(
            atom_feats=atom_feats_conditioned,
            pair_feats=pair_feats,
            selected_point_feats=selected_point_feats,
            selected_point_coords=selected_point_coords,
            atom_coords=r_noisy,
            atom_mask=atom_mask,
            prompt_point=prompt_point,
            global_features=global_features,
        )

        coordinate_update = self.decoder(atom_feats_conditioned)

        return coordinate_update


class AtomDiffusion(Module):
    """Full EDM sampling and training loop for ligand coordinates."""

    def __init__(
        self,
        diffusion_model_args: DiffusionModuleArgs,
        sigma_min: float = 0.0004,
        sigma_max: float = 24.0,
        sigma_data: float = 8.0,
        rho: float = 7.0,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
    ) -> None:
        """Initialize the atom diffusion module.

        Parameters
        ----------
        diffusion_model_args : DiffusionModelArgs
            The arguments for the diffusion model.
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
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        """
        super().__init__()

        self.diffusion_model = DiffusionModule(
            atom_dim=diffusion_model_args.atom_dim,
            pair_dim=diffusion_model_args.pair_dim,
            point_dim=diffusion_model_args.point_dim,
            aggregation_dim=diffusion_model_args.aggregation_dim,
            fourier_dim=diffusion_model_args.fourier_dim,
            conditioning_transition_layers=diffusion_model_args.conditioning_transition_layers,
            num_point_blocks=diffusion_model_args.num_point_blocks,
            num_heads=diffusion_model_args.num_heads,
            head_dim=diffusion_model_args.head_dim,
            activation_checkpointing=diffusion_model_args.activation_checkpointing,
        )

        # EDM noise scheduling parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.diffusion_model.parameters()).device

    def sample_schedule(self, num_steps: int) -> torch.Tensor:
        """Generate noise levels for sampling steps using EDM scheduling.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps.

        Returns
        -------
        torch.Tensor
            Noise levels (sigmas), shape [num_steps + 1]
        """
        # EDM noise schedule: sigma(i) in log-space interpolated with power 1/rho
        inv_rho = 1 / self.rho
        steps = torch.arange(num_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data  # [num_sampling_steps]
        sigmas = F.pad(
            sigmas, (0, 1), value=0.0
        )  # last step is sigma value of 0. [num_sampling_steps + 1]
        return sigmas

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute c_skip preconditioning coefficient."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute c_out preconditioning coefficient."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute c_in preconditioning coefficient."""
        return 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def preconditioned_network_forward(
        self,
        atom_init_feats: torch.Tensor,
        atom_feats: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_feats: torch.Tensor,
        selected_point_feats: torch.Tensor,
        selected_point_coords: torch.Tensor,
        global_features: torch.Tensor,
        r_noisy: torch.Tensor,
        sigma: torch.Tensor,
        prompt_point: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through preconditioned network.

        Parameters
        ----------
        atom_init_feats : torch.Tensor
            Initial atom features, shape [B, N_a, C_a]
        atom_feats : torch.Tensor
            Processed atom features, shape [B, N_a, C_a]
        atom_mask : torch.Tensor
            Atom mask, shape [B, N_a]
        pair_feats : torch.Tensor
            Pair features, shape [B, N_a, N_a, C_p]
        selected_point_feats : torch.Tensor
            Pre-selected point features, shape [B, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates, shape [B, num_points, 3]
        global_features : torch.Tensor
            Global features, shape [B, aggregation_dim]
        r_noisy : torch.Tensor
            Noisy coordinates, shape [B, N_a, 3]
        sigma : torch.Tensor
            Noise level, shape [B] or scalar
        prompt_point : torch.Tensor
            Prompt point, shape [B, 3]

        Returns
        -------
        torch.Tensor
            Denoised coordinates, shape [B, N_a, 3]
        """
        # Apply input preconditioning
        c_in_val = self.c_in(sigma)
        if c_in_val.dim() == 1:
            c_in_val = c_in_val.view(-1, 1, 1)

        r_scaled = c_in_val * r_noisy

        # Get network prediction
        F_theta = self.diffusion_model(
            atom_init_feats=atom_init_feats,
            atom_feats=atom_feats,
            atom_mask=atom_mask,
            pair_feats=pair_feats,
            selected_point_feats=selected_point_feats,
            selected_point_coords=selected_point_coords,
            global_features=global_features,
            r_noisy=r_scaled,
            sigma=sigma,
            prompt_point=prompt_point,
        )

        # Apply output preconditioning
        c_skip_val = self.c_skip(sigma)
        c_out_val = self.c_out(sigma)

        if c_skip_val.dim() == 1:
            c_skip_val = c_skip_val.view(-1, 1, 1)
        if c_out_val.dim() == 1:
            c_out_val = c_out_val.view(-1, 1, 1)

        r_denoised = c_skip_val * r_noisy + c_out_val * F_theta

        return r_denoised

    @torch.inference_mode()
    def sample(
        self,
        atom_inputs: torch.Tensor,
        atom_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        selected_point_feats: torch.Tensor,
        selected_point_coords: torch.Tensor,
        global_features: torch.Tensor | None,
        atom_mask: torch.Tensor,
        ref_pos: torch.Tensor,
        prompt_points: torch.Tensor,
        num_sampling_steps: int,
        multiplicity: int = 1,
    ) -> torch.Tensor:
        """Sample from the diffusion model using EDM sampling.

        Parameters
        ----------
        atom_inputs : torch.Tensor
            The atom inputs, shape [B, N_a, C_a]
        atom_feats : torch.Tensor
            The atom features, shape [B, N_a, C_a]
        pair_feats : torch.Tensor
            The pair features, shape [B, N_a, N_a, C_p]
        selected_point_feats : torch.Tensor
            Pre-selected point features from EM embedder, shape [B*multiplicity, num_points, C_v]
        selected_point_coords : torch.Tensor
            Pre-selected point coordinates from EM embedder, shape [B*multiplicity, num_points, 3]
        global_features : torch.Tensor | None
            The global features, shape [B*multiplicity, aggregation_dim]
        atom_mask : torch.Tensor
            The atom mask, shape [B, N_a]
        ref_pos : torch.Tensor
            The reference positions, shape [B, N_a, 3]
        prompt_points : torch.Tensor
            Multiple prompt points for different multiplicities, shape [B*multiplicity, 3].
            Each sample in the batch will use its corresponding prompt_point for conditioning.
        num_sampling_steps : int
            The number of sampling steps.
        multiplicity : int, optional
            The multiplicity, by default 1.

        Returns
        -------
        torch.Tensor
            The sampled atom coordinates, shape [B*multiplicity, N_a, 3]
        """
        # Get original shapes first
        batch_size, _, _ = atom_inputs.shape

        # Repeat inputs for multiplicity (except tensors which are pre-expanded by EM embedder)
        atom_inputs = atom_inputs.repeat_interleave(multiplicity, 0)
        atom_feats = atom_feats.repeat_interleave(multiplicity, 0)
        pair_feats = pair_feats.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        ref_pos = ref_pos.repeat_interleave(multiplicity, 0)
        # Note: selected_point_feats, selected_point_coords, global_features
        # are already expanded by EM embedder to [B*multiplicity, ...]

        # `prompt_points` is already [B*multiplicity, 3]
        prompt_point = prompt_points

        # Update batch size for multiplicity
        batch_size = batch_size * multiplicity

        # Generate noise schedule
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.gamma_min, self.gamma_0, 0.0
        )  # [num_sampling_steps + 1]
        sigmas_and_gammas = list(
            zip(sigmas[:-1], sigmas[1:], gammas[1:])
        )  # [(sigma_0, sigma_1, gamma_1), (sigma_1, sigma_2, gamma_2), ..., (sigma_N-1, sigma_N, gamma_N)]

        # Initialize with noise scaled by largest sigma
        init_sigma = sigmas[0]
        atom_coords = torch.randn_like(ref_pos) * init_sigma

        logger.debug(f"EDM diffusion sampling with {num_sampling_steps} steps")

        # Iterative denoising loop
        for sigma_tm, sigma_t, gamma in sigmas_and_gammas:
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)  # [1]
            eps = (
                self.noise_scale
                * math.sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn_like(ref_pos)  # [B, 3]
            )
            atom_coords_noisy = atom_coords + eps  # [B, N, 3]
            t_hat_expanded = torch.full((batch_size,), t_hat, device=self.device)

            # Denoise using preconditioned network
            atom_coords_denoised = self.preconditioned_network_forward(
                atom_init_feats=atom_inputs,
                atom_feats=atom_feats,
                atom_mask=atom_mask,
                pair_feats=pair_feats,
                selected_point_feats=selected_point_feats,
                selected_point_coords=selected_point_coords,
                global_features=global_features,
                r_noisy=atom_coords_noisy,
                sigma=t_hat_expanded,
                prompt_point=prompt_point,
            )

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return atom_coords
