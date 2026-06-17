from pathlib import Path

import torch
from torch import nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .munet import MUNetBackbone
from .threshold import threshold_li

from loguru import logger

_MODEL_DIR = str(Path.home() / ".emap2lig" / "models")


def tensor_to_point_cloud(
    tensor: torch.Tensor,
    instance_mask: torch.Tensor,
    voxel_size: torch.Tensor,
    global_origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a 5D tensor of shape [B, C, D, H, W] into a point cloud with coordinates for each batch.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        voxel_size (torch.Tensor): Tensor of shape [B, 3] for voxel scaling per batch.
        global_origin (torch.Tensor): Tensor of shape [B, 3] for the global origin per batch.

    Returns:
        point_cloud (torch.Tensor): Tensor of shape [B, N, C], where N = D * H * W.
        coordinates (torch.Tensor): Tensor of shape [B, N, 3] for the coordinates of each point.
    """
    # Assert that the tensor is a 5D tensor
    assert tensor.ndim == 5, "Tensor must have 5 dimensions"
    assert voxel_size.ndim == 2, "Voxel size must have 2 dimensions"
    assert global_origin.ndim == 2, "Global origin must have 2 dimensions"
    assert voxel_size.shape[1] == 3, "Voxel size must have 3 dimensions"
    assert global_origin.shape[1] == 3, "Global origin must have 3 dimensions"

    B, C, D, H, W = tensor.shape
    N = D * H * W

    # Create a grid of indices for the spatial dimensions.
    # Using 'indexing="ij"' ensures that the first dimension corresponds to D (z), second to H (y), and third to W (x).
    z, y, x = torch.meshgrid(
        torch.arange(D, device=tensor.device),
        torch.arange(H, device=tensor.device),
        torch.arange(W, device=tensor.device),
        indexing="ij",  # available in PyTorch 1.10+; remove for older versions.
    )

    # Flatten the indices and stack them in the order (x, y, z)
    indices = torch.stack([x, y, z], dim=-1)  # shape: [N, 3]

    # Compute coordinates for each batch using broadcasting:
    # - indices: [N, 3] -> [1, N, 3]
    # - voxel_size: [B, 3] -> [B, 1, 3]
    # - global_origin: [B, 3] -> [B, 1, 3]
    coordinates = indices.unsqueeze(0) * voxel_size.unsqueeze(
        1
    ) + global_origin.unsqueeze(1)  # shape: [B, N, 3]

    # Reshape tensor from [B, C, D, H, W] to [B, N, C]
    point_cloud = tensor.reshape(B, C, N).permute(0, 2, 1).contiguous()
    # Reshape instance_mask from [B, D, H, W] to [B, N]
    point_mask = instance_mask.reshape(B, N).contiguous()

    return point_cloud, point_mask, coordinates


def _axis_starts(dim: int, spatial_size: int, stride: int) -> list[int]:
    """Generate start positions for one axis, ensuring coverage to the end."""
    if dim <= spatial_size:
        return [0]
    starts = list(range(0, dim - spatial_size + 1, stride))
    if starts[-1] != dim - spatial_size:
        starts.append(dim - spatial_size)
    return starts


def scan_voxel_positions(z_dim, y_dim, x_dim, spatial_size, stride):
    """
    Generate all valid voxel start and end positions as an iterator.

    Args:
        z_dim, y_dim, x_dim (int): Dimensions of the volume
        spatial_size (int): Size of each voxel
        stride (int): Stride for voxel generation

    Yields:
        tuple: (z_start, y_start, x_start, z_end, y_end, x_end)
    """
    z_starts = _axis_starts(z_dim, spatial_size, stride)
    y_starts = _axis_starts(y_dim, spatial_size, stride)
    x_starts = _axis_starts(x_dim, spatial_size, stride)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                yield (
                    z0,
                    y0,
                    x0,
                    z0 + spatial_size,
                    y0 + spatial_size,
                    x0 + spatial_size,
                )


class VolumeInferenceDataset(Dataset):
    def __init__(self, input_map, coords):
        self.input_map = input_map[0]
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        z_start, y_start, x_start, z_end, y_end, x_end = self.coords[idx]
        sub_volume = self.input_map[:, z_start:z_end, y_start:y_end, x_start:x_end]
        return {"volume": sub_volume, "coords": self.coords[idx]}


def custom_collate(batch):
    volumes = torch.stack([item["volume"] for item in batch])
    coords = [item["coords"] for item in batch]  # Keep as list of tuples for slicing
    return volumes, coords


class SegHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_groups: int = 32
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class RegHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.Conv3d(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class MUNetRegSeg(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 32,
        reg_channels: int = 6,
        seg_channels: int = 6,
        spatial_size: int = 64,
        n_res_blocks: int = 1,
        attention_levels: list[int] | None = None,
        channel_multipliers: list[int] | None = None,
        n_heads: int = 4,
        tf_layers: int = 1,
        kernel_size: int = 5,
        num_groups: int = 32,
        load_pretrained: bool = False,
        repo_id: str = "KiharaLab/Emap2lig",
        filename: str = "emap2lig-find-v0.0.1.safetensors",
    ) -> None:
        super().__init__()

        # Set defaults for mutable arguments
        if attention_levels is None:
            attention_levels = [2]
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]

        self.backbone = MUNetBackbone(
            in_channels,
            mid_channels,
            n_res_blocks,
            attention_levels,
            channel_multipliers,
            n_heads,
            tf_layers,
            kernel_size,
            num_groups,
        )

        self.reg_head = RegHead(mid_channels, reg_channels)
        self.seg_head = SegHead(mid_channels, seg_channels)

        self.reg_act = torch.relu
        self.seg_act = torch.sigmoid

        self.num_labels = seg_channels
        self.spatial_size = spatial_size
        self.labels = ["backbone", "sidechain", "sugar", "phosphate", "base", "ligand"]

        if load_pretrained:
            local_path = Path(_MODEL_DIR) / filename
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=_MODEL_DIR
                )
            self.load_state_dict(load_file(model_path))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mid_emb = self.backbone(x)
        reg_emb = self.reg_head(mid_emb)
        seg_emb = self.seg_head(mid_emb)

        # if training, return both reg and seg
        if self.training:
            return self.reg_act(reg_emb), self.seg_act(seg_emb)
        # if testing, return only seg
        else:
            return self.seg_act(seg_emb)

    def _binarize_output(
        self, output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, C, _, _, _ = output.shape
        reshaped_output = output.reshape(B * C, -1)
        thresholds = torch.stack(
            [threshold_li(slice, initial_guess=0.1) for slice in reshaped_output]
        )
        thresholds = thresholds.reshape(B, C, 1, 1, 1)
        binary_tensor = output > thresholds

        channel_thresholds: dict[str, float] = {}
        for c in range(C):
            label = self.labels[c] if c < len(self.labels) else f"channel_{c}"
            channel_thresholds[label] = thresholds[0, c, 0, 0, 0].item()

        return binary_tensor, channel_thresholds

    @torch.no_grad()
    def sliding_window_inference(
        self,
        input_map: torch.Tensor,
        roi_size: int = 64,
        batch_size: int = 4,
        device=torch.device("cuda"),
        output_device=torch.device("cpu"),
        overlap_ratio: float = 0.5,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """
        Perform sliding window inference on a large volume using this model.

        Args:
            input_map (torch.Tensor): Input tensor of shape [1, 1, D, H, W].
            roi_size (int): Size of the region of interest (cube).
            batch_size (int): Batch size for inference.
            device (torch.device): Device to perform computation on.
            output_device (torch.device): Device to store output on.
            overlap_ratio (float): Overlap ratio between adjacent windows.
            num_workers (int): Number of workers for data loading.

        Returns:
            torch.Tensor: Segmentation map of shape [1, C, D, H, W].
        """
        self.eval()
        self.to(device)

        output_num_channels = self.num_labels

        H, W, D = input_map.shape[2:]
        logger.info(f"Input tensor shape: H:{H}/W:{W}/D:{D}")
        if H < roi_size or W < roi_size or D < roi_size:
            logger.warning(
                f"Input tensor spatial dimensions {H}/{W}/{D} are smaller than the ROI size {roi_size}."
            )

            pad_H = max(roi_size - H, 0)
            pad_W = max(roi_size - W, 0)
            pad_D = max(roi_size - D, 0)
            input_map = torch.nn.functional.pad(
                input_map, (0, pad_D, 0, pad_W, 0, pad_H), mode="constant", value=0
            )
            H, W, D = input_map.shape[2:]
            logger.info(f"After padding, input tensor shape: H:{H}/W:{W}/D:{D}")

        output_map = torch.zeros(
            (1, output_num_channels, H, W, D), device=output_device, dtype=torch.float
        )
        count_map = torch.zeros(
            (1, output_num_channels, H, W, D), device=output_device, dtype=torch.float
        )
        stride = int(roi_size * (1 - overlap_ratio))

        coords = []
        num_voxels = 0
        for position in scan_voxel_positions(H, W, D, roi_size, stride):
            num_voxels += 1
            z_start, y_start, x_start, z_end, y_end, x_end = position
            sub_volume = input_map[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]
            if not torch.all(sub_volume == 0):
                coords.append(position)

        logger.info(
            f"Number of non-empty ROIs / All possible ROIs: {len(coords)} / {num_voxels}"
        )

        dataset = VolumeInferenceDataset(input_map, coords)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=custom_collate,
        )

        for batch, batch_coords in tqdm(dataloader, desc="Processing inference"):
            batch = batch.to(device)
            model_output = self(batch)
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            model_output = model_output.to(output_device)

            for idx, (z_start, y_start, x_start, z_end, y_end, x_end) in enumerate(
                batch_coords
            ):
                output_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += (
                    model_output[idx]
                )
                count_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1

        output_map[count_map > 0] /= count_map[count_map > 0]

        return output_map


class FragmentRegSeg(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        reg_channels: int,
        seg_channels: int,
        spatial_size: int,
        n_res_blocks: int,
        attention_levels: list[int],
        channel_multipliers: list[int],
        n_heads: int,
        tf_layers: int,
        kernel_size: int,
        num_groups: int = 32,
        load_pretrained: bool = False,
        repo_id: str = "KiharaLab/Emap2lig",
        filename: str = "emap2lig-frag.safetensors",
    ) -> None:
        super().__init__()
        self.labels = ["5R", "6R", "4H", "5H", "DR", "HEM-like"]
        self.mid_channels = mid_channels

        self.backbone1 = MUNetBackbone(
            in_channels,
            mid_channels,
            n_res_blocks,
            attention_levels,
            channel_multipliers,
            n_heads,
            tf_layers,
            kernel_size,
            num_groups,
        )

        self.backbone2 = MUNetBackbone(
            self.mid_channels,
            self.mid_channels,
            n_res_blocks,
            attention_levels,
            channel_multipliers,
            n_heads,
            tf_layers,
            kernel_size,
            num_groups,
        )

        self.reg_head = RegHead(self.mid_channels, reg_channels)
        self.seg_head = SegHead(self.mid_channels, seg_channels)

        self.reg_act = torch.relu
        self.seg_act = torch.sigmoid

        self.num_labels = seg_channels
        self.spatial_size = spatial_size

        if load_pretrained:
            local_path = Path(_MODEL_DIR) / filename
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=_MODEL_DIR
                )
            self.load_state_dict(load_file(model_path))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mid_emb = self.backbone1(x)
        mid_emb = self.backbone2(mid_emb, x)
        reg_emb = self.reg_head(mid_emb)
        seg_emb = self.seg_head(mid_emb)

        if self.training:
            return self.reg_act(reg_emb), self.seg_act(seg_emb)
        else:
            return self.seg_act(seg_emb)

    def _binarize_output(
        self, output: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, C, _, _, _ = output.shape
        reshaped_output = output.reshape(B * C, -1)
        thresholds = torch.stack(
            [threshold_li(slice, initial_guess=0.1) for slice in reshaped_output]
        )
        thresholds = thresholds.reshape(B, C, 1, 1, 1)
        binary_tensor = output > thresholds

        channel_thresholds: dict[str, float] = {}
        for c in range(C):
            label = self.labels[c] if c < len(self.labels) else f"channel_{c}"
            channel_thresholds[label] = thresholds[0, c, 0, 0, 0].item()

        return binary_tensor, channel_thresholds

    @torch.no_grad()
    def sliding_window_inference(
        self,
        input_map: torch.Tensor,
        roi_size: int = 64,
        batch_size: int = 4,
        device=torch.device("cuda"),
        output_device=torch.device("cpu"),
        overlap_ratio: float = 0.5,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """
        Perform sliding window inference on a large volume using this model.

        Args:
            input_map (torch.Tensor): Input tensor of shape [1, 1, D, H, W].
            roi_size (int): Size of the region of interest (cube).
            batch_size (int): Batch size for inference.
            device (torch.device): Device to perform computation on.
            output_device (torch.device): Device to store output on.
            overlap_ratio (float): Overlap ratio between adjacent windows.
            num_workers (int): Number of workers for data loading.

        Returns:
            torch.Tensor: Segmentation map of shape [1, C, D, H, W].
        """
        self.eval()
        self.to(device)

        output_num_channels = self.num_labels

        H, W, D = input_map.shape[2:]
        logger.info(f"Input tensor shape: H:{H}/W:{W}/D:{D}")
        if H < roi_size or W < roi_size or D < roi_size:
            logger.warning(
                f"Input tensor spatial dimensions {H}/{W}/{D} are smaller than the ROI size {roi_size}."
            )

            pad_H = max(roi_size - H, 0)
            pad_W = max(roi_size - W, 0)
            pad_D = max(roi_size - D, 0)
            input_map = torch.nn.functional.pad(
                input_map, (0, pad_D, 0, pad_W, 0, pad_H), mode="constant", value=0
            )
            H, W, D = input_map.shape[2:]
            logger.info(f"After padding, input tensor shape: H:{H}/W:{W}/D:{D}")

        output_map = torch.zeros(
            (1, output_num_channels, H, W, D), device=output_device, dtype=torch.float
        )
        count_map = torch.zeros(
            (1, output_num_channels, H, W, D), device=output_device, dtype=torch.float
        )
        stride = int(roi_size * (1 - overlap_ratio))

        coords = []
        num_voxels = 0
        for position in scan_voxel_positions(H, W, D, roi_size, stride):
            num_voxels += 1
            z_start, y_start, x_start, z_end, y_end, x_end = position
            sub_volume = input_map[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]
            if not torch.all(sub_volume == 0):
                coords.append(position)

        logger.info(
            f"Number of non-empty ROIs / All possible ROIs: {len(coords)} / {num_voxels}"
        )

        dataset = VolumeInferenceDataset(input_map, coords)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=custom_collate,
        )

        for batch, batch_coords in tqdm(dataloader, desc="Processing inference"):
            batch = batch.to(device)
            model_output = self(batch)
            if isinstance(model_output, tuple):
                model_output = model_output[0]

            model_output = model_output.to(output_device)

            for idx, (z_start, y_start, x_start, z_end, y_end, x_end) in enumerate(
                batch_coords
            ):
                output_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += (
                    model_output[idx]
                )
                count_map[:, :, z_start:z_end, y_start:y_end, x_start:x_end] += 1

        output_map[count_map > 0] /= count_map[count_map > 0]

        return output_map
