import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3d(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(
        self, channels: int, out_channels=None, kernel_size=3, num_groups: int = 32
    ):
        """
        :param channels: the number of input channels
        :param out_channels: is the number of out channels. defaults to `channels`.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.norm1 = nn.GroupNorm(num_groups, channels, eps=1e-6)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv3d(channels, out_channels, 3, padding=1)

        # Final convolution layer
        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=1e-6)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=out_channels,
        )

        # `channels` to `out_channels` mapping layer for residual connection
        self.skip_connection: nn.Module
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Initial convolution
        x = self.norm1(x)
        h = self.conv1(self.activation1(x))
        # Final convolution
        h = self.norm2(h)
        h = self.conv2(self.dropout(self.activation2(h)))
        # Add skip connection
        return self.skip_connection(x) + h


class UpSample3d(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(
        self,
        channels: int,
        output_channels: int | None = None,
        scale_factor: int = 2,
    ):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        output_channels = output_channels or channels
        # Apply a convolution to refine the upsampled features
        self.conv = nn.Conv3d(channels, output_channels, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, depth, height, width]`
        """
        # Upsample and then apply convolution
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear")
        return self.conv(x)


class DownSample3d(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width, depth]`
        """
        # Apply convolution
        return self.conv(x)
