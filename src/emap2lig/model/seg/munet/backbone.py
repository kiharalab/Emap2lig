import torch
from torch import nn

from .conv import ConvBlock3d, DownSample3d, UpSample3d
from .transformer import SpatialTransformerBlock3d


class MUNetBackbone(nn.Module):
    """
    Modern U-Net backbone with transformers.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: list[int],
        channel_multipliers: list[int],
        n_heads: int,
        tf_layers: int,
        kernel_size: int,
        num_groups: int,
    ):
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(
            ConvBlock3d(
                in_channels, out_channels=channels - 1, kernel_size=3, num_groups=1
            )
        )
        encoder_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i in range(levels):
            for _ in range(n_res_blocks):
                layers: list[nn.Module] = [
                    ConvBlock3d(
                        channels, out_channels=channels_list[i], num_groups=num_groups
                    )
                ]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(
                            channels, n_heads, tf_layers, num_groups=num_groups
                        )
                    )
                else:
                    layers.append(
                        ConvBlock3d(
                            channels, kernel_size=kernel_size, num_groups=num_groups
                        )
                    )
                # Add them to the input half of the U-Net and keep track of the number of channels of its output
                self.encoder_blocks.append(nn.Sequential(*layers))
                encoder_block_channels.append(channels)

            # Down sample at all levels except last
            if i != levels - 1:
                self.encoder_blocks.append(DownSample3d(channels))
                encoder_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = nn.Sequential(
            SpatialTransformerBlock3d(
                channels, n_heads, tf_layers, num_groups=num_groups
            ),
        )

        # Second half of the U-Net
        self.decoder_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [
                    ConvBlock3d(
                        channels + encoder_block_channels.pop(),
                        out_channels=channels_list[i],
                        kernel_size=3,
                        num_groups=num_groups,
                    )
                ]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(
                        SpatialTransformerBlock3d(
                            channels, n_heads, tf_layers, num_groups=num_groups
                        )
                    )
                else:
                    layers.append(
                        ConvBlock3d(
                            channels, kernel_size=kernel_size, num_groups=num_groups
                        )
                    )
                if i != 0 and j == n_res_blocks:
                    layers.append(UpSample3d(channels))
                self.decoder_blocks.append(nn.Sequential(*layers))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, x_original: torch.Tensor | None = None):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height, depth]`
        """
        if x_original is None:
            x_original = x
        x_encoder_block = []

        # Input half of the U-Net
        for idx, input_module in enumerate(self.encoder_blocks):
            x = input_module(x)
            if idx == 0:
                x = torch.cat((x, x_original), dim=1)
            x_encoder_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x)
        # Output half of the U-Net
        for _i, decoder_module in enumerate(self.decoder_blocks):
            # dim=1 is the channel dimension
            x = torch.cat([x, x_encoder_block.pop()], dim=1)
            x = decoder_module(x)

        return x
