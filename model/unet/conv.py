from typing import Optional

from torch.nn import Conv3d, BatchNorm3d, GroupNorm, Dropout3d, PReLU
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            normalization: Optional[str] = None,
            kernel_size: int = 5,
            padding: int = 0,
            padding_mode: str = 'zeros',
            dilation: Optional[int] = None,
            activation: Optional[int] = "PReLU",  # the last layer don't need this
            dropout: float = 0.3,
            ):
        super().__init__()

        block = nn.ModuleList()

        # dilation = 1 if dilation is None else dilation
        # if padding:
        #     total_padding = kernel_size + 2 * (dilation - 1) - 1
        #     padding = total_padding // 2

        conv_layer = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            # dilation=dilation,
        )

        norm_layer = None
        if normalization == 'Batch':
            # num_features = in_channels if preactivation else out_channels
            norm_layer = BatchNorm3d(out_channels)
        elif normalization == 'Group':
            # num_features = in_channels if preactivation else out_channels
            norm_layer = GroupNorm(num_groups=1, num_channels=out_channels)

        block.append(conv_layer)
        if norm_layer is not None:
            block.append(norm_layer)
        if activation is not None:
            activation_layer = PReLU()
            block.append(activation_layer)

        if dropout:
            dropout_layer = Dropout3d(p=dropout)
            block.append(dropout_layer)

        # A Sequential object runs each of the modules contained within it, in a sequential manner. This is a simpler
        # way of writing our neural network.
        self.block = nn.Sequential(*block)
        self.out_channels = out_channels

    def forward(self, x):
        return self.block(x)
