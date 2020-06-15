from typing import Optional

from torch.nn import Conv3d, BatchNorm3d, GroupNorm, Dropout3d
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            normalization: Optional[str] = None,
            kernel_size: int = 5,
            activation: Optional[str] = 'ReLU',
            # preactivation: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            dilation: Optional[int] = None,
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
        if normalization is not None:
            if normalization == 'Batch':
                # num_features = in_channels if preactivation else out_channels
                norm_layer = BatchNorm3d(out_channels)
            elif normalization == 'Group':
                # num_features = in_channels if preactivation else out_channels
                norm_layer = GroupNorm(num_groups=1, num_channels=out_channels)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        # if preactivation:
        #     self.add_if_not_none(block, norm_layer)
        #     self.add_if_not_none(block, activation_layer)
        #     self.add_if_not_none(block, conv_layer)
        # else:
        self.add_if_not_none(block, conv_layer)
        self.add_if_not_none(block, norm_layer)
        self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            dropout_layer = Dropout3d(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        # self.dropout_layer = dropout_layer

        # A Sequential object runs each of the modules contained within it, in a sequential manner. This is a simpler
        # way of writing our neural network.
        self.block = nn.Sequential(*block)

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)
