from typing import Optional
import torch.nn as nn
from .conv import ConvolutionalBlock
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            dimensions: int,
            pooling_type: str,
            num_encoding_blocks: int,
            normalization: Optional[str],
            # preactivation: bool = False,
            # residual: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            # initial_dilation: Optional[int] = None,
            dropout: float = 0.3,
            all_size_input: bool = False,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        # self.dilation = initial_dilation
        is_first_block = True
        for i in range(num_encoding_blocks):  # 3
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
                # preactivation,
                is_first_block=is_first_block,
                # residual=residual,
                padding=2,
                padding_mode=padding_mode,
                activation=activation,
                # dilation=self.dilation,
                dropout=dropout,
                num_block=i,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:  # ?
                in_channels = out_channels_first
                out_channels_first = in_channels * 2

            # dilation is always None
            # if self.dilation is not None:
            #     self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            dimensions: int,
            normalization: Optional[str],
            pooling_type: str = None,
            # preactivation: bool = False,
            is_first_block: bool = False,
            # residual: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            # dilation: Optional[int] = None,
            dropout: float = 0.3,
            num_block: int = 0,
    ):
        super().__init__()

        self.num_block = num_block
        # self.preactivation = preactivation
        # self.normalization = normalization

        # self.residual = residual

        # if is_first_block:
        # normalization = None
        # preactivation = None
        # else:
        # normalization = self.normalization
        # normalization = None
        # preactivation = self.preactivation

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            # preactivation=preactivation,
            padding=2,
            padding_mode=padding_mode,
            activation=activation,
            # dilation=dilation,
            dropout=dropout,
        )

        out_channels_second = out_channels_first

        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=normalization,
            # preactivation=self.preactivation,
            padding=2,
            activation=activation,
            # dilation=dilation,
            dropout=dropout,
        )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
        return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(
        dimensions: int,
        pooling_type: str,
        kernel_size: int = 2,
        stride: int = 2,
) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)