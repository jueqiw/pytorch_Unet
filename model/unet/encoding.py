from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from .conv import ConvolutionalBlock


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            pooling_type: str,
            num_encoding_blocks: int,
            normalization: Optional[str],
            preactivation: bool = False,
            # residual: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
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
                in_channels=in_channels,
                out_channels_first=out_channels_first,
                normalization=normalization,
                pooling_type=pooling_type,
                is_first_block=is_first_block,
                # residual=residual,
                padding=2,
                padding_mode=padding_mode,
                # dilation=self.dilation,
                dropout=dropout,
                all_size_input=all_size_input,
                num_block=i,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            in_channels = out_channels_first
            out_channels_first = in_channels * 2
            self.out_channels = self.encoding_blocks[-1].out_channels

            # dilation is always None
            # if self.dilation is not None:
            #     self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x


class EncodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            normalization: Optional[str],
            pooling_type: str = None,
            is_first_block: bool = False,
            # residual: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
            # dilation: Optional[int] = None,
            dropout: float = 0.3,
            all_size_input: bool = False,
            num_block: int = 0,
    ):
        super().__init__()

        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels,
            out_channels=out_channels_first,
            normalization=normalization,
            padding=2,
            padding_mode=padding_mode,
            # dilation=dilation,
            dropout=dropout,
        )

        self.conv2 = ConvolutionalBlock(
            in_channels=out_channels_first,
            out_channels=out_channels_first,
            normalization=normalization,
            padding=2,
            # dilation=dilation,
            dropout=dropout,
        )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(pooling_type)

        self.all_size_input = all_size_input
        self.num_block = num_block
        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            if self.all_size_input:
                # (padding_left,padding_right, \text{padding\_top}, \text{padding\_bottom}padding_top,padding_bottom
                # \text{padding\_front}, \text{padding\_back})padding_front,padding_back)
                pd = (1, 0, 1, 0, 1, 0)
                if self.num_block % 2 == 0:
                    pd = (0, 1, 0, 1, 0, 1)
                # could only use mode = "constant" because the reflection is only used for the last dimension
                x = F.pad(x, pd, mode='constant', value=0)
        return x, skip_connection


def get_downsampling_layer(
        pooling_type: str,
        kernel_size: int = 2,
        stride: int = 2,
) -> nn.Module:
    class_name = '{}Pool3d'.format(pooling_type.capitalize())
    class_ = getattr(nn, class_name)
    return class_(kernel_size)
